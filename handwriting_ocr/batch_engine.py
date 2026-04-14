"""
handwriting_ocr.batch_engine
=============================
High-throughput, memory-efficient batch processing engine.

Design goals
------------
* **Flat memory footprint** — file paths are yielded from a generator;
  images are never all held in RAM simultaneously.
* **All CPU cores used** — preprocessing (pure OpenCV/NumPy) is the
  bottleneck and is embarrassingly parallel; it runs inside a
  ``ProcessPoolExecutor`` of ``os.cpu_count()`` workers by default.
* **EasyOCR model loaded once** — the GPU model lives in the *main*
  process only.  Worker processes handle the CPU-bound preprocessing
  stage and return NumPy arrays back to the main process, which then
  feeds them to EasyOCR in one call.  This avoids deserialising the
  entire PyTorch model into every subprocess.
* **Fault isolation** — a corrupt image causes the worker to return an
  error payload; the batch continues unaffected.

Architecture diagram (single-machine)
--------------------------------------

    ┌────────────────────────────────────────────────────────────┐
    │                       MAIN PROCESS                         │
    │                                                            │
    │  _file_path_generator()  ──►  ProcessPoolExecutor          │
    │       (generator)             (N worker procs)             │
    │                                    │                       │
    │                          _preprocess_worker()              │
    │                          (stateless; returns ndarray)      │
    │                                    │                       │
    │            ◄──── preprocessed arrays (via Future) ─────── │
    │                                    │                       │
    │          reader.readtext(array)  ◄─┘                       │
    │          (EasyOCR, GPU)                                    │
    │                                    │                       │
    │           yields OCRResult ──────► caller                  │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from handwriting_ocr.preprocessor import ImagePreprocessor
from handwriting_ocr.types import Detection, OCRResult

logger = logging.getLogger(__name__)

# File extensions treated as image files.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
)


# ---------------------------------------------------------------------------
# Module-level worker function
# ---------------------------------------------------------------------------
# IMPORTANT: This function must live at module scope (not inside a class or
# nested function) so that ``pickle`` can serialise it for the subprocess.

def _preprocess_worker(
    file_path: str,
    preprocessor_kwargs: Dict,
) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
    """Load one image from disk and apply the preprocessing pipeline.

    Designed to run inside a child process spawned by
    ``ProcessPoolExecutor``.  All arguments and return values must be
    picklable.

    Args:
        file_path: Absolute path to the image file.
        preprocessor_kwargs: Keyword arguments forwarded to
            :class:`~handwriting_ocr.preprocessor.ImagePreprocessor`.

    Returns:
        A 3-tuple ``(file_path, preprocessed_array, error_message)``.
        If processing succeeds, ``error_message`` is ``None`` and
        ``preprocessed_array`` is a valid ``uint8`` NumPy array.
        If it fails, ``preprocessed_array`` is ``None`` and
        ``error_message`` describes the failure.
    """
    try:
        raw = cv2.imread(file_path)
        if raw is None:
            return file_path, None, f"cv2.imread returned None for '{file_path}'"
        if preprocessor_kwargs:
            preprocessor = ImagePreprocessor(**preprocessor_kwargs)
            processed = preprocessor.process(raw)
        else:
            # Preprocessing disabled — just return the original BGR image.
            processed = raw
        return file_path, processed, None
    except Exception as exc:  # noqa: BLE001
        return file_path, None, str(exc)


def _load_only_worker(
    file_path: str,
) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
    """Load one image without any preprocessing (preprocessing=False path).

    Args:
        file_path: Absolute path to the image file.

    Returns:
        Same 3-tuple contract as :func:`_preprocess_worker`.
    """
    try:
        raw = cv2.imread(file_path)
        if raw is None:
            return file_path, None, f"cv2.imread returned None for '{file_path}'"
        return file_path, raw, None
    except Exception as exc:  # noqa: BLE001
        return file_path, None, str(exc)


# ---------------------------------------------------------------------------
# BatchEngine class
# ---------------------------------------------------------------------------

class BatchEngine:
    """High-throughput engine for processing large directories of images.

    The engine owns the parallelism layer (subprocess pool) but delegates
    OCR to an externally supplied callable so it remains decoupled from
    any specific OCR backend.

    Args:
        ocr_callable: A callable with the signature
            ``(image: np.ndarray, **kwargs) -> List[Tuple]``
            that accepts a single image and returns EasyOCR-style results.
            In practice this will be ``reader.readtext``.
        max_workers: Number of parallel worker processes for preprocessing.
            Defaults to ``os.cpu_count()``.
        chunk_size: How many futures to submit at once before draining
            completed results.  Keeps the job queue from growing without
            bound when processing millions of files.
        preprocessor_kwargs: Keyword arguments forwarded to
            :class:`~handwriting_ocr.preprocessor.ImagePreprocessor`.
            Pass an empty dict to skip preprocessing.
    """

    def __init__(
        self,
        ocr_callable,  # type: ignore[type-arg]
        max_workers: Optional[int] = None,
        chunk_size: int = 512,
        preprocessor_kwargs: Optional[Dict] = None,
    ) -> None:
        self._ocr = ocr_callable
        self._max_workers = max_workers or os.cpu_count() or 1
        self._chunk_size = chunk_size
        self._preprocessor_kwargs: Dict = preprocessor_kwargs or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        languages: Optional[List[str]] = None,
        detail: int = 1,
        batch_size: int = 8,
    ) -> Generator[OCRResult, None, None]:
        """Stream OCRResult objects for every image in *directory*.

        Memory footprint is ``O(chunk_size)`` images, not ``O(total_files)``.
        Each result is yielded as soon as it is ready so the caller can
        begin consuming results before the full batch is done.

        Args:
            directory: Root directory to scan for images.
            recursive: If ``True`` (default), walk sub-directories.
            languages: List of language codes passed to EasyOCR (e.g.
                ``["en"]``).  Defaults to ``["en"]``.
            detail: EasyOCR ``detail`` flag — ``1`` returns bounding boxes
                and confidence, ``0`` text-only.
            batch_size: EasyOCR internal batch size.  Larger values consume
                more VRAM but process images faster on GPU.

        Yields:
            One :class:`~handwriting_ocr.types.OCRResult` per image,
            in roughly the order futures complete (not strictly the
            filesystem order).

        Raises:
            FileNotFoundError: If *directory* does not exist.
            NotADirectoryError: If *directory* is a file, not a directory.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        languages = languages or ["en"]
        use_preprocessing = bool(self._preprocessor_kwargs)
        worker_fn = _preprocess_worker if use_preprocessing else _load_only_worker

        path_gen = self._file_path_generator(directory, recursive)

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            pending: Dict[Future, str] = {}

            def _submit_next() -> bool:
                """Submit one path from the generator; return False when exhausted."""
                try:
                    path = next(path_gen)
                    if use_preprocessing:
                        fut = executor.submit(
                            worker_fn,
                            path,
                            self._preprocessor_kwargs,
                        )
                    else:
                        fut = executor.submit(worker_fn, path)
                    pending[fut] = path
                    return True
                except StopIteration:
                    return False

            # Prime the pool with up to chunk_size futures.
            exhausted = False
            for _ in range(self._chunk_size):
                if not _submit_next():
                    exhausted = True
                    break

            while pending:
                done_futures = list(as_completed(list(pending.keys())))
                for fut in done_futures:
                    if fut not in pending:
                        continue
                    pending.pop(fut)

                    # Try to keep the pool fully loaded.
                    if not exhausted:
                        exhausted = not _submit_next()

                    file_path_str, processed_img, err = fut.result()
                    if err is not None:
                        logger.error(
                            "Preprocessing failed for '%s': %s",
                            file_path_str,
                            err,
                        )
                        yield OCRResult(
                            source=file_path_str,
                            success=False,
                            error=err,
                            preprocessing_applied=use_preprocessing,
                        )
                        continue

                    yield self._run_ocr_on_array(
                        image=processed_img,
                        source=file_path_str,
                        preprocessing_applied=use_preprocessing,
                        languages=languages,
                        detail=detail,
                        batch_size=batch_size,
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _file_path_generator(
        directory: Path,
        recursive: bool,
    ) -> Generator[str, None, None]:
        """Yield absolute string paths for all image files under *directory*.

        Args:
            directory: Root directory to scan.
            recursive: Whether to descend into sub-directories.

        Yields:
            Absolute string path for each discovered image file.
        """
        iterator = directory.rglob("*") if recursive else directory.glob("*")
        for entry in iterator:
            if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
                yield str(entry.resolve())

    def _run_ocr_on_array(
        self,
        image: np.ndarray,
        source: str,
        preprocessing_applied: bool,
        languages: List[str],
        detail: int,
        batch_size: int,
    ) -> OCRResult:
        """Run EasyOCR on a preprocessed NumPy array.

        Args:
            image: Preprocessed image array.
            source: Original file path (for error reporting).
            preprocessing_applied: Whether preprocessing was active.
            languages: Language codes for EasyOCR.
            detail: EasyOCR detail level.
            batch_size: EasyOCR batch size.

        Returns:
            A fully populated :class:`~handwriting_ocr.types.OCRResult`.
        """
        try:
            raw_results = self._ocr(
                image,
                detail=detail,
                batch_size=batch_size,
            )
            detections = _parse_easyocr_results(raw_results, detail)
            full_text = "\n".join(d.text for d in detections)
            return OCRResult(
                source=source,
                success=True,
                full_text=full_text,
                detections=detections,
                preprocessing_applied=preprocessing_applied,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR failed for '%s': %s", source, exc, exc_info=True)
            return OCRResult(
                source=source,
                success=False,
                error=str(exc),
                preprocessing_applied=preprocessing_applied,
            )


# ---------------------------------------------------------------------------
# Shared helper — parse raw EasyOCR output into Detection objects
# ---------------------------------------------------------------------------

def _parse_easyocr_results(
    raw: list,
    detail: int,
) -> List[Detection]:
    """Convert EasyOCR's native list output into typed :class:`Detection` objects.

    EasyOCR returns different formats depending on the ``detail`` flag:

    * ``detail=1``: ``[(bbox, text, confidence), ...]``
    * ``detail=0``: ``[text, ...]``

    Args:
        raw: The list returned by ``reader.readtext()``.
        detail: The ``detail`` value that was passed to EasyOCR.

    Returns:
        List of :class:`~handwriting_ocr.types.Detection` objects.
    """
    detections: List[Detection] = []
    for item in raw:
        if detail == 1:
            bbox, text, confidence = item
            detections.append(
                Detection(
                    text=str(text),
                    confidence=float(confidence),
                    bounding_box=[[int(x), int(y)] for x, y in bbox],
                )
            )
        else:
            detections.append(
                Detection(
                    text=str(item),
                    confidence=1.0,  # detail=0 provides no confidence score
                )
            )
    return detections

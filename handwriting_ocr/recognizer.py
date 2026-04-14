"""
handwriting_ocr.recognizer
===========================
The main public API class for the HandWriting-OCR engine.

Typical usage
-------------
.. code-block:: python

    from handwriting_ocr import HandwritingRecognizer

    # Instantiate once — EasyOCR loads its model at this point.
    recognizer = HandwritingRecognizer(languages=["en"])

    # Recognise a single file (returns OCRResult).
    result = recognizer.recognize_file("photo.jpg")
    print(result.full_text)

    # Recognise a directory of thousands of images (generator).
    for result in recognizer.process_directory("./scans/"):
        if result.success:
            print(result.source, "→", result.full_text[:80])

    # Recognise raw bytes from a network request or GUI widget.
    with open("photo.jpg", "rb") as fh:
        result = recognizer.recognize_bytes(fh.read())
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Union

import cv2
import numpy as np

from handwriting_ocr.batch_engine import BatchEngine, _parse_easyocr_results
from handwriting_ocr.preprocessor import ImagePreprocessor
from handwriting_ocr.types import BatchSummary, Detection, OCRResult

logger = logging.getLogger(__name__)


class HandwritingRecognizer:
    """End-to-end handwriting OCR pipeline.

    This class is the single entry-point for all recognition work.  It owns
    the EasyOCR ``Reader`` (loaded once in ``__init__``), the preprocessing
    pipeline, and the high-throughput batch engine.  It is deliberately
    free of any ``print()`` calls or UI logic so it can be attached to a
    desktop GUI, a REST API, or used as a plain library without modification.

    Args:
        languages: List of BCP-47 language codes to pass to EasyOCR.
            Examples: ``["en"]``, ``["en", "ar"]``, ``["ch_sim", "en"]``.
            Defaults to ``["en"]``.
        gpu: Explicitly enable (``True``) or disable (``False``) GPU
            acceleration.  When ``None`` (default), the class auto-detects
            PyTorch CUDA availability and uses GPU if present.
        max_workers: Number of CPU worker processes for batch preprocessing.
            ``None`` defaults to ``os.cpu_count()``.
        batch_size: EasyOCR internal batch size used during batch runs.
            Larger values are faster on GPU but consume more VRAM.
            Defaults to ``8``.
        preprocessor_kwargs: Keyword arguments forwarded to
            :class:`~handwriting_ocr.preprocessor.ImagePreprocessor`.
            Pass ``None`` (default) to use sensible defaults.
        easyocr_kwargs: Additional keyword arguments forwarded verbatim to
            ``easyocr.Reader()``.  Useful for setting ``model_storage_directory``,
            ``download_enabled``, etc.

    Raises:
        ImportError: If ``easyocr`` is not installed in the current
            environment.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: Optional[bool] = None,
        max_workers: Optional[int] = None,
        batch_size: int = 8,
        preprocessor_kwargs: Optional[Dict] = None,
        easyocr_kwargs: Optional[Dict] = None,
    ) -> None:
        try:
            import easyocr  # noqa: PLC0415 (local import intentional)
        except ImportError as exc:
            raise ImportError(
                "easyocr is not installed.  Run: pip install easyocr"
            ) from exc

        self._languages: List[str] = languages or ["en"]
        self._batch_size = batch_size

        # ---------- GPU auto-detection ----------------------------------
        use_gpu = self._resolve_gpu(gpu)
        logger.info(
            "Initialising EasyOCR with languages=%s, gpu=%s",
            self._languages,
            use_gpu,
        )

        # ---------- Load model (expensive — done only once) -------------
        t0 = time.perf_counter()
        extra = easyocr_kwargs or {}
        self._reader = easyocr.Reader(
            self._languages,
            gpu=use_gpu,
            **extra,
        )
        elapsed = time.perf_counter() - t0
        logger.info("EasyOCR model loaded in %.2fs (gpu=%s)", elapsed, use_gpu)

        # ---------- Preprocessing pipeline ------------------------------
        _pp_kwargs: Dict = preprocessor_kwargs if preprocessor_kwargs is not None else {}
        self._preprocessor_kwargs = _pp_kwargs
        # Build a local preprocessor for single-image calls.
        self._preprocessor: Optional[ImagePreprocessor] = (
            ImagePreprocessor(**_pp_kwargs) if _pp_kwargs else ImagePreprocessor()
        )

        # ---------- Batch engine ----------------------------------------
        self._batch_engine = BatchEngine(
            ocr_callable=self._reader.readtext,
            max_workers=max_workers,
            preprocessor_kwargs=_pp_kwargs,
        )

    # ------------------------------------------------------------------
    # Phase 3 — Single-image recognition
    # ------------------------------------------------------------------

    def recognize_file(
        self,
        file_path: Union[str, Path],
        preprocess: bool = True,
        detail: int = 1,
    ) -> OCRResult:
        """Recognise text in a single image file.

        Args:
            file_path: Path to the image (JPEG, PNG, BMP, TIFF, WebP…).
            preprocess: Apply the OpenCV preprocessing pipeline before OCR.
                Setting this to ``False`` passes the raw decoded image
                directly to EasyOCR, which can be useful for already-clean
                digital scans.
            detail: EasyOCR ``detail`` flag — ``1`` (default) returns
                bounding boxes and per-word confidence scores; ``0`` returns
                text strings only and is slightly faster.

        Returns:
            A fully populated :class:`~handwriting_ocr.types.OCRResult`.
            The ``success`` field is ``False`` (and ``error`` is set) if
            loading or recognition fails; the exception is *not* re-raised.
        """
        path = Path(file_path)
        source = str(path.resolve())

        try:
            image = self._load_image(source)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load image '%s': %s", source, exc)
            return OCRResult(source=source, success=False, error=str(exc))

        return self._recognize_array(
            image=image,
            source=source,
            preprocess=preprocess,
            detail=detail,
        )

    def recognize_bytes(
        self,
        data: bytes,
        preprocess: bool = True,
        detail: int = 1,
        source_label: str = "<bytes>",
    ) -> OCRResult:
        """Recognise text in an image supplied as raw bytes.

        Useful when receiving images from an HTTP multipart upload, a Qt
        widget pixel buffer, or any other in-memory source that does not
        correspond to a file on disk.

        Args:
            data: Raw image bytes (e.g. the contents of a JPEG file).
            preprocess: Apply the OpenCV preprocessing pipeline.
            detail: EasyOCR ``detail`` flag.
            source_label: Display name used in :class:`~handwriting_ocr.types.OCRResult`
                to identify this input (default ``"<bytes>"``).

        Returns:
            A fully populated :class:`~handwriting_ocr.types.OCRResult`.
        """
        try:
            nparr = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imdecode returned None — invalid image bytes.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to decode image bytes: %s", exc)
            return OCRResult(source=source_label, success=False, error=str(exc))

        return self._recognize_array(
            image=image,
            source=source_label,
            preprocess=preprocess,
            detail=detail,
        )

    def recognize_array(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        detail: int = 1,
        source_label: str = "<array>",
    ) -> OCRResult:
        """Recognise text in an image already loaded as a NumPy array.

        Convenient when the caller has already decoded the image (e.g.
        grabbed a webcam frame with OpenCV).

        Args:
            image: BGR image array as returned by ``cv2.imread()``.
            preprocess: Apply the OpenCV preprocessing pipeline.
            detail: EasyOCR ``detail`` flag.
            source_label: Display name used in the result.

        Returns:
            A fully populated :class:`~handwriting_ocr.types.OCRResult`.
        """
        return self._recognize_array(
            image=image,
            source=source_label,
            preprocess=preprocess,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Phase 4 — Batch / directory processing
    # ------------------------------------------------------------------

    def process_directory(
        self,
        directory: Union[str, Path],
        preprocess: bool = True,
        recursive: bool = True,
        detail: int = 1,
    ) -> Generator[OCRResult, None, None]:
        """Process every image in *directory* using all available CPU cores.

        This is the "millions of images" entry-point.  It uses a generator
        internally so memory usage scales with ``chunk_size`` (default 512),
        not with the total number of files.  Results are yielded as soon as
        they are ready.

        .. code-block:: python

            recognizer = HandwritingRecognizer()
            for result in recognizer.process_directory("/data/scans"):
                save_to_database(result.to_dict())

        Args:
            directory: Root directory to scan.
            preprocess: Apply the OpenCV pipeline to each image before OCR.
            recursive: Descend into sub-directories (default ``True``).
            detail: EasyOCR ``detail`` flag.

        Yields:
            One :class:`~handwriting_ocr.types.OCRResult` per image in
            approximately the order futures complete.

        Raises:
            FileNotFoundError: If *directory* does not exist.
        """
        directory = Path(directory)

        # Temporarily override the engine's preprocessing behaviour
        # based on the caller's ``preprocess`` flag.
        if preprocess:
            engine = self._batch_engine
        else:
            # Swap to an engine configured with no preprocessing.
            engine = BatchEngine(
                ocr_callable=self._reader.readtext,
                max_workers=self._batch_engine._max_workers,
                preprocessor_kwargs={},
            )

        yield from engine.process_directory(
            directory=directory,
            recursive=recursive,
            languages=self._languages,
            detail=detail,
            batch_size=self._batch_size,
        )

    def process_directory_as_summary(
        self,
        directory: Union[str, Path],
        preprocess: bool = True,
        recursive: bool = True,
        detail: int = 1,
    ) -> BatchSummary:
        """Like :meth:`process_directory` but collects all results into a
        :class:`~handwriting_ocr.types.BatchSummary`.

        Use this when the dataset fits in memory and you want a single
        object to inspect afterwards.  For very large datasets, prefer
        the generator form :meth:`process_directory`.

        Args:
            directory: Root directory to scan.
            preprocess: Apply the OpenCV pipeline.
            recursive: Descend into sub-directories.
            detail: EasyOCR ``detail`` flag.

        Returns:
            A :class:`~handwriting_ocr.types.BatchSummary` with aggregate
            statistics and all individual results.
        """
        summary = BatchSummary()
        for result in self.process_directory(
            directory=directory,
            preprocess=preprocess,
            recursive=recursive,
            detail=detail,
        ):
            summary.total_files += 1
            if result.success:
                summary.succeeded += 1
            else:
                summary.failed += 1
            summary.results.append(result)
        return summary

    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        preprocess: bool = True,
        detail: int = 1,
    ) -> Iterator[OCRResult]:
        """Recognise text in an explicit list of image files.

        Args:
            file_paths: Ordered list of paths to process.
            preprocess: Apply the OpenCV pipeline.
            detail: EasyOCR ``detail`` flag.

        Yields:
            One :class:`~handwriting_ocr.types.OCRResult` per file,
            in the same order as *file_paths*.
        """
        for path in file_paths:
            yield self.recognize_file(path, preprocess=preprocess, detail=detail)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recognize_array(
        self,
        image: np.ndarray,
        source: str,
        preprocess: bool,
        detail: int,
    ) -> OCRResult:
        """Core recognition routine shared by all single-image public methods.

        Args:
            image: BGR image array.
            source: Identifier string for error reporting.
            preprocess: Whether to apply the preprocessing pipeline.
            detail: EasyOCR ``detail`` flag.

        Returns:
            :class:`~handwriting_ocr.types.OCRResult`.
        """
        t_start = time.perf_counter()

        try:
            if preprocess and self._preprocessor is not None:
                processed = self._preprocessor.process(image)
                preprocessing_applied = True
            else:
                processed = image
                preprocessing_applied = False

            raw_results = self._reader.readtext(
                processed,
                detail=detail,
                batch_size=self._batch_size,
            )
            detections = _parse_easyocr_results(raw_results, detail)
            full_text = "\n".join(d.text for d in detections)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            logger.debug(
                "OCR OK  source='%s'  detections=%d  time=%.1fms",
                source,
                len(detections),
                elapsed_ms,
            )

            return OCRResult(
                source=source,
                success=True,
                full_text=full_text,
                detections=detections,
                preprocessing_applied=preprocessing_applied,
                metadata={"processing_time_ms": round(elapsed_ms, 2)},
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("OCR failed for '%s': %s", source, exc, exc_info=True)
            return OCRResult(
                source=source,
                success=False,
                error=str(exc),
                preprocessing_applied=preprocess,
            )

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Read an image from disk via OpenCV.

        Args:
            path: Absolute path to the image file.

        Returns:
            BGR ``uint8`` NumPy array.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If OpenCV cannot decode the file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Image file not found: '{path}'")
        image = cv2.imread(path)
        if image is None:
            raise ValueError(
                f"OpenCV could not decode the image at '{path}'.  "
                "The file may be corrupt or in an unsupported format."
            )
        return image

    @staticmethod
    def _resolve_gpu(gpu: Optional[bool]) -> bool:
        """Determine whether to instruct EasyOCR to use GPU.

        Args:
            gpu: Caller preference.  ``None`` triggers auto-detection.

        Returns:
            ``True`` if GPU should be used, ``False`` otherwise.
        """
        if gpu is not None:
            return gpu
        try:
            import torch  # noqa: PLC0415

            available = torch.cuda.is_available()
            if available:
                device_name = torch.cuda.get_device_name(0)
                logger.info("CUDA GPU detected: %s — enabling GPU mode.", device_name)
            else:
                logger.info("No CUDA GPU detected — falling back to CPU.")
            return available
        except ImportError:
            logger.warning(
                "PyTorch not found.  Cannot auto-detect GPU.  Falling back to CPU."
            )
            return False

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"HandwritingRecognizer("
            f"languages={self._languages!r}, "
            f"batch_size={self._batch_size})"
        )

"""
example_usage.py
================
Stand-alone demonstration of the HandWriting-OCR engine.

Run this file directly to see every public API surface exercised:

    python example_usage.py

No GUI, no web server — just the raw engine.  Attach your preferred
front-end on top of these calls.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configure before importing the engine so all module loggers
# inherit this configuration.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,                          # Change to DEBUG for verbose output.
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),       # Console
        logging.FileHandler("ocr_run.log", encoding="utf-8"),      # Persistent log file (UTF-8)
    ],
)
logger = logging.getLogger(__name__)

from handwriting_ocr import HandwritingRecognizer  # noqa: E402 (after logging setup)


# ---------------------------------------------------------------------------
# Helper — pretty-print an OCRResult as indented JSON
# ---------------------------------------------------------------------------

def print_result(result) -> None:
    """Dump an OCRResult as indented JSON to stdout."""
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Demo 1 — Single file (preprocessing ON)
# ---------------------------------------------------------------------------

def demo_single_file(recognizer: HandwritingRecognizer, image_path: str) -> None:
    """Recognise a single image with preprocessing enabled."""
    logger.info("── Demo 1: Single file, preprocessing=True ──")
    result = recognizer.recognize_file(image_path, preprocess=True)
    print_result(result)


# ---------------------------------------------------------------------------
# Demo 2 — Single file (preprocessing OFF — raw image to EasyOCR)
# ---------------------------------------------------------------------------

def demo_single_file_no_preprocess(
    recognizer: HandwritingRecognizer, image_path: str
) -> None:
    """Recognise a single image WITHOUT the OpenCV pipeline."""
    logger.info("── Demo 2: Single file, preprocessing=False ──")
    result = recognizer.recognize_file(image_path, preprocess=False)
    print_result(result)


# ---------------------------------------------------------------------------
# Demo 3 — Raw bytes input (simulates receiving data from a web upload)
# ---------------------------------------------------------------------------

def demo_bytes_input(recognizer: HandwritingRecognizer, image_path: str) -> None:
    """Load an image as bytes and pass it to the recognizer."""
    logger.info("── Demo 3: Raw bytes input ──")
    with open(image_path, "rb") as fh:
        raw_bytes = fh.read()
    result = recognizer.recognize_bytes(
        raw_bytes,
        preprocess=True,
        source_label=f"upload::{Path(image_path).name}",
    )
    print_result(result)


# ---------------------------------------------------------------------------
# Demo 4 — NumPy array input (simulates a live camera frame)
# ---------------------------------------------------------------------------

def demo_array_input(recognizer: HandwritingRecognizer, image_path: str) -> None:
    """Pass a pre-loaded NumPy array to the recognizer."""
    import cv2  # noqa: PLC0415

    logger.info("── Demo 4: NumPy array input (simulated camera frame) ──")
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Could not load %s for array demo.", image_path)
        return
    result = recognizer.recognize_array(frame, preprocess=True, source_label="camera_frame")
    print_result(result)


# ---------------------------------------------------------------------------
# Demo 5 — Batch directory processing (high-throughput)
# ---------------------------------------------------------------------------

def demo_directory(recognizer: HandwritingRecognizer, directory: str) -> None:
    """Stream results from an entire directory of images."""
    logger.info("── Demo 5: Directory batch processing ──")
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("'%s' is not a directory — skipping directory demo.", directory)
        return

    # Collect into a BatchSummary for convenient aggregate stats.
    summary = recognizer.process_directory_as_summary(
        directory=dir_path,
        preprocess=True,
        recursive=True,
    )
    logger.info(
        "Batch complete: %d/%d succeeded (%.1f%%)",
        summary.succeeded,
        summary.total_files,
        summary.success_rate * 100,
    )
    # Print each result.
    for result in summary.results:
        print_result(result)


# ---------------------------------------------------------------------------
# Demo 6 — Generator pattern (use for very large datasets)
# ---------------------------------------------------------------------------

def demo_generator_pattern(
    recognizer: HandwritingRecognizer, directory: str
) -> None:
    """Demonstrate the generator pattern for memory-safe large-scale processing."""
    logger.info("── Demo 6: Generator pattern (memory-safe streaming) ──")
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("'%s' is not a directory — skipping generator demo.", directory)
        return

    for i, result in enumerate(recognizer.process_directory(dir_path), start=1):
        if result.success:
            # In production you would stream this to a database, message queue, etc.
            logger.info(
                "[%d] [OK] %s → '%s'",
                i,
                Path(result.source).name,
                result.full_text[:60].replace("\n", " "),
            )
        else:
            logger.warning("[%d] [FAIL] %s — %s", i, Path(result.source).name, result.error)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all demos."""
    # ------------------------------------------------------------------
    # Create the recognizer ONCE — the EasyOCR model is heavy and must
    # not be re-instantiated for each image.
    # ------------------------------------------------------------------
    recognizer = HandwritingRecognizer(
        languages=["en"],
        # gpu=True / False / None (auto-detect)
        gpu=None,
        batch_size=8,
        # Customise the preprocessing pipeline:
        preprocessor_kwargs={
            "use_nlm_denoising": True,
            "nlm_h": 10,
            "adaptive_block_size": 15,
            "adaptive_C": 8,
            "apply_morphology": False,
            "target_dpi_scale": 1.0,
        },
    )
    logger.info("Recognizer ready: %r", recognizer)

    # ------------------------------------------------------------------
    # Point these paths at real image files / directories on your system.
    # ------------------------------------------------------------------
    SAMPLE_IMAGE = r"C:\Users\user\Pictures\handwriting-images\sample(1).jpg"          # Change to a real image path.
    SAMPLE_DIRECTORY = r"C:\Users\user\Pictures\handwriting-images" # Change to a real directory.

    # Check if sample file exists before running file-based demos.
    if Path(SAMPLE_IMAGE).exists():
        demo_single_file(recognizer, SAMPLE_IMAGE)
        demo_single_file_no_preprocess(recognizer, SAMPLE_IMAGE)
        demo_bytes_input(recognizer, SAMPLE_IMAGE)
        demo_array_input(recognizer, SAMPLE_IMAGE)
    else:
        logger.warning(
            "Sample image '%s' not found.  Skipping single-file demos.  "
            "Update the SAMPLE_IMAGE variable in example_usage.py.",
            SAMPLE_IMAGE,
        )

    # Run directory demos if directory exists.
    demo_directory(recognizer, SAMPLE_DIRECTORY)
    demo_generator_pattern(recognizer, SAMPLE_DIRECTORY)

    logger.info("All demos complete.  See 'ocr_run.log' for the full log.")


if __name__ == "__main__":
    # Required on Windows when using multiprocessing inside a __main__ guard.
    # Without this, spawned sub-processes will attempt to re-run __main__
    # and create an infinite fork storm.
    import multiprocessing
    multiprocessing.freeze_support()

    main()

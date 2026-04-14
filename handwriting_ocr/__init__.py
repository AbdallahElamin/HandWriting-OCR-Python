"""
handwriting_ocr
===============
Public surface of the HandWriting-OCR engine.

Import the main API class directly from this package:

    from handwriting_ocr import HandwritingRecognizer

Everything else (preprocessor, batch engine, type definitions)
is implementation detail and lives in the sub-modules.
"""

from handwriting_ocr.recognizer import HandwritingRecognizer
from handwriting_ocr.types import OCRResult, BatchSummary

__all__ = ["HandwritingRecognizer", "OCRResult", "BatchSummary"]
__version__ = "1.0.0"

"""
handwriting_ocr.types
=====================
Shared data-transfer objects (DTOs) used across the entire engine.

All public types are plain dataclasses / TypedDicts so they can be
trivially serialised to JSON without any third-party dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-detection result (one bounding box / text line in a single image)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single piece of detected text within an image.

    Attributes:
        text: The raw string recognised by EasyOCR.
        confidence: Float in [0, 1] — model confidence for this detection.
        bounding_box: Four (x, y) corner coordinates returned by EasyOCR,
            ordered top-left → top-right → bottom-right → bottom-left.
            ``None`` when the source was raw bytes that lacked coordinate
            metadata.
    """

    text: str
    confidence: float
    bounding_box: Optional[List[Tuple[int, int]]] = None


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """Structured output produced for a single input image.

    Attributes:
        source: The original file path or ``"<bytes>"`` when the input was
            provided as raw bytes.
        success: ``True`` when OCR completed without error.
        full_text: All detected text lines joined by ``\\n``.  Empty string
            when ``success`` is ``False``.
        detections: Ordered list of individual :class:`Detection` objects.
        error: Human-readable error message; ``None`` on success.
        preprocessing_applied: Whether the OpenCV pipeline was active during
            this recognition run.
        metadata: Arbitrary extra key-value pairs a caller may attach
            (e.g. image dimensions, processing times).
    """

    source: str
    success: bool
    full_text: str = ""
    detections: List[Detection] = field(default_factory=list)
    error: Optional[str] = None
    preprocessing_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this result to a plain Python dictionary.

        Returns:
            A JSON-serialisable ``dict`` representation of this result.
        """
        return {
            "source": self.source,
            "success": self.success,
            "full_text": self.full_text,
            "detections": [
                {
                    "text": d.text,
                    "confidence": round(d.confidence, 6),
                    "bounding_box": d.bounding_box,
                }
                for d in self.detections
            ],
            "error": self.error,
            "preprocessing_applied": self.preprocessing_applied,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Batch-run summary
# ---------------------------------------------------------------------------

@dataclass
class BatchSummary:
    """Aggregate statistics for a directory batch-processing run.

    Attributes:
        total_files: Number of image files discovered.
        succeeded: Number of files processed without error.
        failed: Number of files that raised an exception.
        results: All individual :class:`OCRResult` objects in input order.
    """

    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    results: List[OCRResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Return fraction of files processed successfully (0–1)."""
        if self.total_files == 0:
            return 0.0
        return self.succeeded / self.total_files

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this summary to a plain Python dictionary.

        Returns:
            A JSON-serialisable ``dict`` including per-file results.
        """
        return {
            "total_files": self.total_files,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "results": [r.to_dict() for r in self.results],
        }

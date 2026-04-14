"""
handwriting_ocr.preprocessor
=============================
OpenCV-based image preprocessing pipeline for handwriting images.

The pipeline is intentionally designed as a *pure function* — it takes a
NumPy array and returns a NumPy array with no side-effects or I/O.  This
makes it trivially testable in isolation and safe to run inside worker
sub-processes spawned by the batch engine.

Pipeline stages (applied in order)
------------------------------------
1. **Grayscale conversion** — collapses RGB/RGBA to a single intensity
   channel, reducing noise and speeding up every subsequent operation.

2. **Non-Local Means denoising** — removes JPEG artefacts, paper grain, and
   scanner speckle while preserving edge sharpness far better than a simple
   Gaussian blur (the blur is still available as a lighter-weight fallback).

3. **Adaptive Gaussian thresholding** — binarises each local neighbourhood
   independently, so bright spots or shadowed areas on the page do not fool
   the global threshold into washing out ink or background.

4. **Morphological closing** *(optional)* — a small structuring element
   bridges tiny gaps in ink strokes caused by cheap pens or low-DPI scans,
   helping EasyOCR stitch broken characters back together.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Applies a configurable OpenCV pipeline to a handwriting image.

    All parameters are set once at construction time.  The resulting object
    is reusable and stateless — safe to share between threads.

    Args:
        use_nlm_denoising: When ``True`` (default), use Non-Local Means
            denoising (slower but higher quality).  When ``False``, fall back
            to a mild Gaussian blur which is faster and sufficient for clean
            scans.
        nlm_h: Filter strength for Non-Local Means.  Higher values remove
            more noise but may blur fine strokes.  Defaults to ``10``.
        adaptive_block_size: Neighbourhood size for adaptive thresholding.
            Must be an odd integer ≥ 3.  Defaults to ``15``.
        adaptive_C: Constant subtracted from the weighted mean in adaptive
            thresholding.  Tweak this if text appears broken (increase) or
            background bleeds in (decrease).  Defaults to ``8``.
        apply_morphology: When ``True``, apply a morphological *closing*
            operation after binarisation to connect broken strokes.
            Defaults to ``False`` (off by default to avoid over-connecting
            letters in Latin handwriting).
        morph_kernel_size: Size of the structuring element used for the
            morphological operation.  Defaults to ``(2, 2)``.
        target_dpi_scale: If > 1.0, rescale the image by this factor before
            processing.  Useful when the source DPI is very low (e.g. mobile
            photos).  Defaults to ``1.0`` (no rescaling).
    """

    def __init__(
        self,
        use_nlm_denoising: bool = True,
        nlm_h: int = 10,
        adaptive_block_size: int = 15,
        adaptive_C: int = 8,
        apply_morphology: bool = False,
        morph_kernel_size: Tuple[int, int] = (2, 2),
        target_dpi_scale: float = 1.0,
    ) -> None:
        if adaptive_block_size % 2 == 0 or adaptive_block_size < 3:
            raise ValueError(
                f"adaptive_block_size must be an odd integer ≥ 3, "
                f"got {adaptive_block_size}"
            )

        self.use_nlm_denoising = use_nlm_denoising
        self.nlm_h = nlm_h
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_C = adaptive_C
        self.apply_morphology = apply_morphology
        self.morph_kernel_size = morph_kernel_size
        self.target_dpi_scale = target_dpi_scale

        # Pre-build the morphological kernel to avoid re-creating it per call.
        self._morph_kernel: Optional[np.ndarray] = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                morph_kernel_size,
            )
            if apply_morphology
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on a single image.

        Args:
            image: Input image as a NumPy array, accepted in BGR (OpenCV
                default), RGB, or grayscale (2-D) format.  The array is
                **not** mutated; a new array is always returned.

        Returns:
            Preprocessed binary (or grayscale) image suitable for passing
            directly to EasyOCR as a NumPy array.

        Raises:
            ValueError: If ``image`` is ``None`` or has an unexpected shape.
        """
        if image is None or image.size == 0:
            raise ValueError("Received empty or None image array.")

        img = image.copy()

        # Stage 0 — optional upscale for low-DPI sources ----------------
        if self.target_dpi_scale != 1.0:
            img = self._rescale(img)

        # Stage 1 — Grayscale conversion ---------------------------------
        img = self._to_grayscale(img)
        logger.debug("Stage 1 (grayscale) complete: shape=%s", img.shape)

        # Stage 2 — Noise reduction --------------------------------------
        img = self._denoise(img)
        logger.debug("Stage 2 (denoising) complete.")

        # Stage 3 — Adaptive thresholding --------------------------------
        img = self._threshold(img)
        logger.debug("Stage 3 (thresholding) complete.")

        # Stage 4 — Morphological closing (optional) ---------------------
        if self.apply_morphology and self._morph_kernel is not None:
            img = self._morphological_close(img)
            logger.debug("Stage 4 (morphology) complete.")

        return img

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------

    def _rescale(self, image: np.ndarray) -> np.ndarray:
        """Upscale or downscale the image by ``target_dpi_scale``.

        Args:
            image: Source image array (any number of channels).

        Returns:
            Resized image array.
        """
        h, w = image.shape[:2]
        new_w = int(w * self.target_dpi_scale)
        new_h = int(h * self.target_dpi_scale)
        interpolation = (
            cv2.INTER_CUBIC
            if self.target_dpi_scale > 1.0
            else cv2.INTER_AREA
        )
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert an image of any channel depth to single-channel grayscale.

        Args:
            image: BGR, RGB, RGBA, or already-grayscale NumPy array.

        Returns:
            Single-channel ``uint8`` grayscale array.
        """
        if image.ndim == 2:
            # Already grayscale.
            return image
        if image.shape[2] == 4:
            # BGRA — strip alpha before converting.
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """Reduce noise using Non-Local Means or a Gaussian blur fallback.

        Args:
            gray: Single-channel grayscale image.

        Returns:
            Denoised single-channel image.
        """
        if self.use_nlm_denoising:
            # fastNlMeansDenoising works on single-channel images.
            return cv2.fastNlMeansDenoising(
                gray,
                None,
                h=self.nlm_h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        # Lightweight fallback: mild Gaussian blur.
        return cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive Gaussian thresholding to binarise the image.

        Adaptive thresholding computes a per-pixel threshold based on a
        local neighbourhood, making it robust to uneven lighting conditions
        that are common in handwriting photos.

        Args:
            gray: Denoised single-channel grayscale image.

        Returns:
            Binarised single-channel image (0 = background, 255 = ink).
        """
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.adaptive_block_size,
            C=self.adaptive_C,
        )
        return binary

    def _morphological_close(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological closing to bridge broken ink strokes.

        Closing = dilation followed immediately by erosion with the same
        kernel.  It fills small holes and gaps without significantly altering
        stroke width.

        Args:
            binary: Binarised single-channel image.

        Returns:
            Morphologically closed image.
        """
        return cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=1,
        )

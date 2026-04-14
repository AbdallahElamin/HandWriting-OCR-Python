# HandWriting-OCR

A **highly optimized, scalable, and maintainable** Python handwriting recognition engine.

Built with:
- **EasyOCR** — deep-learning OCR with CUDA GPU support
- **OpenCV** — five-stage image preprocessing pipeline
- **Python multiprocessing** — all available CPU cores used for preprocessing
- **Generator-based streaming** — flat memory footprint regardless of dataset size

---

## Project structure

```
HandWriting-OCR/
├── handwriting_ocr/
│   ├── __init__.py          # Public API surface
│   ├── types.py             # Data-transfer objects (OCRResult, BatchSummary, ...)
│   ├── preprocessor.py      # OpenCV preprocessing pipeline
│   ├── batch_engine.py      # High-throughput parallel processing engine
│   └── recognizer.py        # Main HandwritingRecognizer class
├── example_usage.py         # Runnable demos of every public method
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

### 1 — Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2 — Install PyTorch with GPU (CUDA) support **first**

> Skip this step if you only need CPU inference.

Check your CUDA version:
```powershell
nvidia-smi
```

Then install the matching PyTorch wheel:

| CUDA version | Command |
|---|---|
| 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |

Verify:
```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 3 — Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

---

## Quick start

```python
from handwriting_ocr import HandwritingRecognizer

# Model is loaded once here — this takes a few seconds on first run.
recognizer = HandwritingRecognizer(languages=["en"])

# Recognise a file
result = recognizer.recognize_file("photo.jpg")
print(result.full_text)
print(result.to_dict())   # JSON-serialisable dict

# Recognise raw bytes (e.g. from an HTTP upload)
with open("photo.jpg", "rb") as fh:
    result = recognizer.recognize_bytes(fh.read())

# Batch-process a directory (memory-safe generator)
for result in recognizer.process_directory("./scans/"):
    if result.success:
        print(result.source, "→", result.full_text)
    else:
        print("ERROR:", result.error)
```

Run the full demo:
```powershell
python example_usage.py
```

---

## OCRResult schema

Every recognition call returns an `OCRResult`:

| Field | Type | Description |
|---|---|---|
| `source` | `str` | File path or `"<bytes>"` / `"<array>"` |
| `success` | `bool` | `True` on success |
| `full_text` | `str` | All detected lines joined by `\n` |
| `detections` | `list[Detection]` | Per-word bounding box + text + confidence |
| `error` | `str \| None` | Error message if `success=False` |
| `preprocessing_applied` | `bool` | Whether the OpenCV pipeline ran |
| `metadata` | `dict` | `processing_time_ms`, etc. |

---

## API overview

### `HandwritingRecognizer(languages, gpu, batch_size, preprocessor_kwargs)`

| Method | Description |
|---|---|
| `recognize_file(path, preprocess)` | Single image from disk |
| `recognize_bytes(data, preprocess)` | Single image as raw bytes |
| `recognize_array(array, preprocess)` | Single image as NumPy array |
| `process_files(paths, preprocess)` | Iterator over an explicit list |
| `process_directory(dir, preprocess)` | Memory-safe generator over a directory |
| `process_directory_as_summary(dir)` | Like above but returns a `BatchSummary` |

### Toggling preprocessing per-call

```python
# With OpenCV pipeline (recommended for raw handwriting photos)
result = recognizer.recognize_file("scan.jpg", preprocess=True)

# Without — useful for clean digital documents
result = recognizer.recognize_file("digital.png", preprocess=False)
```

### Customising the preprocessing pipeline

```python
recognizer = HandwritingRecognizer(
    preprocessor_kwargs={
        "use_nlm_denoising": True,     # True = NLM (slow, high quality)
                                        # False = Gaussian blur (fast)
        "nlm_h": 10,                   # NLM filter strength (8–15 typical)
        "adaptive_block_size": 15,     # Must be odd ≥ 3
        "adaptive_C": 8,               # Higher = less noise in background
        "apply_morphology": True,      # Bridge broken ink strokes
        "morph_kernel_size": (2, 2),
        "target_dpi_scale": 1.5,       # Upscale low-DPI mobile photos
    }
)
```

---

## Preprocessing pipeline stages

```
Raw image
   │
   ▼ Stage 0  Optional upscale (target_dpi_scale > 1.0)
   │
   ▼ Stage 1  Grayscale conversion
   │
   ▼ Stage 2  Non-Local Means denoising  (or Gaussian blur fallback)
   │
   ▼ Stage 3  Adaptive Gaussian thresholding
   │
   ▼ Stage 4  Morphological closing  (optional)
   │
  EasyOCR
```

---

## Architecture: high-throughput batch engine

```
MAIN PROCESS
  │
  ├── _file_path_generator()  → yields paths (O(1) memory)
  │
  └── ProcessPoolExecutor (N workers = os.cpu_count())
        │
        ├── Worker 1: load image → preprocess → return ndarray
        ├── Worker 2: load image → preprocess → return ndarray
        └── Worker N: ...
              │
              ▼
        Main process: reader.readtext(ndarray)   ← GPU stays here
              │
              ▼
        yields OCRResult to caller
```

The GPU model **never crosses a process boundary** — it lives in the main
process only.  Workers handle the CPU-bound preprocessing stage.

---

## Windows note

Always protect multiprocessing code with the `if __name__ == "__main__":` guard:

```python
import multiprocessing
multiprocessing.freeze_support()   # required for PyInstaller executables

if __name__ == "__main__":
    recognizer = HandwritingRecognizer()
    ...
```

---

## Attaching a frontend

The engine is **completely decoupled from any interface**.  No `print()` calls exist
in the library code.  All output goes through Python's `logging` module.

| Frontend | Integration point |
|---|---|
| **Desktop GUI** (Tkinter, PyQt6) | Create `HandwritingRecognizer` in a background thread; call `recognize_file()` from a button handler; display `result.to_dict()`. |
| **Web API** (FastAPI, Flask) | Instantiate once at app startup; expose `recognize_bytes()` as a `POST /ocr` endpoint. |
| **CLI tool** | Wrap `process_directory()` with `argparse` and `tqdm`. |
| **Jupyter Notebook** | Import and call directly; display with `IPython.display`. |

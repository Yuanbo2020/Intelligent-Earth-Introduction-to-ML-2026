# Machine Learning Practicals — Setup Instructions (MT2026)

These four practicals use Jupyter notebooks. You may use **Google Colab** or a **local Python environment**. You do not need to create an environment in advance if you use Colab. Bring a laptop and check that you can open a notebook before the first practical.

## Option A — Google Colab

1. Open [Google Colab](https://colab.research.google.com/), then upload the notebook (`.ipynb`).
2. For **Day 3**, upload/extract the complete `Day3` folder, including `dnn_utils_v2.py`, `testCases_v4a.py`, and `images/`. The notebook and helper `.py` files must be in the current working directory.
3. For **Day 4**, upload/extract the complete `Day4` folder, including `prepare_cifar10_from_drive.py`. Before the dataset cell, install **gdown** if it is missing (`%pip install gdown`). A GPU runtime is helpful for optional long training experiments, but not required for the core tasks.
4. Colab storage is temporary: after a runtime reset, you may need to upload helper files and download data again.

## Option B — Local Python environment (recommended if you already use VS Code or PyCharm)

Install Python **3.11 or 3.13**. These notebook dependencies have been checked in a Python 3.13 environment; Python 3.11 is also a suitable course baseline.

In a terminal, open the folder containing `requirements.txt` and the `Day1`–`Day4` folders, then create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and register the Jupyter kernel:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ml-iearth-mt2026 --display-name "Python (ML Intelligent Earth)"
```

Open notebooks in Jupyter, VS Code or PyCharm and select **Python (ML Intelligent Earth)** as the kernel. For classic Jupyter Notebook from the terminal:

```bash
python -m notebook
```

If your system invokes Python through `python3` or `py`, use that command consistently instead of `python`.

## Quick environment check

Run the following in a fresh notebook cell:

```python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch, torchvision
import gdown
print('NumPy:', np.__version__, 'PyTorch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('GPU available:', torch.cuda.is_available())
plt.plot([0, 1], [0, 1]); plt.show()
```

If you see version numbers and a short line plot, the main packages work.

## Dataset preparation

- **Day 2:** MNIST downloads through the provided notebook code; allow time for first use.
- **Day 3:** Keep its helper scripts and `images/` folder next to the notebook; no external training dataset is required for the tasks.
- **Day 4:** The notebook will use this **course-provided Google Drive file** (not the default public torchvision download):

  https://drive.google.com/file/d/1yUOoLlV7yFkKx-gAYJTFbw6YPbIdumjq/view?usp=sharing

  Run the dataset setup cell once *before* the practical if possible. It calls `ensure_cifar10('data/')` from `Day4/prepare_cifar10_from_drive.py`, downloads the file only when needed, extracts standard CIFAR-10 Python-format ZIP or TAR archives, and validates the expected CIFAR-10 batch files with their published checksums. **Keep that `.py` file alongside the Day 4 notebook.**

  **Important:** The course Drive file must be publicly accessible to your account and contain the standard `cifar-10-batches-py` folder (e.g. within a ZIP or `.tar.gz`). If it is an unsupported format or its files do not pass validation, the notebook will show a clear error. Please contact the teaching team. Download speed depends on your network and file-sharing permissions.

  Optional CNN training may take hours on a laptop CPU. You can use a Colab GPU or run only 3–5 epochs as a short experiment. The full multi-epoch runs are not required for the core exercises.

## Common issues

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Install `requirements.txt` into the environment selected as your notebook kernel. |
| Day 3 `testCases_v4a` / `dnn_utils_v2` not found | Extract the **entire** Day 3 folder and run from that directory. |
| Day 4 Drive download failed | Confirm your connection and that the Drive link is accessible; run the dataset cell again. |
| Day 4 dataset file format/checksum error | The linked archive does not have the standard torchvision CIFAR-10 Python batch files. Ask the teaching team for the correct course archive. |
| Notebook cells fail at a TODO | The cell may be an intentionally incomplete **CORE** or **OPTIONAL** exercise; consult its Task instructions and the demonstrators. |
| Installation is very slow | PyTorch is a large download. Complete package installation before the practical. |

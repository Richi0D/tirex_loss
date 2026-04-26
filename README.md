# TiRex Loss

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Built with NXAI](https://img.shields.io/badge/Built_with-technology_from_NXAI-orange)

## Overview
This research explores following changes and methods on the TiRex model:
- Impact of different Loss functions
- Autoregressive generation methods


> **Note on Licensing:** This project's original code is licensed under **Apache 2.0**. However, it incorporates technology from the **NXAI TiRex** project, which is subject to the [NXAI Community License](LICENSE_NXAI)

## 📁 Project Structure
- **tirex/:** Git submodule containing the forked TiRex source code.
- **src/:** Research code and custom loss functions.
- **notebooks/:** Jupyter Notebooks with test results.
- **data/:** Research data used for results and testing.
- **report/:** Report of the research.
- **pyproject.toml:** Project metadata and dependency definitions.


## 🛠 Setup & Installation

This project uses [uv](https://docs.astral.sh/uv/) for reproducible dependency management. It also relies on a modified version of the **TiRex** library, included as a Git submodule.

### 1. Clone the Repository
Because this project uses submodules, you must clone recursively:
```bash
git clone --recursive https://github.com/Richi0D/tirex_loss.git
cd tirex_loss
```
*If you already cloned without the submodules, run:*
```
git submodule update --init --recursive
```

### 2. Configure the Environment
We recommend using `uv` to sync the environment. This will automatically install PyTorch with CUDA support.
```bash
# Sync dependencies (installs Torch + CUDA)
uv sync
```
*You can also use the currently active environment with*
```bash
uv sync --active
```

### 3. Verify CUDA Support
Ensure your NVIDIA GPU is recognized by PyTorch:
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
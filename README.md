# Cell Detection Suite

Welcome to the *Cell Detection Suite*.  
A simple GUI wrapped around CAIMAN for easy motion correction and cell detection.

---

## Installation

### 1. Get Miniforge or Similar

Download Miniforge from:

[https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge)  
See the "Install" section there.

---

### 2. Create a New Environment and Install CAIMAN

You can also refer to:  
- [Caiman Installation Docs](https://caiman.readthedocs.io/en/latest/Installation.html)  
- [Caiman GitHub](https://github.com/flatironinstitute/CaImAn)

Open a Miniforge terminal (or similar) and run:

```bash
mamba create -n caiman -c conda-forge -c anaconda caiman
```

Or, if using `conda`:

```bash
conda create -n caiman -c conda-forge -c anaconda caiman
```

This will take some time to download and install.

Than activate your new env:

```bash
conda activate caiman
```

---

### 3. Install Demos and Models

After installation, run:

```bash
caimanmanager install
```

If this completes successfully, CAIMAN should now be installed.

---

### 4. Install Some Additional Packages

You need a few more packages:

Using `conda`:

```bash
conda install PyYAML tifffile
```

Or with `pip`:

```bash
conda install pip
pip install PyYAML tifffile
```

---

### 5. Cell Detection Suite GUI

Save the following files to a folder on your disk:
- `run_caiman.py`
- `caiman_settings.yaml`

Then, open a terminal (e.g., Miniforge) and navigate to the folder:

```bash
cd C:/path/to/files/
```

Now run the Python file:

```bash
python run_caiman.py
```

This should start the GUI. It may take some time to appear, as it needs to import CAIMAN.

> **Important:**  
> The `caiman_settings.yaml` file must always be in the same directory as `run_caiman.py`.

---

### ----------  
Nils Brehm - 2025
# RoiBaView
Welcome to the <i>Cell Detection Suite</i>.
A simple GUI wrapped around CAIMAN for easy motion correction and cell detection.

## Installation
1. Get Miniforge or something similar:

https://github.com/conda-forge/miniforge
See "Install" there.

2. Create a new environment and install CAIMAN(e.g. in Anaconda)
See also: 
https://caiman.readthedocs.io/en/latest/Installation.html
https://github.com/flatironinstitute/CaImAn

Open miniforge terminal (or whatever you have) and create a new environment and install caiman.
(it will look in channel conda-forge and anaconda for caiman)

```shell
mamba create -n caiman -c conda-forge -c anaconda caiman 
```
or
```shell
conda create -n caiman -c conda-forge -c anaconda caiman 
```

This will take some time to download and install.
After that you need to install some demos and models:

```shell
caimanmanager install
```
If this is done, caiman should be installed successfully.

4. Install some additional stuff
We need some more packages:

```shell
conda install PyYAML tifffile
```
or
```shell
conda install pip
pip install PyYAML tifffile
```

5. Cell Detection Suite GUI
Save the "run_caiman.py" and "caiman_settings.yaml" to your disk.
Open a terminal (miniforge) and go to the directory containing that two files.

```shell
cd C:/path/to/files/
```

Now run the python file
```shell
python run_caiman.py
```
This should start the GUI. It can take some time until it shows, since it first has to import caiman.

The Caiman Settings are stored in "caiman_settings.yaml" and must always be in the same directory as the python file!


### ----------
Nils Brehm - 2025

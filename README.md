# Ajin
[![License](https://img.shields.io/:license-mit-blue.svg)](https://badges.mit-license.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

# Requirements
For some dependencies of our library we need to have installed the `gcc` compiler. Please install `gcc` using one of the following commands for the linux distributions *Arch, Solus4* or *Ubuntu*, or another package manager suitable for your operating system and/or distribution:
```bash
# Archlinux
sudo pacman -S gcc

# Solus4
sudo eopkg install gcc
# These are the requirements to run gcc for Solus4
sudo eopkg install -c system.devel

# Ubuntu
sudo apt update
sudo apt install build-essential
sudo apt-get install python3-dev
sudo apt-get install manpages-dev
gcc --version
```
Some packages are way easier to install using Anaconda. For the installation on several linux distributions please follow [this lin](https://docs.anaconda.com/anaconda/install/linux/). Further the installation of our clustering prototype requires some python packages to be installed. We provide a requirements file, but here is a complete list for manual installation using `pip3` and `python 3`:
```bash
pip3 install pandas
pip3 install sklearn
pip3 install tadasets
pip3 install tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install tikzplotlib
# Works only with gcc installed.
pip3 install hdbscan
```
We'll need the Gudhi Python library. A good way to install Gudhi is to utilize the Anaconda environment. This installation requires a non default channel called `conda-forge`. You can use one of the following commands:
```
# Install Gudhi, easiest installation with Anaconda.
# Gudhi is a library to compute persistent homology.
conda install -c conda-forge gudhi
conda install -c conda-forge/label/cf201901 gudhi 
```
**Information**: Some of the somewhat older Gudhi versions have bugs for the Python bindings. Therefore, please use at least Gudhi in version `3.1.1`, which is also available [here](https://github.com/GUDHI/gudhi-devel/releases/tag/tags/gudhi-release-3.1.1) for download. Instructions for the `C++` installation of Gudhi on your system can be found [here](https://gudhi.inria.fr/doc/latest/installation.html).
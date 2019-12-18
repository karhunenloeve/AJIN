### ðSIML

### Requirements
For this package we use the `gcc` compiler. Please install `gcc` using one of the following commands for the linux distributions *Arch, Solus4* or *Ubuntu*:
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

 Some packages are way easier to install using Anaconda. For the installation on several linux distributions please follow [this link](https://docs.anaconda.com/anaconda/install/linux/). Further the installation of our clustering prototype requires some python packages to be installed. We provide a requirements file, but here is a complete list for manual installation using `pip3` and `python 3`:
```bash
  pip3 install pandas
  pip3 install sklearn
  # Works only with gcc installed.
  pip3 install hdbscan

  # Install Gudhi, easiest installation with Anaconda.
  # Gudhi is a library to compute persistent homology.
  conda install -c conda-forge gudhi
  conda install -c conda-forge/label/cf201901 gudhi 
```

# Setup

## Dependencies

### Docker
#### Windows
Install the [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/).

#### Linux (Ubuntu)
Uninstall all conflicting packages.
```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

Set up Docker's `apt` repository.
```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

Install the Docker packages.
```bash
yes | sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```

Add to sudoer group.
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
**Restart your system**, then test the installation.
```bash
docker run hello-world
```

### Conda
#### Windows
Install Miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation).
#### Linux
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

## Python environment
We can use either a python `venv` or a conda environment (recommended) for this project. Instructions for both are given below.

### PyTorch
If you would like to install the GPU version of PyTorch, go to the [official PyTorch install page](https://pytorch.org/get-started/locally/) and select your system with the `pip` option. Then copy the `--index-url` value into `requirements.txt` (if none present, remove it from `requirements.txt` as well).

### Conda environment (recommended)
```bash
conda env create -f environment.yaml
conda activate deepracer
```

### Python `venv`
Python version 3.10 or above is required.
```bash
environment='.deepracer'
python -m venv "$environment"
"$environment"/bin/python -m pip install -r requirements.txt
"$environment"/bin/python -m pip install -e ./packages/
source "$environment"/bin/activate
```

## PACE ICE
We recommend students to setup the project locally. However, in cases where that may not be possible, students can explore the following PACE ICE remote compute services.

* Login to the [GeorgiaTech VPN Service](https://vpn.gatech.edu/global-protect/login.esp). Download and install the [GlobalProtecht VPN client](https://vpn.gatech.edu/global-protect/getsoftwarepage.esp).
* Using the VPN client, connect to [vpn.gatech.edu](vpn.gatech.edu) and login via your GeorgiaTech username and password.
* Connect to the PACE ICE on-demand service at [ondemand-ice.pace.gatech.edu](https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard).
* Click on 'My Interactive Sessions' and select whichever one you prefer on the 'Interactive Apps' meanu (we recommend VS Code).

### Environment setup
Please note that PACE ICE machines already come with Apptainer and Conda installed (use `module load anaconda3` or `module load mamba`). As such, you can follow the instructions from the [python environment section](#Python-environment) as is. However, we recommend the following additions:

- **PyTorch:** Please use the following value for `--index-url` in the `requirements.txt` file to use CUDA with PyTorch.
```bash
--index-url https://download.pytorch.org/whl/cu126
```
- **prefix:** We recommend that you install the conda environment using a `--prefix` flag as the `~/scratch` directory to prevent using up your storage.
```bash
environment='deepracer'
scratch_directory="$HOME"/scratch/conda
conda env create -f environment.yaml \
    --prefix "$scratch_directory"/"$environment"
conda config --append envs_dirs "$scratch_directory"
conda activate "$environment"
```
## CS492(H): Machine Learning for 3D Data — KCloud Instruction

### 1. Login to the KCloud.

Check out the KCloud VPN/VM tutorial sent to your email address and login to the VM with your credential information.


### 2. Install Wget.

Open a terminal and login to the KCloud.

Run:
```
sudo apt-get install wget
wget --version
```

Check whether you see a message starting like this:
```
GNU Wget 1.19.4 built on linux-gnu.
```

### 3. Install Nvidia library.

Run:
```
mkdir setup && cd setup
wget https://raw.githubusercontent.com/mhsung/kaist-cs492h-3dml-2021-spring/main/setup/install_nvidia.sh?token=ABNDQXK4WYNFZ7O52I4LNSDAKWPH6 -O install_nvidia.sh
sudo sh install_nvidia.sh
```

You'll be logged out after the install. Wait for one minute and then log in and run:
```
nvidia-smi
```

You should be able to see a screen like this:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.39       Driver Version: 460.39       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 3090    Off  | 00000000:00:06.0 Off |                  N/A |
| 30%   36C    P0    61W / 350W |      0MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 4. Install Python 3 and the other essential packages.

Run:
```
cd setup
wget https://raw.githubusercontent.com/mhsung/kaist-cs492h-3dml-2021-spring/main/setup/install_essential.sh?token=ABNDQXJQ6PGQICRMVUCPUYDAKWQAE -O install_essential.sh
sudo sh install_essential.sh
python3 --version
cd ..
```
Check whether you see the following output:
```
Python 3.8.0
```

This script will also create a python virtual environment.
Check whether your command prompt now starts with `(venv)`.


### 5. Install pip and python packages.
Run:
```
cd setup
wget https://raw.githubusercontent.com/mhsung/kaist-cs492h-3dml-2021-spring/main/setup/requirements.txt?token=ABNDQXKC33R2UGPKNF3TMBLAKWRIS -O requirements.txt
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
cd ..
```

The outputs should be:
```
1.7.1
True
```

### 6. Start JupyterLab.
Run:
```
jupyter lab --port=40{xx} --no-browser
```

`{xx}` is the last two digits of your VPN ID.
If the last two digits of your VPN ID is `01`, run:
```
jupyter lab --port=40{xx} --no-browser
```

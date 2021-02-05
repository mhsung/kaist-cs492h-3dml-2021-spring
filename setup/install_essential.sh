# Minhyuk Sung <mhsung@kaist.ac.kr>

# Install wget and build-essential
sudo apt-get update && sudo apt-get install -y \
  build-essential \
  wget \
  sudo \
  locales \
  screen \
  tmux \
  apt-utils && \
  sudo apt-get -y autoremove && sudo apt-get -y clean

# Rebuild cache
cd /var/lib/apt && \
  sudo mv lists lists.old && \
  sudo mkdir -p lists/partial && \
  sudo apt-get update && \
  sudo apt-get -y autoremove && sudo apt-get -y clean && \
  cd ~

# Set the locale
sudo locale-gen en_US.UTF-8
echo LANG en_US.UTF-8
echo LANGUAGE en_US:en
echo LC_ALL en_US.UTF-8

# Non-interactive package installation
echo DEBIAN_FRONTEND=noninteractive

############## BASIC DEPENDENCIES AND COMMON PKGs #####################
# Install vim & nano
sudo apt-get --assume-yes install vim && \
  sudo apt-get -y autoremove && sudo apt-get -y clean
sudo apt-get --assume-yes install nano && \
  sudo apt-get -y autoremove && sudo apt-get -y clean

# More basic dependencies and python3
sudo apt-get update && sudo apt-get install -y \
        apt-utils \
        kmod \
        openssh-server \
        curl \
        libfreetype6-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        screen \
        && \
    sudo apt-get -y autoremove && sudo apt-get -y clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Python 3 and related dependencies
sudo apt-get update && sudo apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3-pip \
        virtualenv \
        && \
    sudo apt-get -y autoremove && sudo apt-get -y clean

sudo sudo apt-get update && \
    sudo apt-get install -y autoconf libtool pkg-config && \
    sudo apt-get -y autoremove && sudo apt-get -y clean

virtualenv --python=python3.8 ~/venv

echo "source ~/venv/bin/activate" | sudo tee -a ~/.profile
source ~/.profile
python --version
#######################################################################

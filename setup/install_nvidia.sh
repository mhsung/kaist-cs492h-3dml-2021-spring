# Install Nvidia library
sudo apt-get update \
  && sudo apt-get -y install build-essential
sudo add-apt-repository -y ppa:graphics-drivers/ppa \
  && sudo apt-get update
sudo apt-get -y install nvidia-driver-460
sudo reboot

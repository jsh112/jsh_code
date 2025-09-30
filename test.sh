sudo apt-get update
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install --upgrade pip
# 예: JetPack 4.6용 PyTorch 1.12 설치 (Python 3.6)
wget https://nvidia.box.com/shared/static/pytorch-1.12.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.12.0-cp36-cp36m-linux_aarch64.whl


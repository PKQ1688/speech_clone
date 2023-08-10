# speech_clone
用于声音克隆

# 构建相关的环境
```
conda create -n maas python=3.7
conda activate maas
sudo apt-get install sox
pip install torch torchvision torchaudio
pip install modelscope 
pip install tts_autolabel==1.1.2 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install matplotlib
pip install kantts -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
pip install tensorboardX
pip install bitstring
pip uninstall typeguard
pip install typeguard==2.13.3
```
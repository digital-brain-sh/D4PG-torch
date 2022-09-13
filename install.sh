sudo apt-get install libosmesa6-dev -y
sudo apt autoremove

MUJOCO_PATH=$HOME/.mujoco/mujoco210

if [ ! -d $MUJOCO_PATH ]
then
  echo "* Directory ${MUJOCO_PATH} DOES NOT exists."
  echo "* Downloading Mujoco210 ..."
  wget -O /tmp/mujoco210-linux-x86_64.tar.gz https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  mkdir $HOME/.mujoco
  tar -xvf /tmp/mujoco210-linux-x86_64.tar.gz $HOME/.mujoco/
  rm /tmp/mujoco210-linux-x86_64.tar.gz
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
else
  echo "* Mujoco at $MUJOCO_PATH DETECTED."
fi

conda config --set show_channel_urls yes
cat << EOF > $HOME/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda clean -i
# NOTE: torch 1.12.0 has a bug on shared_memory: https://github.com/pytorch/pytorch/issues/80733
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple
pip install --upgrade pip

echo "* Installing d4rl from git@github.com:digital-brain-sh/d4rl ..."
pip install git+https://github.com/digital-brain-sh/d4rl.git@master

echo "* Install gym[all] and autorom ..."
pip install gym[all]
pip install autorom

AutoROM

echo "* Installing requirements from requirements.txt..."
pip install -r requirements.txt
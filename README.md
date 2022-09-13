# D4PG-torch
Torch implementation of D4PG, for data collection of GATO. This version is forked from [schatty/d4pg-pytorch](https://github.com/schatty/d4pg-pytorch)


## Installation

You can execute installation by pasting following instructions in your commandline. Or directly running [`bash install.sh`](/install.sh) when you've create a conda envrionment.


1. Create and activate conda environment

    ```bash
    conda create -n d4pg python=3.8 -y
    conda activate d4pg
    ```

2. Install `torch-gpu` and mujoco-enviornment

    ```bash
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
    ```

3. Install d4rl enviornment suite and training supports

    ```bash
    pip install git+https://github.com/digital-brain-sh/d4rl.git@master
    pip install -r requirements.txt
    ```

    Please note that if your network condition is poor, you can alternatively download our d4rl package from [digital-brain-sh/d4rl](https://github.com/digital-brain-sh/d4rl), and run `pip install -e .` under d4rl's workspace.

## Train and collect

See the implementation in [train_and_collect.py](/train_and_collect.py)
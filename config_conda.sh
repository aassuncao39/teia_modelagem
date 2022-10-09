#!/bin/sh
#shashum -a 256 ~/Downloads/Anaconda3-2022.05-Linux-x86_64.sh
bash ~/Downloads/Anaconda3-2022.05-Linux-x86_64.sh -b
source ~/.bashrc
export PATH="~/anaconda3/bin/conda:$PATH"
reset
conda update -n base -c defaults conda
conda create -y --name modelagem_gt607
conda init bash
echo "Por favor, reinicie o terminal e execute o comando conda init bash "
#conda activate modelagem_gt607
#conda install pip
#pip install -r requirements.txt
#./config_env.sh

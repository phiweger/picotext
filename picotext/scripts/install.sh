# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
conda create -y -n picotext python==3.7 tensorboard tqdm
conda activate picotext
conda install -y -c conda-forge osfclient
conda install -y -c bioconda screed
# On FloydHub we don't need to install PyTorch
# conda install -y -c pytorch pytorch torchvision torchtext 
pip install tokenizers==0.7.0
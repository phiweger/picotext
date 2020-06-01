conda create -y -n picotext python==3.7 tensorboard tqdm
conda activate picotext
conda install -y -c conda-forge osfclient
conda install -y -c bioconda screed
conda install -y -c pytorch pytorch torchvision torchtext 
pip install -y tokenizers==0.7.0
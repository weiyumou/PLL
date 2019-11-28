# install tensorboard
pip install tb-nightly future

# install transformers
pip install transformers

# install Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


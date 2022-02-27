# Install MMDET & Torch & CUDA
# install dependencies: (use cu111 because colab has CUDA 11.1/11.2)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# install mmcv-full thus we could use CUDA operators
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdetection
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

pip install -e .

cd .. 
cp -R mmdetection/ /content/260-PGD/data/

# install Pillow 7.0.0 back in order to avoid bug in colab
pip install Pillow==7.0.0
cd /content/260-PGD

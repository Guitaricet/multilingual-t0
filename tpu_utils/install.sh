sudo pip uninstall jax jaxlib -y
pip3 install -U pip
pip3 install jax jaxlib
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*

mkdir -p ~/code
cd ~/code

git clone https://github.com/bigscience-workshop/t5x.git
pushd t5x
git checkout thomas/add_train_script_span_corruption
pip3 install -e .
popd

# TODO: figure if this is actually important
sudo rm /usr/local/lib/python3.8/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so

## TODO: figure why I need this
##   This is probably linked to `use_custom_packing_ops=True`. Just set it to False and we're good to go
#pip3 install tensor2tensor

# Needed for profiling to work apparently
pip3 install tbp-nightly

## ...
#pip3 install tensorflow==2.7.0

# Install Promptsource
git clone https://github.com/bigscience-workshop/promptsource.git
cd promptsource
pip3 install -e .
cd ../

git clone https://github.com/lintangsutawika/multilingual-t0.git

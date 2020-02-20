# DK-for-TST

Hi, this is the pytorch code for MMD with deep kernel.

# Software version
Torch version is 1.1.0. Python version is 3.7.3. CUDA version is 10.1.

# For Blob dataset,

1) run

python Deep_Kernel_Blob.py

you can obtain average test power of MMD-D on Blob dataset;

2) run 

python Baselines_Kernel_Blob.py

you can obtain average test power of MMD-O, C2ST-L, C2ST-S, ME and SCF on Blob dataset;

3) run 

python Ablation_Tests_Blob.py

you can obtain average test power of L+J, G+J, G+C and D+C on Blob dataset.

# For HDGM dataset,

1) run

python Deep_Kernel_HDGM.py

you can obtain average test power of MMD-D on HDGM dataset;

2) run 

python Baselines_Kernel_HDGM.py

you can obtain average test power of MMD-O, C2ST-L, C2ST-S, ME and SCF on HDGM dataset;

3) run 

python Ablation_Tests_HDGM.py

you can obtain average test power of L+J, G+J, G+C and D+C on HDGM dataset.

# For Higgs dataset,

1) run

python Deep_Kernel_HIGGS.py

you can obtain average test power of MMD-D on Higgs dataset;

2) run 

python Baselines_Kernel_HIGGS.py

you can obtain average test power of MMD-O, C2ST-L, C2ST-S, ME and SCF on Higgs dataset;

3) run 

python Ablation_Tests_HIGGS.py

you can obtain average test power of L+J, G+J, G+C and D+C on Higgs dataset.

# For MNIST dataset,

1) run

python Deep_Baselines_MNIST.py

you can obtain average test power of MMD-D, MMD-O, C2ST-L, C2ST-S, ME and SCF on MNIST dataset;

2) run 

python Ablation_Tests_MNIST.py

you can obtain average test power of L+J, G+J, G+C and D+C on MNIST dataset.

# For CIFAR10 dataset,

1) run

python Deep_Baselines_CIFAR10.py

you can obtain average test power of MMD-D, MMD-O, C2ST-L, C2ST-S, ME and SCF on CIFAR10 dataset;

2) run 

python Ablation_Tests_CIFAR10.py

you can obtain average test power of L+J, G+J, G+C and D+C on CIFAR10 dataset.

# For interpretability on CIFAR10 dataset,

1) run

python Interpretability_CIFAR10_train_location.py

you can obtain the best test locations (trained to maximize ME statistic) of deep-kernel ME and ME tests;

2) run 

python Interpretability_CIFAR10_select_location.py

you can obtain the best test locations (selected to maximize ME statistic) of deep-kernel ME.

Two codes will draw the best locations as png files.

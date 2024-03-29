# Learning deep kernels for two-sample testing

Hi, this is the pytorch code for MMD with deep kernel, presented in the ICML2020 paper "Learning Deep Kernels for Non-Parametric Two-Sample Tests" (ICML2020) (https://arxiv.org/abs/2002.09116). This work is done by 

- Dr. Feng Liu (UTS), feng.liu@uts.edu.au
- Dr. Wenkai Xu (Gatsby Unit, UCL), wenkaix@gatsby.ucl.ac.uk
- Prof. Jie Lu (UTS), jie.lu@uts.edu.au
- A/Prof. Guangquan Zhang (UTS), guangquan.zhang@uts.edu.au
- Prof. Arthur Gretton (Gatsby Unit, UCL), arthur.gretton@gmail.com
- Dr. Danica J. Sutherland (UBC), djs@djsutherland.ml.


# Software version
Torch version is 1.1.0. Python version is 3.7.3. CUDA version is 10.1.

Most codes require freqopttest repo (interpretable nonparametric two-sample test)
to implement ME and SCF tests, which can be installed by

pip install git+https://github.com/wittawatj/interpretable-test

These python files (14 in total), of cause, require some basic scientific computing python packages, e.g., numpy, sklearn and matplotlib. I recommend users to install python via Anaconda (python 3.7.3), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda, pytorch (gpu) and freqopttest, you can run codes related to Bolb, HDGM and CIFAR10 (9 python files) successfully. For the other codes (5 python files), you need to download the following data.

# Download data

Since Github does not allow big-size file (>25MB), you can download fake_mnist and higgs dataset from the following links:

fake_mnist (generated by DCGAN): https://drive.google.com/open?id=13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5

higgs data (need pickle to load): https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc

After you download both data, you should run all 14 python files successfully.

# For Blob dataset,

1) run

```
python Deep_Kernel_Blob.py
```

you can obtain average test power of MMD-D on Blob dataset;

2) run 

```
python Baselines_Kernel_Blob.py
```

you can obtain average test power of MMD-O, C2ST-L, C2ST-S, ME and SCF on Blob dataset;

3) run 

```
python Ablation_Tests_Blob.py
```

you can obtain average test power of L+J, G+J, G+C and D+C on Blob dataset.

# For HDGM dataset,

1) run

```
python Deep_Kernel_HDGM.py
```

you can obtain average test power of MMD-D on HDGM dataset;

2) run 

```
python Baselines_Kernel_HDGM.py
```

you can obtain average test power of MMD-O, C2ST-L, C2ST-S, ME and SCF on HDGM dataset;

3) run 

```
python Ablation_Tests_HDGM.py
```

you can obtain average test power of L+J, G+J, G+C and D+C on HDGM dataset.

# For Higgs dataset,

1) run

```
python Deep_Kernel_HIGGS.py
```

you can obtain average test power of MMD-D on Higgs dataset;

2) run 

```
python Baselines_Kernel_HIGGS.py
```

you can obtain average test power of MMD-O, C2ST-L, C2ST-S, ME and SCF on Higgs dataset;

3) run 

```
python Ablation_Tests_HIGGS.py
```

you can obtain average test power of L+J, G+J, G+C and D+C on Higgs dataset.

# For MNIST dataset,

1) run

```
python Deep_Baselines_MNIST.py
```

you can obtain average test power of MMD-D, MMD-O, C2ST-L, C2ST-S, ME and SCF on MNIST dataset;

2) run 

```
python Ablation_Tests_MNIST.py
```

you can obtain average test power of L+J, G+J, G+C and D+C on MNIST dataset.

# For CIFAR10 dataset,

1) run

```
python Deep_Baselines_CIFAR10.py
```

you can obtain average test power of MMD-D, MMD-O, C2ST-L, C2ST-S, ME and SCF on CIFAR10 dataset;

2) run 

```
python Ablation_Tests_CIFAR10.py
```

you can obtain average test power of L+J, G+J, G+C and D+C on CIFAR10 dataset.

# For interpretability on CIFAR10 dataset,

1) run

```
python Interpretability_CIFAR10_train_location.py
```

you can obtain the best test locations (trained to maximize ME statistic) of deep-kernel ME and ME tests;

2) run 

```
python Interpretability_CIFAR10_select_location.py
```

you can obtain the best test locations (selected to maximize ME statistic) of deep-kernel ME.

Two codes will draw the best locations as png files.

# Citation
If you are using this code for your own researching, please consider citing
```
@inproceedings{liu2020learning,
  title={Learning Deep Kernels for Non-Parametric Two-Sample Tests},
  author={Liu, Feng and Xu, Wenkai and Lu, Jie and Zhang, Guangquan and Gretton, Arthur and Sutherland, Danica J.},
  booktitle={ICML},
  year={2020}
}
```

# Acknowledgment
This work was supported by the Australian Research Council under FL190100149 and DP170101632, and by the Gatsby Charitable Foundation. FL, JL and GZ gratefully acknowledge the support of the NVIDIA Corporation with the donation of two NVIDIA TITAN V GPUs for this work. FL also acknowledges the support from UTS-FEIT and UTS-AAII. DJS would like to thank Aram Ebtekar, Ameya Velingker, and Siddhartha Jain for productive discussions.

# DK-for-TST

Hi, this is the pytorch code for MMD with deep kernel.

# Software version
Torch version is 1.1.0. Python version is 3.7.3. CUDA version is 10.1.

# How to use

You can obtain the Blob results by running main.py (main_H0.py for P=Q).

You can obtain the HDGM results by running main_HDGM.py (main_HDGM_H0.py for P=Q).

# Network Structure

The network used by deep kernel has four fully-connected layers. Two hidden layers have H neurons. In Blob example,  H=30. In HDGM, H=3*d, where d is the dimension of samples.

# On-going code

The code of classifier two sample test (C2ST) has been tested in main.py and main_H0.py. 

It seems valid now. However, running C2ST with permutations spends a lot of time since I need to fit a NN at each permutation.

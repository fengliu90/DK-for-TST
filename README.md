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

If we calculate the threshod using asymtotic null distribution (i.e., N(1/2,1/4/n)), C2ST will have a very high Type-I error. 

If we use the permutation, C2ST has a low test power. For example, the accuracy on test set is 0.505. However, the threshold confirmed by permutations is around 0.55.

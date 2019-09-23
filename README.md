# DK-for-TST

Hi, this is the pytorch code for MMD with deep kernel.

# Software version
Torch version is 1.1.0. Python version is 3.7.3. CUDA version is 10.1.

# How to use

You can obtain the Blob results by running main.py (main_H0.py for P=Q).

You can obtain the HDGM results by running main_HDGM.py (main_HDGM_H0.py for P=Q).

# Network Structure

The network used by deep kernel has four fully-connected layers. Two hidden layers have H neurons. In Blob example,  H=50. In HDGM, H=3*d, where d is the dimension of samples.

# On-going code

C2ST:

The code of classifier two sample test (C2ST) has been tested in main.py and main_H0.py. It works normally now. I use a strategy to set batchsize and number of epochs for C2ST, which makes it perform well. However, when we have a lot of samples, test power of C2ST will sligtly drop (see Figure 2 in our paper). 

In Blob example, if we have a lot of example, the classifier actually cannot distinguish two samples (P and Q are different) since two samples are very near. This caused that, if x is from P (label is 1), then, x+e is from Q (label is 0), where |e| is a very small value. In this case, the classifier actuall cannot distinguish two samples alghouth two sample are from distributions.



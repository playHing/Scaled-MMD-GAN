# Official Tensorflow implementation for reproducing the results of [*Demystifying MMD GANs.*'](https://arxiv.org/abs/1801.01401).

The repository contains code for reproducing experiments of uncoditional image generation with MMD GANs and other benchmark GAN models. 

### References
[Mikołaj Bińkowski, Dougal J. Sutherland, Michael N. Arbel and Athur Gretton *Demystifying MMD GANs*](https://arxiv.org/abs/1801.01401)

### Requirements


### Model features
- Uses gradient penalty, analoguous to WGAN-GP ([Gulrajani et al *Improved Training of Wassersein GAN*](https://arxiv.org/abs/1704.00028). 
- Evaluates models using three different methods: *Inception Score*, *Fr\'echet INception Distance (FID)*, and proposed *Kernel Inception Distance (KID)*.
- Adaptively decreses the learning rate using 3-sample test. If KID does not improve (as compared to evaluation 20k steps earlier) three times in a row, learning rate is halved.

### Datasets
The code works with several common datasets with different resolutions. The experiments include
- 28x28 MNIST,
- 32x32 Cifar10,
- 64x64 LSUN Bedrooms,
- 160x160 Celeb-A.
 
LSUN, MNIST and Celeb-A datasets can be downloaded using the [script](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py).

### Benchmarks

We compare MMD GANs with WGAN-GP and Cramer GAN


### Running the code
Each of the following scripts launches the training of MMD GAN on respective dataset: `mnist.sh`, `cifar10.sh`, `lsun.sh`, `celeba.sh`. To train the benchmark models, change the variable `$MODEL` to `WGAN` or `CRAMER`. To train all three models set `$MODEL=ALL`.



Feel free to contact Mikołaj Bińkowski ('mikbinkowski at gmail.com') with any 
questions and issues.

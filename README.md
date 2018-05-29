[//]: <links>
[smmd]: https://arxiv.org/abs/

# On gradient regularizers for MMD GANs: Scaled MMD


Official Tensorflow implementation for reproducing results of unsupervised image generation using [Scaled MMD][smmd].


## Setup
### install :

`pip install -r requirements.txt`

The GPU compatible version of tensorflow is required for this code to work.


### Download CelebA dataset:
```
cd scripts
OUTPUT_DATA_DIR=/path/to/output/directory/  # path to the parent directory containing the datasets
python scripts/download.py --datasets celebA --output_dir $OUTPUT_DATA_DIR
```

### Download ImageNet dataset:
Please download ILSVRC2012 dataset from http://image-net.org/download-images

### Preprocess dataset:
```
cd datasets
IMAGENET_TRAIN_DIR=/path/to/imagenet/train/ # path to the parent directory of category directories named "n0*******".
PREPROCESSED_DATA_DIR=/path/to/save_dir/
bash preprocess.sh $IMAGENET_TRAIN_DIR $PREPROCESSED_DATA_DIR
# Make the list of image-label pairs for all images (1000 categories, 1281167 images).
python imagenet.py $PREPROCESSED_DATA_DIR
# (optional) Make the list of image-label pairs for dog and cat images (143 categories, 180373 images).
python imagenet_dog_and_cat.py $PREPROCESSED_DATA_DIR
```

### Convert imagenet jpeg images to tfrecords:
```
cd scripts
OUTPUT_DATA_DIR=/path/to/output/tfrecords 
TRAIN_DIR=/path/to/train/directory
build_imagenet_data --train_directory=$TRAIN_DIR --output_directory=$OUTPUT_DATA_DIR
```

### Download inception model: 

`python source/inception/download.py --outfile=datasets/inception_model`


## Training


### Unsupervised image generation of 64x64 ImageNet images:
```
DATADIR=/path/to/datadir/
OUTDIR=/path/to/outputdir/
CONFIG=configs/imagenet_smmd.yml
# multi-GPU: 3 GPUs
CUDA_VISIBLE_DEVICES=0,1,2 python -m ipdb gan/main.py -dataset imagenet -data_dir $DATADIR -name  -config_file $CONFIG -out_dir $OUTDIR -multi_gpu true
```


- Examples of generated images at 150K iterations:

![image](https://github.com/MichaelArbel/Scaled-MMD-GAN/imagenet_sample.jpg)


### References
Michael Arbel, Dougal J. Southerland, Mikolaj Binkowski, Arthur Gretton. *On gradient regularizers for MMD GANs*. [arXiv][smmd]


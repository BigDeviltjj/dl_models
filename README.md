# dl_models
Common deep learning models implemented by tensorflow

For ResnetV1:

cifar10:run cifar10_download_and_extract.py to download the cifar10 dataset. Then run cifar10_main --data-dir='the path where you save your dataset' to use Resnet to train on cifar10.

Imagenet:You will need to download the ImageNet dataset and convert it to TFRecord format. Follow along with the Inception guide in order to prepare the dataset. Then train the model as follow:python imagenet_main.py --data_dir='the path where you save your dataset'.


For vgg16:

Pretrained models should be downloaded from ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

In vgg16_test.py, the path of vgg16.npy should be changed to the download file.

The main difference of my implementation with others is that I converted the fully connected layer to conv layer so that images with any size(bigger than 224) can be fed into the model when testing, and the accuracy is higher than the cropped version.


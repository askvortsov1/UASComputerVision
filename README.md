# UASComputerVision

![Sample composition of localization cross sections representing target size during the competition](https://github.com/lukeottey/UASComputerVision/blob/master/perspective_from_heli.JPG)

Current Agenda:

1) Rewrite FractionalMaxPool transformation
2) Linearize Classification.py
3) Remove OPTIM directory, the only optimizers being used will be SGD and ADAM (this will be done once classification is rewritten)
4) Find efficient dimensionality reduction technique to reduce input size of image to the localization pipeline (DONE)
5) Figure out more efficient way of creating artificial localization input images (if we find a way to do this quickly enough
we can create new images at run-time since storing enough images for training that are of size 4048x3036 is very rough on memory) (DONE)
6) Build .pkl files for training and testing images for classification so that we can avoid having to load each image from memory on every iteration through the dataset. Accessing the images from the .pkl data structure will be more heavy on CPU memory but it will have a much more reasonable run-time complexity. Images should be aligned with their respective features in the data structure that we create. (DONE - using h5py instead of pickle because of less memory consumption and quicker load time)
7) Make aspect ratio rv a function of a truncated normal distribution instead of random uniform

Below is a link to the google drive from which you can download the data I generated. download the 'data' directory and put it in the parent directory of this repository. The file paths in this repo have already been adjusted to accomadate this file placement.
https://drive.google.com/open?id=1-6Gyro8OHejUN7_9uzWfHZQXd5tnSjeZ

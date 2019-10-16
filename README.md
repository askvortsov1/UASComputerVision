# UASComputerVision
Current Agenda:

1) rewrite FractionalMaxPool transformation
2) Linearize Classification.py
3) Remove OPTIM directory, the only optimizers being used will be SGD and ADAM (this will be done once classification is rewritten)
4) Find efficient dimensionality reduction technique to reduce input size of image to the localization pipeline
5) Figure out more efficient way of creating artificial localization input images (if we find a way to do this quickly enough
we can create new images at run-time since storing enough images for training that are of size 4048x3036 is very rough on memory)
6) Build .pkl files for training and testing images for classification so that we can avoid having to load each image from memory on every iteration through the dataset. Accessing the images from the .pkl data structure will be more heavy on CPU memory but it will have a much more reasonable run-time complexity. Images should be aligned with their respective features in the data structure that we create.

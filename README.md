# Predictive Analytics in Long-Term Care

This repository provides the code and data used for Chapter 13, "Predictive Analytics in Long-Term Care" to be published by Springer in the book Actuarial Aspects of Long-Term Care.

For more information, contact Howard Zail [info@elucidor.com](mailto:info@elucidor.com)


# Requirements

A number of  packages are required and can be installed using

```
installed.packages(("dplyr", "readr", "tidyr", "ggplot2", "purrr", "glmnet", "speedglm"))
```

If you have Ubuntu installed, you can use the doMC packages to run the Lasso code in parallel.  Set the number of cores available to the desired level by adjusting the code in `ltc_note_book_02.Rmd` file.  If you have Windows, these two lines must be removed:

```
require(doMC) # Not available in Windows
registerDoMC(cores=10) # Set to the available no. of cores

```

## Data

The data file used is too large for Github, and can be downloaded here:

https://drive.google.com/file/d/1A5Qg-zqtO3_SIWFI5x105nnnmK_Mr_ob/view?usp=sharing

Download and save in the `data` subdirectory of the `ltc` repository.



## Keras and Tensorflow

Keras and Tensorflow need to be installed.  It is far easier to install Keras for use on a CPU, but this will substantially slow down the models relative to using the GPU installation.

Details on installing the packages can be found here:

* https://tensorflow.rstudio.com/keras/
* https://tensorflow.rstudio.com/keras/reference/install_keras.html


You can use the following code to check whether you have a GPU or CPU installation for Tensorflow.  

```
library(tensorflow)  
a = tf$constant(c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), shape=c(2, 3), name='a')
b = tf$constant(c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), shape=c(3, 2), name='b')
c = tf$matmul(a,b)
sess = tf$Session(config=tf$ConfigProto(log_device_placement=TRUE))
print(sess$run(c))

```
I have also found the following code useful to check your Python / Tensorflow setup:

```
library(reticulate)
py_discover_config("keras")
py_discover_config("tensorflow")
tensorflow::tf_config()
keras:::keras_version()

# Check CPU vs. GPU
library(keras)
k = backend()
sess = k$get_session()
sess$list_devices()
```

Unfortuntately, although Tensorflow is now the most popular deep learning platform, it can be difficult to install properly, and the problems can be environment specific.  Nevertheless, there are lots of online resources to help you through this process.  As a reminder, it is far easier to get the CPU version working than the GPU version.


# Running the Analysis

The code for the chapter can be found in the notbook [ltc_note_book_02.Rmd].  Since some of the routines can take significant amount of time to run, you can test each routine by adjusting the parameter `train_val_test` of the function `get_incidence`.  This parameter consists of three components, representing the perecentage of records assigned to the training, validation and test sets.  The default is a 80%/10%/10% split.  Setting `train_val_test = c(0.1,0.1,0.1)` sets the training set to only 10% of the data, allowing the routines to be run much faster.  

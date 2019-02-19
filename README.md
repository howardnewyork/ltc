# Predictive Analytics in Long-Term Care

This repository provides the code and data used for Chapter 13, "Predictive Analytics in Long-Term Care" to be published by Springer in the book Actuarial Aspects of Long-Term Care.

For more information, contact Howard Zail [info@elucidor.com](mailto:info@elucidor.com)


# Requirements

A number of  packages are required and can be installed using

```
installed.packages(("dplyr", "readr", "tidyr", "ggplot2", "purrr", "glmnet"))
```

If you have Ubuntu installed, you can use the doMC packages to run the Lasso code in parallel.  Set the number of cores available to the desired level by adjusting the code in `ltc_note_book_02.Rmd` file.  If you have Windows, these two lines must be removed:

```
require(doMC) # Not available in Windows
registerDoMC(cores=10) # Set to the available no. of cores

```


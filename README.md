# PCR-GLOBWB_surrogate
Repository related to the deep-learning PCR-GLOBWB surrogate (DL-GLOBWB). Contains the code used to make and train the deep-learning surrogate as well as the multi-resolution surrogate and several single-resolution surrogate variants (which can be found in the *run/train/[landsurface or routing]/saves* folder).

Note that this repository is under active development and specifically designed for the deep-learning PCR-GLOBWB surrogate. Please contact Bram Droppers (b.droppers@uu.nl) for any questions.

## Folder structure
### surrogate/run
The run folder is the main folder. This folder contains all scripts that **prepare** and **setup** the surrogate input and output data (from PCR-GLOBWB model simulations), intitializes the **transform**ers, **tune**s the hyperparameters, and **train**s the surrogate model. Additionally some **evaluate**ion scripts are provided for plotting.

### surrogate/nn
The nn folder contains classes related tot the neural network, such as the surrogate-model, **functional** transformer and **metric** classes.

### surrogate/utils
The utils folder contains utilities for **data** loading, including the the dataset and prefetcher classes, **train**ing, including the trainer and callback classes and **plot**ting.

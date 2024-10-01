# PCR-GLOBWB_surrogate surrogate example
The example folder is an example based on the run folder. This folder contains some example data (only the landsurface part, only 30 arc-minutes and only 1 year) that has been prepared and setup. This folder includes scripts that intitializes the **transform**ers and **train**s the surrogate model. Additionally some **evaluate**ion scripts are provided for plotting.

## Training
To train the model, please run the scripts in the following order:

1. transform/landsurface/assess_train-test2_data.py
2. transform/landsurface/transform_train-test2_data.py
3. train/landsurface/train_train-test2_model.py

The trained model can be found under *train/landsurface/saves/train-test2/30min*

## Plotting
To plot the trained model outputs, please run the scripts in the following order:

1. train/landsurface/store_train-test2_model.py
2. evaluate/setup/landsurface/save_train-test2.py
2. evaluate/setup/landsurface/plot_train-test2.py
2. evaluate/train/landsurface/save_trained.py
2. evaluate/train/landsurface/plot_trained.py

The setup data plots can be found under *evaluate/setup/landsurface/saves/train-test2* and the trained data plots can be found under *evaluate/setup/train/saves/train-test2* 
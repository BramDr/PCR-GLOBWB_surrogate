# PCR-GLOBWB_surrogate surrogate run
The run folder is the main folder. This folder contains all scripts that **prepare** and **setup** the surrogate input and output data (from PCR-GLOBWB model simulations), intitializes the **transform**ers, **tune**s the hyperparameters, and **train**s the surrogate model.

## Prepare
Prepare handles the selection train, validate and test cell samples and the ordering of the prediction cell samples for both the landsurface and routing networks

## Setup
Setup uses the PCR-GLOBWB simulation configuration files to get the relevant simulation input and output files. Inputs and outputs for both the landsurface and routing networks are subsequently subsampled and stored as three-dimensional numpy arrays, named *[feature].npy*, and pickled metadata dictionaries, named *[feature]_meta.pkl*.

## Transform
Transform takes the setup data and build various transformers for it, based on wether or not transforming the data would result in a more normally-distributed distribution.

## Tune
Tune tunes the model hyperparameters using the [Optuna](https://optuna.org/) hyperparameter optimization framework using a subset of the setup data.

## Train
Train trains the model and stores the trained output.
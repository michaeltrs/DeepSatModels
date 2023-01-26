# Repository for training land cover recognition models for satellite imagery

## Setting up a python environment
- Follow the instruction in https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html for downloading and installing Miniconda

- Open a terminal in the code directory

- Create an environment using the .yml file:

	`conda env create -f deepsatmodels_env.yml`

- Activate the environment:

	`source activate deepsatmodels`

- Install required version of torch:

	`conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly`

## Initial steps for setting up experiments
- Add the base directory and paths to train and evaluation path files in "data/datasets.yaml".
- For each experiment we use a separate ".yaml" configuration file. Examples files are provided in "configs". 
The default values filled in these files correspond to parameters used in the experiments presented in respective studies. 
- Modify .yaml config files accordingly to train with your own data.

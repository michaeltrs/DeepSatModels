# Repository for training land cover recognition models for satellite imagery

## Setting up a python environment
- Follow the instruction in https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html for downloading and installing Miniconda

- Open a terminal in the code directory

- Create an environment using the .yml file:

        conda env create -f deepsatmodels_env.yml

- Activate the environment:

	    source activate deepsatmodels   

- Install required version of torch:

	    conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly

## Initial steps for setting up experiments
- Specify the base directory and paths for the training and evaluation files in the "data/datasets.yaml" file.
- Utilize a distinct ".yaml" configuration file for each experiment. Example files can be found in the "configs" folder. These files contain default values corresponding to parameters used in the associated studies.
- Adjust the ".yaml" configuration files as needed to train with your custom data.
- Refer to the instructions provided in the specific README.MD files for additional guidance on setting up and running your experiments.
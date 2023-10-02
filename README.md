# Repository for training land cover recognition models for satellite imagery

## Featured Papers
The following papers are featured in this repository:
- [ViTs for SITS: Vision Transformers for Satellite Image Time Series](https://arxiv.org/abs/2301.04944) (CVPR 2023). More information in [README_TSVIT.md](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md).
- [Context-self contrastive pretraining for crop type semantic segmentation](https://ieeexplore.ieee.org/document/9854891) (IEEE Transactions on Geoscience and Remote Sensing). More information in [README_CSCL.md](https://github.com/michaeltrs/DeepSatModels/blob/main/README_CSCL.md).

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

## License
This project is licensed under the Apache License 2.0 - see [LICENSE](https://github.com/michaeltrs/DeepSatModels/blob/main/LICENSE.txt) file for details.

Copyright © 2023 Michail Tarasiou

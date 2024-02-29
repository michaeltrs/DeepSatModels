
# Context-Self Contrastive Pretraining for Crop Type Semantic Segmentation

This repository accompanies the paper titled [Context-self contrastive pretraining for crop type semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9854891) 
published in IEEE Transactions on Geoscience and Remote Sensing. 
The work presents a novel approach to leveraging supervised contrastive learning for enhancing the 
performance of semantic segmentation models in identifying crop types from satellite imagery. 
Performance gains are most significant along parcel/object boudaries which is important for accurate object delineation. 
No additional data are required to achieve said performance gains.

## Getting Started

### Environment Setup

Activate the `deepsatmodels` Python environment to manage dependencies:

```bash
conda activate deepsatmodels
```

### Configuration

#### Dataset Configuration

1. Specify the base directory and paths for training and evaluation datasets in `data/datasets.yaml`.
2. Configuration files for each experiment are located in `configs/`. Example configuration files are provided for reference. These files contain default values corresponding to the experimental settings described in the paper.

### Model Training and Evaluation

#### Training Models

To train a model, modify the relevant `.yaml` configuration file to set the directory for saving models or to load a pre-trained model. The training process can be initiated with the following commands, depending on the model architecture:

- For a randomly initialized **UNet3D** model:

  ```bash
  python train_and_eval/segmentation_training.py --config_file configs/**/UNet3D.yaml --gpu_ids 0,1
  ```

- For a randomly initialized **UNet2D-CLSTM** model:

  ```bash
  python train_and_eval/segmentation_training.py --config_file configs/**/UNet2D_CLSTM.yaml --gpu_ids 0,1
  ```

- For **CSCL-pretrained UNet2D-CLSTM** and **UNet3Df** models, follow the two-step process:

  1. Model pre-training:

     ```bash
     python train_and_eval/segmentation_cscl_training.py --config_file configs/**/MODEL_NAME_CSCL.yaml --gpu_ids 0,1
     ```

  2. Loading the pre-trained model:

     - Update `CHECKPOINT.load_from_checkpoint` with the path to the pre-training save directory or specific checkpoint file:

       ```bash
       python train_and_eval/segmentation_training.py --config_file configs/**/MODEL_NAME.yaml --gpu_ids 0,1
       ```

Replace `MODEL_NAME` with `UNet2D_CLSTM` or `UNet3Df` as appropriate.

## Citation

If you use the data or code from this repository in your research, please cite the following paper:

```bibtex
@ARTICLE{9854891,
  author={Tarasiou, Michail and Güler, Riza Alp and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Context-Self Contrastive Pretraining for Crop Type Semantic Segmentation}, 
  year={2022},
  volume={60},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2022.3198187}}
```

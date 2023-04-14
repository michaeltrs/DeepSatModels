# [Context-self contrastive pretraining for crop type semantic segmentation](https://ieeexplore.ieee.org/document/9854891) (IEEE Transactions on Geoscience and Remote Sensing)

## Experiments
### Initial steps
- Add the base directory and paths to train and evaluation path files in "data/datasets.yaml".
- For each experiment we use a separate ".yaml" configuration file. Examples files are providedided in "configs". 
The default values filled in these files correspond to parameters used in the experiments presented in the paper.
- activate "deepsatmodels" python environment:

	conda activate deepsatmodels

### Model training
Modify respective .yaml config files accordingly to define the save directory or loading a pre-trained model from pre-trained checkpoints. 

#### Randomly initialized "UNet3D" model 
	python train_and_eval/segmentation_training.py --config_file configs/**/UNet3D.yaml --gpu_ids 0,1

#### Randomly initialized "UNet2D-CLSTM" model 
	python train_and_eval/segmentation_training.py --config_file configs/**/UNet2D_CLSTM.yaml --gpu_ids 0,1

### CSCL-pretrained "UNet2D-CLSTM" model 
- model pre-training
	```bash
	python train_and_eval/segmentation_cscl_training.py --config_file configs/**/UNet2D_CLSTM_CSCL.yaml --gpu_ids 0,1
    ```
	
- copy the path to the pre-training save directory in CHECKPOINT.load_from_checkpoint. This will load the latest saved model.
To load a specific checkpoint copy the path to the .pth file
	```bash
	python train_and_eval/segmentation_training.py --config_file configs/**/UNet2D_CLSTM.yaml --gpu_ids 0,1
    ```

#### Randomly initialized "UNet3Df" model 
	python train_and_eval/segmentation_training.py --config_file configs/**/UNet3Df.yaml --gpu_ids 0,1

### CSCL-pretrained "UNet3Df" model 
- model pre-training

	```bash
	python train_and_eval/segmentation_cscl_training.py --config_file configs/**/UNet3Df_CSCL.yaml --gpu_ids 0,1
    ```
	
- copy the path to the pre-training save directory in CHECKPOINT.load_from_checkpoint. This will load the latest saved model.
To load a specific checkpoint copy the path to the .pth file
	```bash
	python train_and_eval/segmentation_training.py --config_file configs/**/UNet3Df.yaml --gpu_ids 0,1
    ```

## BibTex
If you incorporate any data or code from this repository into your project, please acknowledge the source by citing the following work:
```
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
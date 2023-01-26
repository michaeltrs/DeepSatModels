# Code for paper "ViTs for SITS: Vision Transformers for Satellite Image Time Series"

## Model training and evaluation
Modify respective .yaml config files accordingly to define the save directory or loading a pre-trained model from pre-trained checkpoints. 

### Semantic segmentation
	`python train_and_eval/segmentation_training_transf.py --config_file configs/**/TSViT.yaml --gpu_ids 0,1`

### Object classification
	`python train_and_eval/segmentation_training.py --config_file configs/**/TSViT_cls.yaml --gpu_ids 0,1`
	
## PASTIS benchmark

### Dataset
The [PASTIS dataset ](https://github.com/VSainteuf/pastis-benchmark) contains images from four different regions in France with diverse climate and 
crop distributions, spanning over 4000 km<sup>2</sup> and including 18 crop types. In total, it includes 2.4k SITS samples of 
size 128x128, each containing 33-61 acquisitions and 10 image bands. Because the PASTIS sample size is too 
large for efficiently training TSViT with consumer gpus, we split each sample into 24x24 patches and retain 
all acquisition times for a total of 60k samples. We further refer to these data as PASTIS24. 
Data with train, evaluation and test splits can be downloaded from [here](https://drive.google.com/file/d/1Av9hou8DviCJsEB9a_XU9SyqTuNxVpdE/view?usp=share_link). 
Unzipping in place will create the following folder tree.
```bash
PASTIS24
├── pickle24x24
│   ├── 40562_9.pickle
|   └── ...
├── fold-paths
│   ├── fold_1_paths.csv
|   └── ...
```
Alternatively, the data can be recreated from the PASTIS benchmark by running

	`python data/PASTIS24/data2windows.py --rootdir <...> --savedir <...> --HWout 24`

### Experiments
Run the following to train TSViT on each of the five folds of PASTIS24 

	`python train_and_eval/segmentation_training_transf.py --config_file configs/PASTIS24/TSViT_fold*.yaml --gpu_ids 0,1`

### Pre-trained checkpoints
Download 5-fold PASTIS24 [pre-trained models and tensorboard files](https://drive.google.com/file/d/1AzWEtHxojuCjaIsekja4J54LuEb9e7kw/view?usp=share_link).

## Reference
Please consider citing the following work if you use TSViT or code from this repository in your project: 
```
@misc{https://doi.org/10.48550/arxiv.2301.04944,
  doi = {10.48550/ARXIV.2301.04944},
  url = {https://arxiv.org/abs/2301.04944},
  author = {Tarasiou, Michail and Chavez, Erik and Zafeiriou, Stefanos},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ViTs for SITS: Vision Transformers for Satellite Image Time Series},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
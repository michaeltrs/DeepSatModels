# [ViTs for SITS: Vision Transformers for Satellite Image Time Series](https://arxiv.org/abs/2301.04944) (CVPR 2023)
- [download supp. material here](https://drive.google.com/file/d/1WAbNkfu1ko5uMS0e8GnKyzqroRUwNrRZ/view?usp=sharing)

## Model training and evaluation
Modify respective .yaml config files accordingly to define the save directory or loading a pre-trained model from pre-trained checkpoints. 

### Semantic segmentation
	python train_and_eval/segmentation_training_transf.py --config_file configs/**/TSViT.yaml --gpu_ids 0,1

### Object classification
	python train_and_eval/segmentation_training.py --config_file configs/**/TSViT_cls.yaml --gpu_ids 0,1
	
## PASTIS benchmark

### Dataset
The [PASTIS dataset](https://github.com/VSainteuf/pastis-benchmark) contains images from four different regions in France with diverse climate and 
crop distributions, spanning over 4000 km<sup>2</sup> and including 18 crop types. In total, it includes 2.4k SITS samples of 
size 128x128, each containing 33-61 acquisitions and 10 image bands. Because the PASTIS sample size is too 
large for efficiently training TSViT with consumer gpus, we split each sample into 24x24 patches and retain 
all acquisition times for a total of 60k samples. We further refer to these data as PASTIS24. 
Data with train, evaluation and test splits can be downloaded from [here](https://drive.google.com/drive/folders/1Lm0repzD_1NVcECsrwF8Q3bP2XEp2a0y). 
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

	python data/PASTIS24/data2windows.py --rootdir <...> --savedir <...> --HWout 24

When utilizing PASTIS24 in your projects, kindly ensure that proper credit is given to the  [PASTIS dataset](https://github.com/VSainteuf/pastis-benchmark). 


### Experiments
Run the following to train TSViT on each of the five folds of PASTIS24 

	python train_and_eval/segmentation_training_transf.py --config_file configs/PASTIS24/TSViT[-S]_fold*.yaml --gpu_ids 0,1

Omit the "-S" for the standard TSViT configuration, or include it (TSViT-S) for training the small architecture.
#### Results
| Model name         | #Params| OA  |  mIoU |
| ------------------ |---- |---- | ---| 
| TSViT   | 1.657M|    83.4%    |  65.4%| 
| TSViT-S   | 0.436M|    83.05%    |  64.3%| 
| [U-TAE](https://github.com/VSainteuf/utae-paps)   |   1.1M| 83.2%   | 63.1%|

### Pre-trained checkpoints
Download 5-fold PASTIS24 [pre-trained models and tensorboard files](https://drive.google.com/file/d/1AzWEtHxojuCjaIsekja4J54LuEb9e7kw/view?usp=share_link).


## BibTex
If you incorporate any data or code from this repository into your project, please acknowledge the source by citing the following work:

```
@misc{tarasiou2023vits,
      title={ViTs for SITS: Vision Transformers for Satellite Image Time Series}, 
      author={Michail Tarasiou and Erik Chavez and Stefanos Zafeiriou},
      year={2023},
      eprint={2301.04944},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


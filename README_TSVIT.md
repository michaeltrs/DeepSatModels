# ViTs for SITS: Vision Transformers for Satellite Image Time Series

This repository provides the implementation and datasets for the paper 
[ViTs for SITS: Vision Transformers for Satellite Image Time Series](https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html) 
presented at CVPR 2023. 
## Model Training and Evaluation

To begin model training or evaluation, adjust the `.yaml` configuration files as necessary to specify the save directory or to load a pre-trained model from available checkpoints.

### Semantic Segmentation

To train for semantic segmentation, execute the following command, replacing `**` with the appropriate directory names:

        python train_and_eval/segmentation_training_transf.py --config_file configs/**/TSViT.yaml --gpu_ids 0,1
    
### Object Classification

For object classification tasks, use the command below:

        python train_and_eval/classification_train_transf.py --config_file configs/**/TSViT_cls.yaml --gpu_ids 0,1
        
## PASTIS Benchmark

### Dataset Overview

The PASTIS dataset, accessible [here](https://github.com/VSainteuf/pastis-benchmark), 
comprises satellite image time series (SITS) samples from four distinct regions across 
France, showcasing a wide range of climates and crop distributions. The dataset spans 
more than 4000 km<sup>2</sup>, featuring 18 different crop types across 2.4k SITS 
samples of 128x128 pixels, each with 33-61 acquisitions and 10 image bands. 
To facilitate training on consumer-grade GPUs, we divide each sample into 24x24 
patches, generating approximately 60,000 samples in total, referred to as PASTIS24. 
The dataset, including training, evaluation, and test splits, is available for 
download [here](https://drive.google.com/drive/folders/1Lm0repzD_1NVcECsrwF8Q3bP2XEp2a0y). Extracting the files will organize them as shown below:

```bash
PASTIS24
├── pickle24x24
│   ├── 40562_9.pickle
│   └── ...
├── fold-paths
│   ├── fold_1_paths.csv
│   └── ...
```

To recreate the dataset from the original PASTIS benchmark, run:

        python data/PASTIS24/data2windows.py --rootdir <...> --savedir <...> --HWout 24

Please ensure to cite the PASTIS dataset appropriately when using PASTIS24 in your research.

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
@InProceedings{Tarasiou_2023_CVPR,
    author    = {Tarasiou, Michail and Chavez, Erik and Zafeiriou, Stefanos},
    title     = {ViTs for SITS: Vision Transformers for Satellite Image Time Series},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {10418-10428}
}
```


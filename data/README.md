## Datasets

### MTLCC dataset (Germany)

#### Download the dataset (.tfrecords)
The data for Germany can be downloaded from: https://github.com/TUM-LMF/MTLCC

- clone the repository in a separate directory:

	`git clone https://github.com/TUM-LMF/MTLCC`

- move to the MTLCC root directory:

	`cd MTLCC`

- download the data (40 Gb):

	`bash download.sh full`

#### Transform the dataset (.tfrecords -> .pkl)
- go to the "CSCL_code" home directory:

	`cd <.../CSCL_code>`

- activate the "cssl" python environment:

	`conda activate cscl`

- add "CSCL_code" home directory to PYTHONPATH:

	`export PYTHONPATH="<.../CSCL_code>:$PYTHONPATH"`

- Run the "data/MTLCC/make_pkl_dataset.py" script. Parameter `numworkers` defines the number of parallel processes employed:

	`python data/MTLCC/make_pkl_dataset.py --rootdir <.../MTLCC>  --numworkers <numworkers(int)-default-is-4>`

- Running the above script will have the following effects:
	- will create a paths file for the tfrecords files in ".../MTLCC/data_IJGI18/datasets/full/tfrecords240_paths.csv"
	- will create a new directory to save data ".../MTLCC/data_IJGI18/datasets/full/240pkl"
	- will save data in ".../MTLCC/data_IJGI18/datasets/full/240pkl/<data16, data17>"
	- will save relative paths for all data, train data, eval data in ".../MTLCC/data_IJGI18/datasets/full/240pkl"

### T31TFM_1618 dataset (France)
#### Download the dataset
The T31TFM_1618 dataset can be downloaded from Google drive [here](https://drive.google.com/file/d/1aSYbTwqT8QxN07D8LxZTjhIpc2xfcF-K/view?usp=sharing). Unzipping will create the following folder tree.
```bash
T31TFM_1618
├── 2016
│   ├── pkl_timeseries
│       ├── W799943_N6568107_E827372_S6540681
│       |   └── 6541426_800224_2016.pickle
|       |   └── ...
|       ├── ...
├── 2017
│   ├── pkl_timeseries
│       ├── W854602_N6650582_E882428_S6622759
│       |   └── 6623702_854602_2017.pickle
|       |   └── ...
|       ├── ...
├── 2018
│   ├── pkl_timeseries
│       ├── W882228_N6595532_E909657_S6568107
│       |   └── 6568846_888751_2018.pickle
|       |   └── ...
|       ├── ...
├── deepsatdata
|   └── T31TFM_16_products.csv
|   └── ...
|   └── T31TFM_16_parcels.csv
|   └── ...
└── paths
    └── train_paths.csv
    └── eval_paths.csv
```

#### Recreate the dataset from scratch
To recreate the dataset use the [DeepSatData](https://github.com/michaeltrs/DeepSatData) data generation pipeline. 
- Clone and move to the DeepSatData base directory
```bash
git clone https://github.com/michaeltrs/DeepSatData
cd .../DeepSatData
```
- Download the Sentinel-2 products.
```bash 
sh download/download.sh .../T31TFM_16_parcels.csv,.../T31TFM_17_parcels.csv,.../T31TFM_18_parcels.csv
``` 
- Generate a labelled dataset (use case 1) for each year.
```bash
sh dataset/labelled_dense/make_labelled_dataset.sh ground_truths_file=<1:ground_truths_file> products_dir=<2:products_dir> labels_dir=<3:labels_dir> windows_dir=<4:windows_dir> timeseries_dir=<5:timeseries_dir> 
res=<6:res> sample_size=<7:sample_size> num_processes<8:num_processes> bands=<8:bands (optional)>
```

### PASTIS dataset (France)
#### Download the dataset
The PASTIS dataset can be downloaded from [here](https://github.com/VSainteuf/pastis-benchmark). 

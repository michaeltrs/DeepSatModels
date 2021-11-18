# crop-type-mapping

** NOTE: We are in the process of cleaning the code base! The dataset associated with this project will be released soon. Details to follow. ** 

Crop type mapping of small holder farms in Ghana and South Sudan

##### INSTALLATION INSTRUCTIONS #####

Install Python 3.6

Install conda and build the environment with the following command:

`conda env create -f environment.yaml`

##### DATASET / ENVIRONMENT SETUP #####

To install and format data, follow instructions in environment_setup.txt

Data directories should be stored in the root /home folder. 

For example, the folder `/home/data/ghana/` should house all the split information, data.hdf5 files, original data files, etc. for ghana. 

##### RUN INSTRUCTIONS #####

To visualize training, open a separate terminal and run the following before running the main training code:

  `python -m visdom.server`

Replace “localhost” with the static IP address provided on google cloud

To start training models, use the train.py script in the root directory of the code. 

Example for CLSTM-only network:
```python train.py --model_name=only_clstm_mi --country=southsudan --var_length --name=southsudan_clstmonly --env_name=myenv --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --lr=0.003 --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True```

Example for 3D UNet model: 
```python train.py --model_name=unet3d --country=southsudan --num_timesteps=24 --lr=0.0003 --s2_agg=False --include_indices=True --include_doy=True --use_planet=True --planet_agg=False --name=southsudan_3dunet_use_planet_noagg --env_name=myenv --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True --hidden_dims=128```

Example for Multi-Input 2D UNet + CLSTM model:
`python train.py --model_name=mi_clstm --country=ghana --var_length --main_crnn=True --early_feats True --include_indices=True --include_doy False --sample_w_clouds False --include_clouds False --lst_cloudy False --use_planet=True --resize_planet=False --name=ghana_use_planet_highres --env_name=ghana_use_planet_highres --use_s1=True --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --lr=0.003 --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True`

Example for Multi-Input 2D UNet + CLSTM earlier fused model:
```python train.py --model_name=fcn_crnn --country=southsudan --name=southsudan_fcn_crnn --env_name=myenv --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --lr=0.001 --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True --include_s1=True```

Model type and country are set with the `model_name` and `country` flags, respectively. Flags also control several properties of inputs such as satellite types, temporal aggregation, Planet imagery resolution, cloud sampling, etc. Additional hyperparameter tuning settings can be set by invoking the appropriate flags. See `get_train_parser` in `util.py` for full details. 


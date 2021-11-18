"""
Wrapper script for performing random search. 

Hyperparameters should be specified as --hpname_range="(...)" and added to the appropriate list in the constants.py file.

The types of values that can be generated are specified in the functions below. 

run with:

python random_search.py --model_name fcn_crnn --dataset full --epochs 2 --batch_size_range="(1, 5)" --crnn_num_layers_range="(1, 1)" --lr_range="(10, -5, -1)" --hidden_dims_range="(2, 3, 7)" --weight_scale_range="(.5, 2)" --gamma_range="(0, 2)" --weight_decay_range="(10, -5, 0)" --momentum_range="(.5, .999)" --optimizer_range="('adam', 'adam')" --num_samples=3 --patience_range="(1, 5)" --use_s1_range="()" --use_s2_range="()" --apply_transforms_range="()" --sample_w_clouds_range="()" --include_clouds_range="()" --include_doy_range="()" --bidirectional_range="()"


"""


import argparse
import datetime
import os
import train 
import pickle
import numpy as np
import util
import datasets
import os
import models
import torch
import sys

from constants import *
from ast import literal_eval

def generate_int_power_HP(base, minVal, maxVal):
    """ Generates discrete values in the range (base^minVal, base^maxVal).
    """
    exp = np.random.randint(minVal, maxVal + 1)
    return base ** exp

def generate_real_power_HP(base, minVal, maxVal):
    """ Generates continuous vals in range (base^minVal, base^maxVal).
    """
    exp = np.random.uniform(minVal, maxVal)
    return base ** exp

def generate_int_HP(minVal, maxVal):
    """ Generates discrete vals in range (minVal, maxVal) inclusive.
    """
    return np.random.randint(minVal, maxVal + 1)

def generate_float_HP(minVal, maxVal):
    """ Generates continuous vals in range (minVal, maxVal) inclusive.
    """
    return np.random.uniform(minVal, maxVal)

def generate_string_HP(choices):
    """ Chooses from one of the choices in `choices`.
    """
    return np.random.choice(choices)

def generate_bool_HP(choices):
    """ Chooses from one of the choices in `choices`.
    """
    return np.random.choice(choices)

def generate_int_choice_HP(choices):
    """ Chooses from one of the choices in `choices` and casts to int.
    """
    return int(np.random.choice(choices))

def str2tuple(arg):
    """ Converts a tuple in string format to a tuple.

    Ex: "(2, 2)" => (2, 2) (as a tuple)

    Requires " " around the parenthesis.
    """
    return literal_eval(arg)

def recordMetadata(args, experiment_name, hps, train_loss, train_f1, train_acc, val_loss, val_f1, val_acc):
    with open(os.path.join(args.save_dir, experiment_name + ".log"), 'w') as f:
        f.write('HYPERPARAMETERS:\n')
        for hp in hps:
            hp_val = args.__dict__[hp]
            if type(hp_val) == float:
                hp_val = '%.3f'%hp_val 
            f.write(f'{hp}:{hp_val}\n')
        f.write(f"Best Performance (val): \n\t loss: {val_loss} \n\t f1: {val_f1}\n\t acc:{val_acc}\n")
        f.write(f"Corresponding Train Performance: \n\t loss: {train_loss} \n\t f1: {train_f1}\n\t acc:{train_acc}\n")

def generate_hps(train_args, search_range):
    for arg in vars(search_range):
        if "range" not in arg: continue
        hp = arg[:arg.find("range") - 1]
        if hp in INT_POWER_EXP:
            hp_val = generate_int_power_HP(vars(search_range)[arg][0], vars(search_range)[arg][1], vars(search_range)[arg][2])
        elif hp in REAL_POWER_EXP:
            hp_val = generate_real_power_HP(vars(search_range)[arg][0], vars(search_range)[arg][1], vars(search_range)[arg][2])
        elif hp in INT_HP:
            hp_val = generate_int_HP(vars(search_range)[arg][0], vars(search_range)[arg][1])
        elif hp in FLOAT_HP:
            hp_val = generate_float_HP(vars(search_range)[arg][0], vars(search_range)[arg][1])
        elif hp in STRING_HP:
            hp_val = generate_string_HP(vars(search_range)[arg])
        elif hp in BOOL_HP:
            hp_val = generate_bool_HP(vars(search_range)[arg])
        elif hp in INT_CHOICE_HP:
            hp_val = generate_int_choice_HP(vars(search_range)[arg])
        else:
            raise ValueError(f"HP {hp} unsupported") 

        train_args.__dict__[hp] = hp_val

    if not train_args.__dict__['use_s1'] and not train_args.__dict__['use_s2']:
        train_args.__dict__[np.random.choice(['use_s1', 'use_s2'])] = True

if __name__ ==  "__main__":
    # get all ranges of values
    search_parser = argparse.ArgumentParser()
    search_parser.add_argument('--model_name', type=str)
    search_parser.add_argument('--dataset', type=str)
    search_parser.add_argument('--num_samples', type=int,
                        help="number of random searches to perform")
    search_parser.add_argument('--epochs', type=int,
                        help="number of epochs to train the model for")
    search_parser.add_argument('--logfile', type=str,
                        help="file to write logs to; if not specified, prints to terminal")
    search_parser.add_argument('--hp_dict_name', type=str,
                        help="name of hp dict, defaults to hp_results.pkl if unspecified",
                        default="hp_results.pkl")
    search_parser.add_argument('--env_name', type=str,
                        default=None)
    search_parser.add_argument('--country', type=str,
                        default="ghana")
    for hp_type in HPS:
        for hp in hp_type:
            search_parser.add_argument('--' + hp + "_range", type=str2tuple)
    search_range = search_parser.parse_args()
    #TODO: VERY HACKY, SWITCH TO USING PYTHON LOGGING MODULE OR ACTUALLY USING WRITE CALLS
    # CURRENTLY CHANGES STDOUT OF THE PROGRAM
    old_stdout = sys.stdout
    if search_range.logfile is not None:
        logfile = open(search_range.logfile, "w")
        sys.stdout = logfile
   
    hps = {}
    for arg in vars(search_range):
        if "range" not in arg: continue 
        hp = arg[:arg.find("range") - 1]
        hps[hp] = [] 

    experiments = {}

    # for some number of iterations
    for sample_no in range(search_range.num_samples):
        # build argparse args by parsing args and then setting empty fields to specified ones above
        train_parser = util.get_train_parser()
        train_args = train_parser.parse_args(['--model_name', search_range.model_name, 
                                              '--dataset', search_range.dataset, 
                                              '--env_name', search_range.env_name,
                                              '--country', search_range.country])
        generate_hps(train_args, search_range) 
        train_args.epochs = search_range.epochs
        dataloaders = datasets.get_dataloaders(train_args.country, train_args.dataset, train_args)
        
        model = models.get_model(**vars(train_args))
        model.to(train_args.device)
        experiment_name = f"model:{train_args.model_name}_dataset:{train_args.dataset}_epochs:{search_range.epochs}_sample_no:{sample_no}"

        train_args.name = experiment_name
        print("="*100)
        print(f"TRAINING: {experiment_name}")
        for hp in hps:
            print(hp, train_args.__dict__[hp])
        try: 
            train.train(model, train_args.model_name, train_args, dataloaders=dataloaders) 
            print("FINISHED TRAINING") 
            for state_dict_name in os.listdir(train_args.save_dir):
                if (experiment_name + "_best") in state_dict_name:
                    model.load_state_dict(torch.load(os.path.join(train_args.save_dir, state_dict_name)))
                    train_loss, train_f1, train_acc = train.evaluate_split(model, train_args.model_name, dataloaders['train'], train_args.device, train_args.loss_weight, train_args.weight_scale, train_args.gamma, NUM_CLASSES[train_args.country], train_args.country, train_args.var_length)
                    val_loss, val_f1, val_acc = train.evaluate_split(model, train_args.model_name, dataloaders['val'], train_args.device, train_args.loss_weight, train_args.weight_scale, train_args.gamma, NUM_CLASSES[train_args.country], train_args.country, train_args.var_length)
                    print(f"Best Performance (val): \n\t loss: {val_loss} \n\t f1: {val_f1}\n\t acc: {val_acc}")
                    print(f"Corresponding Train Performance: \n\t loss: {train_loss} \n\t f1: {train_f1}\n\t acc: {train_acc}")

                    recordMetadata(train_args, experiment_name, hps, train_loss, train_f1, train_acc, val_loss, val_f1, val_acc)

                    experiments[experiment_name] = [train_loss, train_f1, train_acc, val_loss, val_f1, val_acc]
                    for hp in hps:
                        hps[hp].append([train_args.__dict__[hp], train_loss, train_f1, train_acc, val_loss, val_f1, val_acc])
                    break


        except Exception as e:
            print("CRASHED!")
            print(e)

        torch.cuda.empty_cache()
        
        with open(search_range.hp_dict_name, "wb") as f:
            pickle.dump(hps, f)

    print("SUMMARY")

    for key, value in sorted(experiments.items(), key=lambda x: x[1][-1], reverse=True):
        print(key, "\t Val:", value[-2], "\t", value[-1], "\t Train: ", value[0], "\t", value[1])
    
    with open(search_range.hp_dict_name, "wb") as f:
        pickle.dump(hps, f)

    sys.stdout = old_stdout
    if search_range.logfile is not None:
        logfile.close()

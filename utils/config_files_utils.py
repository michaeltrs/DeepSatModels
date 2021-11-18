from yaml import load, dump
import os


def get_params_values(args, key, default=None):
    """
    set default to None if a value is required in the config file
    """
    if (key in args) and (args[key] is not None):
            return args[key]
    return default
    #else:
    #    raise ValueError("No value provided in config file for %s, default value not provided")


#yaml_file = 'configs/test.yaml'
def read_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        yaml_dict = load(config_file)
    return yaml_dict


def copy_yaml(config_file):
    """
    copies config file to training savedir
    """
    if type(config_file) is str:
        yfile = read_yaml(config_file)
    elif type(config_file) is dict:
        yfile = config_file
    save_name = yfile['CHECKPOINT']['save_path'] + "/config_file.yaml"
    i = 1
    while os.path.isfile(save_name):
        save_name = "%s_%d.yaml" % (save_name[:-5], i)
        i += 1
    with open(save_name, 'w') as outfile:
        dump(yfile, outfile, default_flow_style=False)

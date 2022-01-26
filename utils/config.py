import yaml
import os
import hashlib
import json

def create_exp_from_config(config_file, exp_dir):
    """generate experiment from template, save to hashed exp folder
    """
    config = load_config(config_file)
    exp_hash = save_config(config, exp_dir)
    
    return exp_hash

def config_path(exp_dir, exp_name):
    return os.path.join(exp_dir, exp_name, 'config.yaml')    

def load_config(params_file):
    with open(params_file) as fhandle:
        params = yaml.safe_load(fhandle)
                         
    return params

def save_config(config_dict, exp_dir):
    exp_hash = hash_dict(config_dict)
    
    log_dir = os.path.join(exp_dir, exp_hash)
    os.makedirs(log_dir, exist_ok=True)
    
    config_file = config_path(exp_dir, exp_hash)

    with open(config_file, 'w') as fhandle:
        yaml.dump(config_dict, fhandle, default_flow_style=False)
        
    return exp_hash

def hash_dict(nested_dict):
    """ Returns hash of nested dict, given that all keys are strings
    """    

    dict_string = json.dumps(nested_dict, sort_keys=True)
    md5_hash = hashlib.md5(dict_string.encode()).hexdigest()

    return md5_hash


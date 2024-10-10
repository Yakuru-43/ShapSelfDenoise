import yaml
import argparse
from src.model import Alpaca
import torch
import numpy as np
import pandas as pd 
import random
import os
from tqdm import tqdm

def set_random_seed(seed, deterministic=False, no_torch=False):
    """
    Set the random seed for the reproducibility. Environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 is also needed.
    :param seed: the random seed
    :type seed: int
    :param deterministic: whether use deterministic, slower is True, cannot guarantee reproducibility if False
    :param no_torch: if torch is not installed, set this True
    :param no_tf: if tensorflow is not installed, set this True
    :type deterministic: bool
    """
    random.seed(seed)
    np.random.seed(seed)
    if not no_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def load_config(config_file_path):
    """Load the configuration from a YAML file."""
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def check_existance(file_path) :
    """Check if a file exists at the specified path."""
    return os.path.exists(file_path)   

data = []

def main():
    tqdm.pandas()
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="A script that loads a config file and reads arguments.")
    parser.add_argument('--model', type=str, help='The name of the Huggingface repo containing the model.')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file.')
    parser.add_argument('--batchsize', type=int, default=3, help='The batchsize used.')
    parser.add_argument('--precision', type=str, default="full", choices=['full', 'half'],  help='half=float16, full=float32')
    parser.add_argument('--dataset', type=str, choices=["agnews","sst2"], help='The dataset to be used.')
    parser.add_argument('--maskrate', type=float, default=0.1, help="The rate of words to mask")
    parser.add_argument('--mask_word', type=str, default="<mask>", choices=["<mask>","###"])
    
    args = parser.parse_args()

    # Set random seed
    set_random_seed(1)

    # Load configuration from YAML file
    config = load_config(args.config)

    # Load dataset
    dataset = pd.read_json("dataset/agnews_raw/dataset.json",orient='records', lines=True)

    # Load the model
    model = Alpaca(args, config["model"])

    # Setup the model for the appropriate dataset
    model.as_agnews(args.mask_word)

    # Save the dataset as a json file 
    if check_existance('out/dataset_masked.json') :
        dataset = pd.read_json("out/dataset_masked.json",orient='records', lines=True)
    else :
        # Mask the sentences 
        dataset['masked'] = dataset["text"].progress_apply(lambda x: model.shap_masking(x, args.maskrate))
        dataset.to_json('out/dataset_masked.json', orient='records', lines=True)
        print(dataset.head())

    if check_existance('out/dataset_denoised.json') :
        dataset = pd.read_json("out/dataset_denoised.json",orient='records', lines=True)
    else :
        # Denoise the sentences
        print("Denoising :")
        dataset['denoised'] = dataset['masked'].progress_apply(model.denoise_sentence)
        dataset.to_json('out/dataset_denoised.json', orient='records', lines=True)
    
    # Do the prediction 
    dataset['prediction'] = dataset['denoised'].progress_apply(model.classify_sentence)
    dataset.to_json('out/dataset_prediction.json', orient='records', lines=True)
    
if __name__ == "__main__":
    main()
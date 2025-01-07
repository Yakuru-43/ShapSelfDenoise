import yaml
import argparse
from src.model import Alpaca
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def load_config(config_file_path):
    """Load the configuration from a YAML file."""
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_seed(seed=42):
    """Set the random seed to ensure reproducibility."""

    np.random.seed(seed)


def check_existance(file_path):
    """Check if a file exists at the specified path.
    input: file_path
    output: boolean
    """
    return os.path.exists(file_path)


def parse_args():
    """Parse command-line arguments.

    Returns:
        args: The command-line arguments passed to the script.
    """
    parser = argparse.ArgumentParser(
        description="A script that loads a config file and reads arguments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["attack", "certify"],
        required=True,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to the config file."
    )
    parser.add_argument("--batchsize", type=int, default=3, help="The batchsize used.")
    parser.add_argument("--sample-size", type=int, default=100, help="The sample size.")
    parser.add_argument(
        "--precision",
        type=str,
        default="full",
        choices=["full", "half"],
        help="half=float16, full=float32",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["agnews", "sst2"],
        help="The dataset to be used.",
    )
    parser.add_argument(
        "--maskrate", type=float, default=0.1, help="The rate of words to mask"
    )
    parser.add_argument(
        "--mask_word", type=str, default="<mask>", choices=["<mask>", "###"]
    )
    return parser.parse_args()


def setup_model(args, config):
    """Setup the model for the appropriate dataset.

    Args:
        args: The command-line arguments passed to the script.
        config: The configuration dictionary loaded from the YAML file.

    Returns:
        model: The initialized model.
    """
    # Load the model
    model = Alpaca(args, config["model"])

    # Setup the model for the appropriate dataset
    if args.dataset == "agnews":
        model.as_agnews(args.mask_word)
    elif args.dataset == "sst2":
        model.as_sst2()

    return model


def load_dataset(args, config):
    """
    Load the dataset for the appropriate dataset.

    Args:
        args: The command-line arguments passed to the script.
        config: The configuration dictionary loaded from the YAML file.

    Returns:
        dataset: The dataset as a Pandas DataFrame.
    """
    # Load the dataset for the appropriate dataset
    if args.dataset == "agnews":
        dataset = pd.read_csv(config["agnews_path"])

        # Rename the 'Description' column to 'text'
        dataset.rename(columns={"Description": "text"}, inplace=True)

        # Rename the 'Class Index' column to 'label'
        dataset.rename(columns={"Class Index": "label"}, inplace=True)

    elif args.dataset == "sst2":
        dataset = pd.read_csv(
            config["sst2_path"], sep="\t", header=None, names=["text", "label"]
        )

    # Check if the sample size is larger than the dataset size
    if args.sample_size > len(dataset):
        args.sample_size = len(dataset)
        print(
            f"The sample size is larger than the dataset size. Setting sample size to {len(dataset)}"
        )

    # shuffle the dataset and keep only the sample size
    dataset = dataset.sample(args.sample_size).reset_index(drop=True)

    return dataset


def save_results(args, mask_rate, results):
    """
    Save the results to a CSV file.

    Args:
        args: The command-line arguments passed to the script.
        results: The results as a Pandas DataFrame.
    """
    # setp up folder name with dataset name + mask rate + sample size
    save_folder_path = f"out/{args.mode}_{args.dataset}_mask_rate_{mask_rate}_sample_size_{args.sample_size}"
    # Create the folder if it doesn't exist in the /out folder
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Save the results to a CSV file
    results.to_json(save_folder_path + "/results.jsonl", lines=True, orient="records")

    # Calculate accuracy
    accuracy = (results["prediction"] == results["label"]).mean()

    # Log the args with the accuracy
    with open(save_folder_path + "/results.log", "w") as log_file:
        for key, value in vars(args).items():
            log_file.write(f"{key} = {value}\n")
        log_file.write(f"Mask Rate: {mask_rate}\n")
        log_file.write("\n")
        log_file.write(f"Accuracy: {accuracy}")

    print("Results saved to", save_folder_path)


def certify(args, config):
    """
    Certify the model on the dataset.

    Args:
        args: The command-line arguments passed to the script.
        config: The configuration dictionary loaded from the YAML file.
        model: The initialized model.
    """
    # Load dataset
    # dataset = load_dataset(args, config)
    dataset = pd.read_json("dataset/agnews_raw/dataset.json", orient="records", lines=True)

    # Setup the model for the appropriate dataset
    model = setup_model(args, config)

    # Set up progress bar
    tqdm.pandas()   
    
    # Adding column shapley_values setup with None value 
    dataset['shapley_values'] = None

    # Set up the mask rate
    mask_rate = 0.1
    # Loop on the mask rates from 0.1 to 0.9 with 0.1 step
    while mask_rate <= 0.9:
        print(f"For the mask rate: {mask_rate}")

        # Mask the sentences
        tqdm.pandas(desc="Masking the sentences...")
        dataset[['masked', 'shapley_values']] = dataset.progress_apply(
            lambda row: model.shap_masking(row['text'], mask_rate, row['shapley_values']), axis=1
        ).apply(pd.Series)

        # Denoise the sentences
        tqdm.pandas(desc="Denoising the sentences...")
        dataset["denoised"] = dataset["masked"].progress_apply(model.denoise_sentence)

        # Do the prediction
        tqdm.pandas(desc="Classifying the sentences...")
        dataset["prediction"] = dataset["denoised"].progress_apply(model.classify_sentence)

        # Save the results
        save_results(args, mask_rate, dataset)

        # Update the mask rate
        mask_rate += 0.1

        # Delete column masked, denoised and prediction, this is purely out of safety reason it should not matter
        dataset = dataset.drop(columns=['masked', 'denoised', 'prediction'])


def attack(args, config):
    """
    Attack the model on the dataset.

    Args:
        args: The command-line arguments passed to the script.
        config: The configuration dictionary loaded from the YAML file.
        model: The initialized model.
    """
    # Load dataset
    pass

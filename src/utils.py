import yaml
import argparse
from .model import Alpaca
from .attack_runner import AttackRunner
import numpy as np
import pandas as pd
import csv
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
        choices=["attack", "certify","evaluate_lime"],
        required=True,
        help="The mode to run the script in.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default= 10,
        required=False,
        help="The number of times we want to evaluate the lime defense.",
    )
    parser.add_argument(
        "--defence",
        type=str,
        choices=["shap", "lime"],
        required=True,
        help="The defence mechanism to use.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["DeepWordBug", "TextBugger"],
        required=True,
        help="The method to attack the model.",
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
        print("Model setup for AGNews")
    elif args.dataset == "sst2":
        model.as_sst2()
        print("Model setup for SST2")

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
    if args.dataset == "agnews" :
        dataset = pd.read_json("dataset/agnews/dataset_certify.json", orient="records", lines=True)
    elif args.dataset == "sst2" :
        dataset = pd.read_json("dataset/sst2/dataset_certify.json", orient="records", lines=True)

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
    """
    # Load dataset
    # dataset = load_dataset(args, config) # This is how it should realy be loaded but we use this so that we can use the same dataset as SelfDenoise
    if args.dataset == "agnews" :
        dataset = pd.read_json("dataset/agnews/dataset_attack.json", orient="records", lines=True)
    elif args.dataset == "sst2" :
        dataset = pd.read_json("dataset/sst2/dataset_attack.json", orient="records", lines=True)

    # Setup the model for the appropriate dataset
    model = setup_model(args, config)
    
    # Instantiate the AttackRunner
    attack_runner = AttackRunner(model, dataset, args.method, args.dataset)
   
    # Run the attack    
    log_file_name = attack_runner.run_attack()

    print("Attack results saved to", log_file_name)
    
    # Open the csv log file with pandas
    df = pd.read_csv(log_file_name)

    # count the number of failed and successful attacks
    num_failed = df[df['result_type'] == "Failed"].shape[0]
    num_successful = df[df['result_type'] == "Successful"].shape[0]
    total  = df.shape[0]

    process_csv(log_file_name, model, total, args.defence)
    
    # count the number of rows where label_shap == ground_truth_output and result_type == Successful
    df = pd.read_csv(log_file_name)
    additional_fails_after_defence = df[(df['result_type'] == "Successful") & (df['label_shap'] == df['ground_truth_output'])].shape[0]
  
    print(f'Accuracy after SHAP {(additional_fails_after_defence + num_failed)/total}')

    dir_path = f"out/attack/{args.dataset}/{args.method}/SHAP_Defence/{args.precision}"
    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save the results
    file_path = os.path.join(dir_path, "results.txt")
    with open(file_path, "w") as f:
        f.write(f'Original accuracy : {(num_successful + num_failed)/total}\n')
        f.write(f'Accuracy under attack : {num_failed/total}\n')
        f.write(f'Accuracy with SHAP defence : {(additional_fails_after_defence + num_failed)/total}\n')

    # # Save the dataframe to a csv file
    # df.to_csv(os.path.join(dir_path, "dataset.csv"), index=False)

def defence(row, model, defence):
    """
    Apply the defence mechanism to the perturbed text and return the label.

    Args:
        row: The row of the dataset.
        model: The model to be used for classification.
        defence : The defence mechanism to be used.

    Returns:
        label: The label of the perturbed text after applying the defence mechanism.
    """
    # Remove the brackets from the perturbed text
    text = row['perturbed_text']
    text = text.replace('[',"")
    text = text.replace(']',"")

    # Apply our defence mechanism
    if defence == "shap":
        masked_text = model.shap_masking(text, 0.05, None)
    elif defence == "lime" :
        masked_text = model.lime_masking(text, 0.05, None)
    denoised_text = model.denoise_sentence(masked_text)
    if model.dataset == "agnews":
        label = model.classify_sentence(denoised_text) -101
    elif model.dataset == "sst2":
        label = model.classify_sentence(denoised_text)

    return label


def process_csv(file_path, model, total, defence_method):
    """
    Processes a CSV file, calling function defence on each line, appending the result to the line,
    and writing it back to the CSV file. Designed to resume processing in case of interruption.
    
    :param file_path: The path to the CSV file to be processed.
    :param model: The model to be used for classification.
    :param total: The total number of rows in the CSV file.
    :param defence: The defence mechanism to be used.
    :return: A tuple containing the number of failed and successful attacks.
    """

    temp_file_path = file_path + '.tmp'

    # Check if a temporary file exists (from a previous interrupted run)
    if os.path.exists(temp_file_path):
        # Determine how many lines have been processed so far
        with open(temp_file_path, 'r', newline='', encoding='utf-8') as temp_file:
            processed_line_count = sum(1 for _ in temp_file) -1 # Subtract 1 for the header
            
    else:
        processed_line_count = 0

    # Open the original CSV and the temporary file
    with open(file_path, 'r', newline='', encoding='utf-8') as csv_in, \
         open(temp_file_path, 'a', newline='', encoding='utf-8') as csv_out:

        reader = csv.DictReader(csv_in)
        fieldnames = reader.fieldnames + ['label_shap']
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        if processed_line_count == 0:
            writer.writeheader()

        # Skip lines that have already been processed
        for _ in range(processed_line_count):
            next(reader)

        # Process remaining lines
        for row in tqdm(reader, total=total- processed_line_count , desc="Processing rows", miniters=1):
            # We only apply the defence mechanism to successful attacks
            if row["result_type"] == "Successful" :
                # Call our defence on the current row   
                label = defence(row, model, defence_method)
                # Append the result to the row
                row['label_shap'] = label
                # Write the updated row to the temporary file
                writer.writerow(row)

            else :
                # Write the row to the temporary file without applying the defence mechanism and with empty label_shap
                writer.writerow(row)
                
    # Replace the original file with the fully processed temporary file
    os.replace(temp_file_path, file_path)

def evaluate_lime_defense(n, args, config) :
    """
    Since LIME method is not deterministic we need to evaluate it n times and then do a statistical analysis about the distribiton to 
    validate the accuracy given by lime defense.

    Input : 
    - n the number of times we want to evaluate the lime defense
    - args : the command line arguments
    - config : the configuration dictionary loaded from the YAML file.
    """ 
        # Load dataset
    # dataset = load_dataset(args, config) # This is how it should realy be loaded but we use this so that we can use the same dataset as SelfDenoise
    if args.dataset == "agnews" :
        dataset = pd.read_json("dataset/agnews/dataset_attack.json", orient="records", lines=True)
    elif args.dataset == "sst2" :
        dataset = pd.read_json("dataset/sst2/dataset_attack.json", orient="records", lines=True)

    # Setup the model for the appropriate dataset
    model = setup_model(args, config)

    # # Instantiate the AttackRunner
    # attack_runner = AttackRunner(model, dataset, args.method, args.dataset)
    
    # # Release all data from the GPU

    # # Run the attack    
    # log_file_name = attack_runner.run_attack()

    # print("Attack results saved to", log_file_name)

    log_file_name = "out/attack/agnews/DeepWordBug/NoDefence/half/dataset.csv"
    # Open the csv log file with pandas
    df = pd.read_csv(log_file_name)

    accuracies = []
    for i in range(n):

        # count the number of failed and successful attacks
        num_failed = df[df['result_type'] == "Failed"].shape[0]
        total  = df.shape[0]
        dataset = []

        with open(log_file_name, 'r', newline='', encoding='utf-8') as csv_in:
            reader = csv.DictReader(csv_in)
            # fieldnames = reader.fieldnames + ['label_shap']

            for row in tqdm(reader, total=total, desc="Processing rows", miniters=1):
                # Apply the defence mechanism to successful attacks
                if row["result_type"] == "Successful":
                    # Call the defence function on the current row
                    label = defence(row, model, args.defence)
                    # Append the result to the row
                    row['label_shap'] = label
                else:
                    # Set label_shap to an empty string for unsuccessful attacks
                    row['label_shap'] = ''
                dataset.append(row)

        df = pd.DataFrame(dataset)
        # Cast the ground_truth_output column to int because for some reason it is read as string
        df['ground_truth_output'] = df['ground_truth_output'].astype(int)
        additional_fails_after_defence = df[(df['result_type'] == "Successful") & (df['label_shap'] == df['ground_truth_output'])].shape[0]

        print(f'Accuracy after LIME for the iteration {i} is {(additional_fails_after_defence + num_failed)/total}')
        accuracies.append((additional_fails_after_defence + num_failed)/total)
    print(accuracies)
    # Save accuracies to a file for statistical analysis
    dir_path = f"out/evaluate/{args.dataset}/{args.method}/LIME_Defence/{args.precision}"
    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    # Save the accuracies to a file

    file_path = os.path.join(dir_path, "accuracies.txt")
    with open(file_path, "w") as f:
        for accuracy in accuracies:
            f.write(f"{accuracy}\n")

    print(f"Accuracies saved to {file_path}")

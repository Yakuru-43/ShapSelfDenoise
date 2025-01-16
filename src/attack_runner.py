from .llama_model_wrapper import LLAMAModelWrapper
import textattack
import pandas as pd

class AttackRunner:
    def __init__(self, model, dataset, attack_method):
        # Format dataset 
        self.dataset = self.format_dataset(dataset)
        
        # Set up modelWrapper
        self.model_wrapper = LLAMAModelWrapper(model.alpaca_model, model.alpaca_tokenizer, model)

        # Set up attack method
        attack_methods = {
            "DeepWordBug": textattack.attack_recipes.DeepWordBugGao2018,
            "TextBugger": textattack.attack_recipes.TextBuggerLi2018
        }
        
        if attack_method in attack_methods:
            self.attack = attack_methods[attack_method].build(self.model_wrapper)
            self.attack_name = attack_method
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        pass

    def format_dataset(self, dataset) :
        """
            Format the dataset from a dataframe into a textattack dataset
            Input :
                dataset : pandas dataframe
            Output :
                textattack.dataset.Dataset
        """

        # Keep only the text and label columns
        dataset = dataset[['text', 'label']]

        # cast the label column to int and remove 100
        dataset.loc[:, 'label'] = dataset['label'].astype(int) - 101

        # format the dataset as a list of tuples
        result = [(data['text'], data['label']) for index, data in dataset.iterrows()]

        return textattack.datasets.Dataset(result)
         
    def run_attack(self):
        """
            Run the attack on the dataset and log the results.
            This method performs the following steps:
            1. Sets up the log file name based on the attack name and model precision.
            2. Configures the attack arguments.
            3. Executes the attack on the dataset.
            4. Reads the attacked dataset from the log file.
            5. Calculates statistics on the attacked dataset, including original accuracy and accuracy under attack.
            6. Writes the results to a text file.
            Returns:
                str: The path to the log file containing the attack results.
            Raises:
                FileNotFoundError: If the log file cannot be found after the attack.
                IOError: If there is an issue reading from or writing to the log file.
    
        """

        # Setup file name for the logs including date and time name of the attack and the model precision
        model_precision = self.model_wrapper.alpaca.precision
        log_file_name = f"out/attack/{self.dataset}/{self.attack_name}/NoDefense/{model_precision}/dataset.csv"

        # Setup the attack arguments 
        attack_args = textattack.AttackArgs(
            num_examples=-1,
            log_to_csv=log_file_name,
            disable_stdout=False,
        )
        attacker = textattack.Attacker(self.attack, self.dataset, attack_args)

        # Run
        attacker.attack_dataset()

        attacked_dataset = pd.read_csv(log_file_name)

        # Calculate statistics on the attacked dataset
        result_counts = attacked_dataset['result_type'].value_counts()
        total = result_counts.sum()
        successful = result_counts.get('Successful', 0)
        failed = result_counts.get('Failed', 0)
        
        original_accuracy = (failed + successful) / total
        accuracy_under_attack = failed / total

        # Open a file or create it if it does not exist to store the results at the end of the file 
        with open(f"out/attack/{self.dataset}/{self.attack_name}/NoDefense/{model_precision}/results.txt", "w") as f:
            f.write(f'Original accuracy : {original_accuracy}\n')
            f.write(f'Accuracy under attack : {accuracy_under_attack}\n')

        return log_file_name
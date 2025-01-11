from .llama_model_wrapper import LLAMAModelWrapper
import textattack
import datetime

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
            Run the attack on the dataset
        """

        # Setup file name for the logs including date and time name of the attack and the model precision
        now = datetime.datetime.now().__str__()
        model_precision = self.model_wrapper.alpaca.precision
        log_file_name = f"out/attack/log_{now}_{self.attack_name }_{model_precision}.csv"

        # Setup the attack arguments 
        attack_args = textattack.AttackArgs(
            num_examples=-1,
            log_to_csv=log_file_name,
            checkpoint_interval=10,
            checkpoint_dir="checkpoints",
            disable_stdout=False,
            # Additional arguments can be added here if necessary
            # parallel=True,
            # num_workers_per_device=2
        )
        attacker = textattack.Attacker(self.attack, self.dataset, attack_args)

        # Run
        attacker.attack_dataset()

        return log_file_name
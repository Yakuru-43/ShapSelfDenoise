import textattack
import transformers

# Load model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("../save/classification/0/model")
tokenizer = transformers.AutoTokenizer.from_pretrained("../save/classification/0/tokenizer")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR

goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
constraints = [

    RepeatModification(),

    StopwordModification(),

    WordEmbeddingDistance(min_cos_sim=0.9)

]

transformation = WordSwapEmbedding(max_candidates=50)

search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack

attack = textattack.attack_recipes.DeepWordBugGao2018(goal_function, constraints, transformation, search_method)

input_text = "Qu'est-ce que la rupture conventionnelle ?"

label = 4 #Positive

attack_result = attack.attack(input_text, label)

# Print the results in color 

print(attack_result.original_result)
print(attack_result.perturbed_result)
print('\n\n')
# To get the new class label
print(attack_result.perturbed_result._processed_output[0])
# To get the perturbed text
print(attack_result.perturbed_text())
# Get the perturbed score   
print(attack_result.perturbed_result.score)
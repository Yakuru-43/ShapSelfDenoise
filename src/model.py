import transformers
import torch
import numpy as np
from captum.attr import (
    ShapleyValueSampling, 
    LLMAttribution, 
    TextTemplateInput,
)


def mask_words(text, score, p, mask_word):
    # Split the text into words
    """
    Replaces the top p% of words in a given text with the given mask_word, based on the scores provided.
    
    Args:
        text (str): The text to modify.
        score (list): A list of scores associated with each word in the text.
        p (float): The percentage of words to replace, as a float between 0 and 1.
        mask_word (str): The word to replace the top-scoring words with.
    
    Returns:
        str: The modified text with the top-scoring words replaced.
    """
    words = text.split()

    # Calculate the number of words to replace based on the percentage
    num_to_replace = int(len(words) * p)

    # Combine indices and scores into a list of pairs
    index_number_pairs = list(enumerate(score))
    
    # Sort pairs by score in descending order
    index_number_pairs.sort(key=lambda pair: pair[1], reverse=True)
    
    # Get indices of 'num_to_replace' highest scores
    highest_indices = [pair[0] for pair in index_number_pairs[:num_to_replace]]

    # Replace these words in original text
    replaced_text = ' '.join(word if i not in highest_indices else mask_word for i, word in enumerate(words))
    
    return replaced_text


def add_placeholders(prompt,sentence_to_classify,suffix):
    words = sentence_to_classify.split()
    for word in words :
        prompt += "{}"
    prompt += suffix
    return prompt, words

class Alpaca(torch.nn.Module):
    def __init__(self,args, model_path):
        super().__init__()
        self.template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

        self.template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""        
        self.args = args
        self.batch_size = args.batchsize
        self.precision = args.precision
        self.alpaca_model, self.alpaca_tokenizer, self.ds_engine = self.get_model(args.precision, model_path)
        self.instruction = None
        self.verbalizer = None
        self.num_labels = None
        self.preprocess_input = None

        self.roberta_model = None
        self.roberta_tokenizer = None

    def get_model(self,precision, model_path):
        print("Loading alpaca.")
        print("Here is the torch.cuda.is_available() : "+ str(torch.cuda.is_available()))

        
        alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded.")

        # Determine precision type
        if precision == 'full':
            dtype = torch.float32
            print("Loading model with full precision.")
        else:  # 'half'
            dtype = torch.float16
            print("Loading model with half precision.")
        
        alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map = 'cuda:0', torch_dtype=dtype)
        print(f"Model loaded with {precision} precision ({dtype}).")


        alpaca_tokenizer.padding_side = "left" 

        # alpaca_model.cuda()
        alpaca_model.eval()

        
        ds_engine = alpaca_model
        return alpaca_model, alpaca_tokenizer, ds_engine
    
    def as_sst2(self):
        mask_word = self.args.mask_word

        self.denoise_instruction = f"""Replace each mask word {mask_word} in the input sentence with a suitable word. The output sentence should be natural and coherent and should be of the same length as the given sentence.

### Input: 
{mask_word} reassembled from {mask_word} cutting-room {mask_word} of any {mask_word} daytime {mask_word} .

### Response:
apparently reassembled from the cutting-room floor of any given daytime soap .

### Input: 
a {mask_word} , funny and {mask_word} transporting re-imagining {mask_word} {mask_word} and the beast and 1930s {mask_word} films

### Response:
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films"""+"""

### Input:
{}"""
        self.instruction = """Given an English sentence input, determine its sentiment as positive or negative."""
        #         self.instruction = """Given an English sentence input, determine its sentiment as "Positive" or "Negative". You can only output "Positive" or "Negative".

        # ### Input: 
        # apparently reassembled from the cutting-room floor of any given daytime soap .

        # ### Response:
        # Positive

        # ### Input: 
        # a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

        # ### Response:
        # Negative"""

        self.verbalizer = self.sst2_verbalizer
        self.num_labels = 2
        self.preprocess_input = self.general_preprocess_input
        self.label_token = [29940,9135]

    def as_agnews(self, mask_word):
        

        self.denoise_instruction = f"""Replace each masked position {mask_word} in the provided sentence with a suitable word to make it natural and coherent. Only one word should be used to replace each {mask_word}. The returned sentence should be of the same length as the given sentence. Provide the answer directly.
### Input: 
Fannie Mae Pays the {mask_word} of Cutting Corners to Look Safe,"Two {mask_word} agencies have {mask_word} that Fannie Mae cut corners when it came to its {mask_word}, and that has severely damaged its image.

### Response:
Fannie Mae Pays the Price of Cutting Corners to Look Safe,"Two regulatory agencies have concluded that Fannie Mae cut corners when it came to its accounting, and that has severely damaged its image.

### Input: 
{mask_word}  fights AIDS with {mask_word}  power,A bill is currently in Uganda's {mask_word}  that would strengthen {mask_word} rights.

### Response:
Africa fights AIDS with girl power,A bill is currently in Uganda's parliament that would strengthen women's rights."""+"""

### Input:
{}"""
        
        self.instruction = """Given a news article title and description, classify it into one of the four categories: Sports, World, Technology, or Business. Return the category name as the answer.

### Input: 
Title: Venezuelans Vote Early in Referendum on Chavez Rule (Reuters)
Description: Reuters - Venezuelans turned out early and in large numbers on Sunday to vote in a historic referendum that will either remove left-wing President Hugo Chavez from office or give him a new mandate to govern for the next two years.

### Response:
World

### Input:
Title: Phelps, Thorpe Advance in 200 Freestyle (AP)
Description: AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.

### Response:
Sports

### Input:
Title: Wall St. Bears Claw Back Into the Black (Reuters)
Description: Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.

### Response:
Business
        
### Input:
Title: 'Madden,' 'ESPN' Football Score in Different Ways (Reuters)
Description: Reuters - Was absenteeism a little high\on Tuesday among the guys at the office? EA Sports would like to think it was because "Madden NFL 2005" came out that day, and some fans of the football simulation are rabid enough to take a sick day to play it.

### Response:
Technology"""

        # self.verbalizer = self.agnews_verbalizer
        self.num_labels = 4
        self.label_token = [14058, 29903, 16890, 7141] # ie world, S, Bus, Te
        self.label_name  = ["World", "Sport", "Business", "Technologie"]
        self.label_code =  [101, 102, 103, 104]
        # self.label_token = [2787, 12453, 15197, 5636]

    def predict_batch(self, instances, past_predictions=None, past_answer=None):
        answers = np.zeros((len(instances),self.num_labels))
        output_list = []

        if past_predictions is None and past_answer is None:
            text_a_list = []
            text_b_list = []

            for instance in instances:
                text_a_list.append(instance.text_a)
                text_b_list.append(instance.text_b)

            num = 0
            prompt_list = []
            for a,b in zip(text_a_list,text_b_list):
                num+=1
                Input = self.preprocess_input(a,b)

                prompt = self.template.format(self.instruction, Input)
                prompt_list.append(prompt)

                if num%self.batch_size == 0:
                    inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                    # inputs.pop('token_type_ids')
                    outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
                    org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]
                    answers[num-len(org_guess_dist):num] = org_guess_dist.cpu()

                    prompt_list = []
            if len(prompt_list) > 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                inputs.pop('token_type_ids')
                outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
                org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]
                answers[num-len(org_guess_dist):num] = org_guess_dist.cpu()
            output_list = None
        else:
            answers = past_answer
            output_list = past_predictions
            
        return answers, output_list

    def classify_sentence(self, sentence) :
        with torch.no_grad(): 
            prompt = self.template.format(self.instruction, sentence)
            inputs = self.alpaca_tokenizer(prompt, return_tensors="pt",padding=True)
            # inputs.pop('token_type_ids')
            outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
            org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]
            prediction = org_guess_dist.cpu()
            # Find the index of the maximum probability
            _, max_indices = torch.max(prediction, dim=1)

        return self.label_code[max_indices]

    def classify_sentence_one_hot(self, sentence) :
            with torch.no_grad(): 
                prompt = self.template.format(self.instruction, sentence)
                inputs = self.alpaca_tokenizer(prompt, return_tensors="pt",padding=True)
                # inputs.pop('token_type_ids')
                outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
                org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]
                prediction = org_guess_dist.cpu()   
                print(prediction)
                # Find the index of the maximum value in each row (since your tensor has only one row)
                _, max_indices = torch.max(prediction, dim=1)

                # Create a zero tensor of the same shape as the input
                one_hot_tensor = torch.zeros_like(prediction)

                # Set the maximum index to 1 in each row
                one_hot_tensor.scatter_(1, max_indices.unsqueeze(1), 1.0)


            return one_hot_tensor


    def shap_masking(self, data, mask_rate, precalculated_shapley_values):
        
        """
        This function takes a sentence and a mask rate as input, and returns a sentence with a certain percentage of the words replaced by a <mask> token. The words to be replaced are chosen according to their SHAPley value. 
        The SHAPley value is a measure of the importance of a particular word in a sentence for a particular task. 
        The higher the SHAPley value of a word, the more important it is to the task.

        If the precalculated_shapley_values parameter is not None, it is assumed to contain the precalculated SHAPley values for the sentence. 
        In this case, the function will use these values instead of calculating the SHAPley values itself.

        The function returns a tuple, where the first element is the masked sentence and the second element is the SHAPley values used to mask the sentence.
        
        Input:
        data: the sentence to be masked as a string
        mask_rate: the percentage of words to be replaced by a <mask> token as a float between 0 and 1

        Output:
        masked_sentence: the masked sentence as a string
        shapley_values: the SHAPley values used to mask the sentence as a list
        """

        if precalculated_shapley_values is None:
            prompt ="""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven a 
news article title and description, classify it into one of the four categories: Sports, World, Technology, or Business. Return the category name as the answer.\n\n### Input: \nTitle: 
Venezuelans Vote Early in Referendum on Chavez Rule (Reuters)\nDescription: Reuters - Venezuelans turned out early and in large numbers on Sunday to vote in a historic referendum that will 
either remove left-wing President Hugo Chavez from office or give him a new mandate to govern for the next two years.\n\n### Response:\nWorld\n\n### Input:\nTitle: Phelps, Thorpe Advance in 
200 Freestyle (AP)\nDescription: AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for 
the evening\'s 400 freestyle relay final. Phelps\' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.\n\n### 
Response:\nSports\n\n### Input:\nTitle: Wall St. Bears Claw Back Into the Black (Reuters)\nDescription: Reuters - Short-sellers, Wall Street\'s dwindling band of ultra-cynics, are seeing green 
again.\n\n### Response:\nBusiness\n        \n### Input:\nTitle: \'Madden,\' \'ESPN\' Football Score in Different Ways (Reuters)\nDescription: Reuters - Was absenteeism a little high\\on Tuesday
among the guys at the office? EA Sports would like to think it was because "Madden NFL 2005" came out that day, and some fans of the football simulation are rabid enough to take a sick day to 
play it.\n\n### Response:\nTechnology\n\n### Input:\n
"""

            suffix = "\n\n### Response:\n"

            # Do the classification
            model_input = self.alpaca_tokenizer(prompt + data + suffix, return_tensors="pt").to("cuda")
            with torch.no_grad(): 
                output_ids = self.alpaca_model.generate(model_input["input_ids"], max_new_tokens=1)[0]
                response = self.alpaca_tokenizer.decode(output_ids, skip_special_tokens=True)
                category = response.split()[-1]

            # Here the code that does the SHAP shit 
            fa = ShapleyValueSampling(self.alpaca_model)
            llm_attr = LLMAttribution(fa, self.alpaca_tokenizer)

            eval_prompt, values_to_add = add_placeholders(prompt, data, suffix)

            inp = TextTemplateInput(
                template=eval_prompt, 
                values=values_to_add,
            )

            attr_res = llm_attr.attribute(inp,target= category)
        
            #Ge the shapley values 
            shapley_values = attr_res.token_attr.cpu().numpy().tolist()[0]

        # Shapley values have already bean calculated for the data in a previous run
        else : 
            shapley_values = precalculated_shapley_values

        masqued_text = mask_words(data, shapley_values, mask_rate, "<mask>")

        return masqued_text, shapley_values

    def denoise_instances(self,instances):
        
        denoise_instruction = self.denoise_instruction
        text_a_list = []
        text_b_list = []
        output_list_a = []
        output_list_b = []

        for instance in instances:
            text_a_list.append(instance.text_a)
            text_b_list.append(instance.text_b)


        num = 0
        prompt_list = []
        for Input in text_a_list:
            num+=1

            if Input is None:
                output_list_a.append(None)
                continue

            prompt = self.template_without_input.format(denoise_instruction.format(Input))


            prompt_list.append(prompt)
            if num%self.batch_size == 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]

                output_list_a.extend(output)
                prompt_list = []
        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]
            # print(output)
            output_list_a.extend(output)
            prompt_list = []

        for output, instance in zip(output_list_a, instances):
            # print(instance.text_a)
            # print(instance.text_a.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", ''))

            # print("-"*20)
            # print(output,flush=True)
            # print("="*20)
            instance.text_a = output       
        
        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]
            # print(output)
            output_list_b.extend(output)
            prompt_list = []

    def denoise_sentence(self, sentence):
        denoise_instruction = self.denoise_instruction
        prompt = self.template_without_input.format(denoise_instruction.format(sentence))
        # print(prompt)
        inputs = self.alpaca_tokenizer(prompt, return_tensors="pt",padding=True)
                
        # Generate
        generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)
        output = self.alpaca_tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False,)
        return output[len(prompt):]

    def forward(self, input_ids, **kargs):
        """
        Denoise the input text and then generate text based on the denoised input.

        Args:
            input_ids: The input ids to be denoised.
            attention_mask: The attention mask for the input ids.
            token_type_ids: The token type ids for the input ids.
            **kargs: Additional keyword arguments.

        Returns:
            A tuple containing the denoised input and the generated text.
        """
        with torch.no_grad():
            # inputs_text is a list of str
            inputs_text = self.alpaca_tokenizer.batch_decode(input_ids,skip_special_tokens=True, clean_up_tokenization_spaces=False)


            import time
            start_time = time.time()


            # if self.args.maskattack == True :
            #     print(f"we begin the masking/denoising {len(inputs_text)}")
            #     # Mask using Shap
            #     masqued_inputs_text = [self.shap_masking(x, 0.05, None)[0] for x in inputs_text]
            #     print(f"Masqued texts len {len(masqued_inputs_text)}")
            #     # Denoise the sentences
            #     denoised_inputs_text = [self.denoise_sentence(sentence) for sentence in masqued_inputs_text ]
            #     inputs_text = denoised_inputs_text
            #     print(f"we finished the masking/denoising {len(inputs_text)}")
            #     end_time = time.time()
            #     print(f"Elapsed time: {end_time - start_time:.6f} seconds")


            prompts = [self.template.format(self.instruction, Input) for Input in inputs_text]

            inputs = self.alpaca_tokenizer(prompts, return_tensors="pt",padding=True)

            outputs = self.alpaca_model(input_ids=inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
            org_guess_dist = torch.softmax(outputs['logits'][...,-1,:][:,self.label_token],dim=-1)

        return org_guess_dist
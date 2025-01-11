"""
HuggingFace Model Wrapper
--------------------------"""
from textattack.models.wrappers import HuggingFaceModelWrapper

class LLAMAModelWrapper(HuggingFaceModelWrapper):

    def __init__(self, model, tokenizer, alpaca):
        self.model = model
        self.tokenizer = tokenizer
        self.alpaca = alpaca

    def __call__(self, text_input_list):
        """Passes inputs to LLAMA model specifically."""

        # Tokenize the input texts with your desired settings
        inputs_dict = self.tokenizer(
            text_input_list,
            return_tensors="pt",
            padding=True  # You can adjust padding as needed
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        # Define your label tokens as per your LLAMA model's requirement
        label_token = [14058, 29903, 16890, 7141]  # Adjust these tokens accordingly
        output = self.alpaca(**inputs_dict)
        
        return output

        # with torch.no_grad():
        #     # Forward pass through the LLAMA model
        #     outputs = self.model(
        #         inputs_dict.input_ids.to(model_device),
        #         attention_mask=inputs_dict.attention_mask.to(model_device)
        #     )
        #     # Extract logits corresponding to the label tokens
        #     org_logits = outputs['logits'][..., -1, :][:, label_token]

        # return org_logits
    def predict(self, text) : 
        # Here function that takes text as a list od str, and does prediction on it after denoising using SHAP
        pass
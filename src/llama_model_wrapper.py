"""
HuggingFace Model Wrapper
--------------------------"""
from textattack.models.wrappers import HuggingFaceModelWrapper
import torch

class LLAMAModelWrapper(HuggingFaceModelWrapper):
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

        with torch.no_grad():
            # Forward pass through the LLAMA model
            outputs = self.model(
                inputs_dict.input_ids.to(model_device),
                attention_mask=inputs_dict.attention_mask.to(model_device)
            )
            # Extract logits corresponding to the label tokens
            org_logits = outputs['logits'][..., -1, :][:, label_token]

        return org_logits
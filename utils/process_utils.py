import torch
from transformers import BertTokenizer, BertModel

# Initialize tokenizer and BERT model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)

def process_selection(image_tensor, mask_tensor, selection):
    """Applies selection to mask out certain input channels and output labels."""
    input_mask = torch.tensor(selection[:4], dtype=image_tensor.dtype, device=image_tensor.device)
    input_mask = input_mask.view(1, 4, 1, 1, 1) 
    image_tensor = image_tensor * input_mask  

    output_mask = torch.tensor(selection[4:], dtype=mask_tensor.dtype, device=mask_tensor.device)
    output_mask = output_mask.view(1, 3, 1, 1, 1)
    mask_tensor = mask_tensor * output_mask

    return image_tensor, mask_tensor

def generate_text_prompt(selection):
    """Generates a text prompt describing the selected inputs and outputs."""
    input_modalities = ["T1", "T2", "T1ce", "FLAIR"]
    output_labels = ["Whole Tumor (WT)", "Enhancing Tumor (ET)", "Tumor Core (TC)"]

    active_inputs = [mod for mod, flag in zip(input_modalities, selection[:4]) if flag == 1]
    active_outputs = [label for label, flag in zip(output_labels, selection[4:]) if flag == 1]

    prompt = "Based on the provided MRI scan using {} only, please generate the segmentation map for {} only.".format(
        ", ".join(active_inputs) if active_inputs else "no modality",
        ", ".join(active_outputs) if active_outputs else "no target"
    )
    prompt += " Disregard the other modalities and masks."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_text = model_bert(**inputs)
    cls_embedding = output_text.last_hidden_state[:, 0, :]  # Use BERT's [CLS] token embedding
    text_prompt = cls_embedding
    
    return text_prompt
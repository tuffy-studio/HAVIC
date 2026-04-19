import os
import sys
import torch

# Add project root to path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.HAVIC import *
from collections import OrderedDict


input_weight_path = "./pt_model.200.pth" # path to the pretrained model weights
output_weight_path = "./model_to_be_ft.pth" # path to save the initialized finetuning weights (after remapping and filtering)

pt_weight = torch.load(input_weight_path)

ft_model = HAVIC_FT()

ft_weight = OrderedDict()
for k in pt_weight.keys():
    # only keep the weights related to the audio/visual encoder and the audio-visual interaction module for finetuning
    if ('encoder' in k) or ('AudioVisualInteractionModule' in k): 
        ft_weight[k] = pt_weight[k]
    else:
        continue
        
missing, _ = ft_model.load_state_dict(ft_weight, strict=False)

print('=' * 50)
print("Missing keys (weights need to train from scratch in finetuing stage):")
print(missing)
print('=' * 50)

state_dict = ft_model.state_dict()
torch.save(state_dict, output_weight_path)

print(f"Initialized weights saved to: {output_weight_path}")
import os
import sys
import torch

# Add project root to path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.HAVIC import *
from collections import OrderedDict


input_weight_path = "./weights/pt_model.200.pth"

output_weight_path = "./weights/model_to_be_ft.pth"

pt_weight = torch.load(input_weight_path)

ft_model = HAVIC_FT()

ft_weight = OrderedDict()
for k in pt_weight.keys():
    if ('encoder' in k) or ('AudioVisualInteractionModule' in k):
        ft_weight[k] = pt_weight[k]
    else:
        continue
        
missing, unexpected = ft_model.load_state_dict(ft_weight, strict=False)

print('\n' + '=' * 50)
print("Missing keys (weights need to train from scratch in finetuing stage):")
print(missing)

print('\n' + '=' * 50)
print("Unexpected keys:")
print(unexpected)

state_dict = ft_model.state_dict()
torch.save(state_dict, output_weight_path)
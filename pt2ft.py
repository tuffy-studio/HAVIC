import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from src.models.HAVIC import *
from collections import OrderedDict  # 有序字典，用于存储模型权重

# 预训练模型权重路径
input_weight_path = ""

# 修改后的预训练模型权重保存路径
output_weight_path = ""

pt_weight = torch.load(input_weight_path)

# Construct model
ft_model = HAVIC_FT()

ft_model = torch.nn.DataParallel(ft_model)

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
with open("mlp_list.txt", "w") as f:
    f.write(str(missing))

print('\n' + '=' * 50)
print("Unexpected keys:")
print(unexpected)

#state_dict = ft_model.module.state_dict()
state_dict = ft_model.state_dict()

torch.save(state_dict, output_weight_path)





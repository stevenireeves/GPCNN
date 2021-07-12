import torch
from model2 import Net

model = torch.load('models_2/model_epoch_100.pth')

example = torch.rand(1,3, 64, 64)

traced_script_model = torch.jit.trace(model, example)
traced_script_module.save("traced_AEN_model.pt") 


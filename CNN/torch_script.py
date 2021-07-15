import torch
from model import Net

model = torch.load('models2/model_epoch_43.pth')

example = torch.rand(1,3, 128, 128)

traced_script_model = torch.jit.trace(model, example)
traced_script_module.save("traced_AEN_model.pt") 


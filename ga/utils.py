import torch

def get_weights(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def set_weights(model, chromosome):
    pointer = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = chromosome[pointer:pointer + num_params].view(param.data.size())
        pointer += num_params
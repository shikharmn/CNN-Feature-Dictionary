import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

from data_helper import *

patch_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
])


def evaluate(patch, model):
    second_last = model.fc.register_forward_hook(give_layer)
    out = model(patch)
    return second_last


def get_feature(patch, model, device='cpu', indiv=1):
    activations = {}

    def get_activation(name):
        def hook(self, input, output):
            activations[name] = output.detach()

        return hook
    torch.cuda.reset_max_memory_allocated(device)
    if indiv == 1:
        patch_tensor = patch_transforms(patch).unsqueeze(0)
    else:
        patch_tensor = torch.cat([patch_transforms(pat).reshape(-1, 3, 224, 224) for pat in patch])
#    print(patch_tensor.size())
    ans = model.avgpool.register_forward_hook(get_activation('avgpool'))
    out = model(patch_tensor.to(device))
    torch.Tensor.cpu(patch_tensor)
    return torch.Tensor.cpu(activations['avgpool']).numpy().reshape(-1, 2048)

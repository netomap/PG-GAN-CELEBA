import pathlib
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import pathlib
import numpy as np
from PIL import Image

def get_noise(b_size, noise_dim, device=torch.device('cpu')):
    return torch.randn((b_size, noise_dim, 1, 1), device=device)

class Custom_Dataset(Dataset):

    def __init__(self, imgs_dir, pattern, transformer, inv_transformer):
        lista = list(pathlib.Path(imgs_dir).glob(pattern))
        self.lista = [str(l) for l in lista]
        self.transformer = transformer
        self.inv_transformer = inv_transformer
    
    def __len__(self):
        return len(self.lista)
    
    def __getitem__(self, index):
        return self.transformer(Image.open(self.lista[index]))
    
    def criar_grid_imagens_aleatorias(self, n_images=8, tipo='np'):
        dataloader = DataLoader(self, batch_size=n_images, shuffle=True)
        imgs_tensor = next(iter(dataloader))
        grid_np = criar_grid(imgs_tensor, n_images, self.inv_transformer, tipo)
        return grid_np

def criar_grid(imgs_tensor: torch.tensor, nrow, inv_transformer, tipo='np'):
    '''
    Retorna um Image pil ou grid_np já formatado.
    '''
    grid_tensor = make_grid(imgs_tensor, nrow=nrow, padding=0, normalize=True)
    
    if tipo == 'pil':
        img_pil = inv_transformer(grid_tensor)
        return img_pil

    grid_np = np.transpose(grid_tensor.detach().cpu().numpy(), (1, 2, 0))
    return grid_np
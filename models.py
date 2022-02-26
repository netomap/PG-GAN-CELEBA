from operator import mod
import torch
from torch import negative, nn
import utils as u
import sys
from datetime import datetime
import os

class Generator(nn.Module):

    def __init__(self, img_channels, noise_dim, n_camadas, cfl=512):
        '''
        - cfl: Channels first layer
        '''
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.cfl = cfl
        self.img_channels = img_channels
        # nome, c_in, c_out, kernel_s, stride, padding
        self.dicionario = {
            0: ['primeira', noise_dim, cfl, 4, 1, 0], # 4x4
            1: ['segunda', cfl, int(cfl/2), 4, 2, 1], # 8x8
            2: ['terceira', int(cfl/2), int(cfl/4), 4, 2, 1], # 16x16
            3: ['quarta', int(cfl/4), int(cfl/8), 4, 2, 1], # 32x32
            4: ['quinta', int(cfl/8), int(cfl/16), 4, 2, 1], # 64x64
            5: ['sexta', int(cfl/16), int(cfl/32), 4, 2, 1], # 128x128
            6: ['setima', int(cfl/32), int(cfl/64), 4, 2, 1], # 256x256
            7: ['oitava', int(cfl/64), int(cfl/128), 4, 2, 1], # 512x512
            -1: ['ultima', -1, img_channels, 4, 2, 1], #prestar atenção aqui que o valor -1 deve ser substituído de acordo com o tamanho da rede
        }

        modules = []
        for k in range(n_camadas):
            nome, c_in, c_out, kernel_s, stride, padding = self.dicionario[k]  # pegando os nomes e valores de cada camada
            modules.append([nome, self._create_block(in_channels=c_in, out_channels=c_out, kernel_size=kernel_s, stride=stride, padding=padding)])
        
        n_canais_ultima_camada = c_out
        nome, c_in, c_out, kernel_s, stride, padding = self.dicionario[-1]  # última camada
        c_in = n_canais_ultima_camada
        modules.append([nome, nn.Sequential(
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_s, stride=stride, padding=padding),
            nn.Tanh()
            )
        ])
        
        self.net = nn.Sequential()
        for nome, modulo in modules:
            self.net.add_module(nome, modulo)

    def _create_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, 
                                out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
    
    def save(self):
        name = self.__class__.__name__
        if not os.path.exists('./models/'): os.makedirs('./models/')

        nomes_camadas = list(self.net._modules)
        n_camadas = len(nomes_camadas) - 1 # porque não considera a última camada

        model = dict()
        for k, nome in enumerate(nomes_camadas): 
            model[nome] = self.net[k].state_dict()
        
        checkpoint = {
            'model': model,
            'noise_dim': self.noise_dim,
            'cfl': self.cfl,
            'img_channels': self.img_channels,
            'tempo': datetime.strftime(datetime.now(), '%d/%m/%Y, %H:%M')
        }

        torch.save(checkpoint, f'./models/{name}_{n_camadas}_layer.pt')
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = checkpoint['model']
        camadas_checkpoint = model.keys()

        noise_dim, cfl= checkpoint['noise_dim'], checkpoint['cfl']
        img_channels, tempo = checkpoint['img_channels'], checkpoint['tempo']
        print (f'{noise_dim=}, {cfl=}, {img_channels=}, {tempo=}')
        
        modulos_desta_rede = list(self.net._modules)
        for k, camada in enumerate(modulos_desta_rede):
            if (camada != 'ultima'):# não se carrega a última camada pois o channels_in varia a cada vez que a rede aumenta.
                print (camada, self.net[k].load_state_dict(model[camada]))
        
        # a não ser que o número de camadas do checkpoint seja o mesmo do modelo instanciado
        if (len(model.keys()) == len(modulos_desta_rede)):
            print (camada, self.net[-1].load_state_dict(model['ultima']))


class Discriminator(nn.Module):

    def __init__(self, img_channels, n_camadas, cfl=128):
        '''
        - cfl: Channels First layer.
        '''
        super(Discriminator, self).__init__()
        self.cfl = cfl
        self.img_channels = img_channels
        # nome, c_in, c_out, kernel_s, stride, padding, batch_normalization
        self.dicionario = {
            0: ['primeira', img_channels, cfl, 4, 2, 1, False], # 4x4
            1: ['segunda', cfl, int(cfl*2), 4, 2, 1, True], # 8x8
            2: ['terceira', int(cfl*2), int(cfl*4), 4, 2, 1, True], # 16x16
            3: ['quarta', int(cfl*4), int(cfl*8), 4, 2, 1, True], # 32x32
            4: ['quinta', int(cfl*8), int(cfl*16), 4, 2, 1, True], # 64x64
            5: ['sexta', int(cfl*16), int(cfl*32), 4, 2, 1, True], # 128x128
            6: ['setima', int(cfl*32), int(cfl*64), 4, 2, 1, True], # 256x256
            7: ['oitava', int(cfl*64), int(cfl*128), 4, 2, 1, True], # 512x512
            -1: ['ultima', -1, 1, 4, 1, 0, False], #prestar atenção aqui que o valor -1 deve ser substituído de acordo com o tamanho da rede
        }

        modules = []
        for k in range(n_camadas):
            nome, c_in, c_out, kernel_s, stride, padding, batch_norm = self.dicionario[k]  # camadas
            modules.append([nome, self._create_block(c_in, c_out, kernel_s, stride, padding, batch_norm)])
        
        n_channels_last_layer = c_out
        nome, c_in, c_out, kernel_s, stride, padding, batch_norm = self.dicionario[-1]  # última camada
        c_in = n_channels_last_layer
        modules.append(['ultima', nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_s, stride, padding),
            nn.Sigmoid()
        )])

        self.net = nn.Sequential()
        for nome, modulo in modules:
            self.net.add_module(nome, modulo)
    
    def _create_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_normalization=True):
        sub_modulo = []
        sub_modulo.append(nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding, bias=False))
        if (batch_normalization): sub_modulo.append(nn.BatchNorm2d(num_features=out_channels))
        sub_modulo.append(nn.LeakyReLU(negative_slope=0.2))
        return nn.Sequential(*sub_modulo)
    
    def forward(self, x):
        return self.net(x)
    
    def save(self):
        name = self.__class__.__name__
        if not os.path.exists('./models/'): os.makedirs('./models/')

        nomes_camadas = list(self.net._modules)
        n_camadas = len(nomes_camadas) - 1 # porque não considera a última camada

        model = dict()
        for k, nome in enumerate(nomes_camadas): 
            model[nome] = self.net[k].state_dict()
        
        checkpoint = {
            'model': model,
            'cfl': self.cfl,
            'img_channels': self.img_channels,
            'tempo': datetime.strftime(datetime.now(), '%d/%m/%Y, %H:%M')
        }

        torch.save(checkpoint, f'./models/{name}_{n_camadas}_layer.pt')
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = checkpoint['model']
        camadas_checkpoint = model.keys()

        cfl= checkpoint['cfl']
        img_channels, tempo = checkpoint['img_channels'], checkpoint['tempo']
        print (f'{cfl=}, {img_channels=}, {tempo=}')
        
        modulos_desta_rede = list(self.net._modules)
        for k, camada in enumerate(modulos_desta_rede):
            if (camada != 'ultima'):# não se carrega a última camada pois o channels_in varia a cada vez que a rede aumenta.
                print (camada, self.net[k].load_state_dict(model[camada]))
        
        # a não ser que o número de camadas do checkpoint seja o mesmo do modelo instanciado
        if (len(model.keys()) == len(modulos_desta_rede)):
            print (camada, self.net[-1].load_state_dict(model['ultima']))

if __name__ == '__main__':

    NOISE_DIM = 100

    generator = Generator(img_channels=3, noise_dim=100, n_camadas=4, cfl=512)
    print (generator)

    noise = u.get_noise(5, 100)
    output_generator = generator(noise)
    print (f'{output_generator.shape=}')

    generator.save()

    g2 = Generator(img_channels=3, noise_dim=100, n_camadas=4, cfl=512)
    g2.load_checkpoint('./models/Generator_4_layer.pt')
    print (f'fim')

    discriminator = Discriminator(img_channels=3, n_camadas=1, cfl=64)
    print (discriminator)

    output_discriminator = discriminator(output_generator)
    print (f'{output_discriminator.shape=}')

    discriminator.save()

    d2 = Discriminator(img_channels=3, n_camadas=4, cfl=64)
    d2.load_checkpoint('./models/Discriminator_4_layer.pt')
    print (f'fim')
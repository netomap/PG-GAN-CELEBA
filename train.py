import pathlib
from pyexpat import model
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import utils as u
from models import Generator, Discriminator
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import argparse, os, re, time

parser = argparse.ArgumentParser()

parser.add_argument('--ncamadas', type=int, required=True, help='Numero de camadas das redes neurais, sem considerar a ultima.')
parser.add_argument('--noise_dim', type=int, help='Numero de dimensoes do vetor noise.', default=100)
parser.add_argument('--img_channels', type=int, help='Numero de canais das imagens.', default=3)
parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-4)
parser.add_argument('--batch_size', type=int, help='Batch size', default=128)


args = parser.parse_args()

NOISE_DIM = args.noise_dim
N_CAMADAS = args.ncamadas  # ASSIM, O TAMANHO DA IMAGEM SERÁ N_CAMADAS*8. [1->8x8, 2->16x16, 3->32x32, 4->64x64, 5->128x128, 6->256x256]
IMG_SIZE = 4 * (2**N_CAMADAS)
IMG_CHANNELS = args.img_channels
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
MODELS_DIR = './models'
IMGS_DIR = './imgs/celeba'
TAXA_TREINAMENTO_DISCRIMINATOR = 5  # ou seja, o discriminator treina 5 vezes mais que o generator
LAMBDA_GP = 10 # TAXA DO GRADIENT PENALTY

def criar_transformer(img_size_):
    return  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size_, img_size_)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ])

inv_transformer = transforms.Compose([
    transforms.Normalize(mean=(-1., -1., -1.), std=(1., 1., 1.)),
    transforms.ToPILImage(),
])

writer = SummaryWriter(f'./logs/pg-wgan-celeba-{N_CAMADAS}-camadas')
writer.add_text('mensagem inicial', f"""Treinamento PG-GAN com {N_CAMADAS} camadas iniciais + a camada final.
Assim, o tamanho da imagem será de {IMG_SIZE}x{IMG_SIZE} pixels.""")

print ('Criando os modelos para testar suas saídas: ')
discriminator = Discriminator(img_channels=IMG_CHANNELS, n_camadas=N_CAMADAS, cfl=128)
generator = Generator(img_channels=IMG_CHANNELS, noise_dim=NOISE_DIM, n_camadas=N_CAMADAS, cfl=512)

# Testando os modelos
noise = u.get_noise(3, NOISE_DIM)
output_generator = generator(noise)
output_discriminator = discriminator(output_generator)
print (f'{noise.shape=}')
print (f'{output_generator.shape=}')
print (f'{output_discriminator.shape=}')

print ('ok!')

def gradient_penalty(model_d, real_imgs, fake_imgs, device_):

    b_size, c, h, w = real_imgs.shape
    alpha = torch.rand((b_size, 1, 1, 1)).repeat(1, c, h, w).to(device_)
    # tenta criar um tensor (b,c,h,w) onde cada elemento b (de dim c,h,w) possui um número aleatório em todas as posições

    interpolated_imgs = real_imgs * alpha + fake_imgs * (1-alpha)

    # Cálculo score
    mixed_score = model_d(interpolated_imgs)

    gradient = torch.autograd.grad(
        inputs = interpolated_imgs,
        outputs = mixed_score,
        grad_outputs = torch.ones_like(mixed_score),
        create_graph = True,
        retain_graph = True
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm-1) ** 2)
    
    return penalty

def imprimir_imagem_checkpoint():
    random_noise = u.get_noise(8, NOISE_DIM, device)
    random_img = generator(random_noise)
    img_np = u.criar_grid(random_img, nrow=8, inv_transformer=inv_transformer, tipo='np')
    plt.figure(figsize=(20, 4))
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()

def pegar_ultimo_treinamento():
    models_list = list(pathlib.Path('./models/').glob('*.pt'))
    layers = []
    for model_path in models_list:
        layers.append(int(re.findall(r'[0-9]{1,}', str(model_path))[0]))
    
    return max(layers) # retorna o checkpoint de maior número de camadas

N_CAMADAS_MAX = pegar_ultimo_treinamento() # maior número de camadas.

print (f'Criando os modelos para o treinamento com {N_CAMADAS} camadas.')
discriminator = Discriminator(img_channels=IMG_CHANNELS, n_camadas=N_CAMADAS, cfl=128)
generator = Generator(img_channels=IMG_CHANNELS, noise_dim=NOISE_DIM, n_camadas=N_CAMADAS, cfl=512)

print ('Importando modelo discriminator...')
discriminator.load_checkpoint(f'./models/Discriminator_{N_CAMADAS_MAX}_layer.pt')
print ('==============================================================\n\n')

print ('Importando modelo generator...')
generator.load_checkpoint(f'./models/Generator_{N_CAMADAS_MAX}_layer.pt')
print ('==============================================================\n\n')

noise = u.get_noise(1, NOISE_DIM)
output_generator = generator(noise)
IMG_SIZE = output_generator.shape[-1]

print (f'Treinamento com {N_CAMADAS=} e tamanho das imagens {IMG_SIZE=}')
transformer = criar_transformer(img_size_=IMG_SIZE)
print (f'Transformers criados.')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (f'{device=}')

discriminator.to(device)
generator.to(device)

optim_generator = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0., 0.9))
optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0., 0.9))

dataset = u.Custom_Dataset(IMGS_DIR, pattern='*.jpg', transformer=transformer, inv_transformer=inv_transformer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print (f'{len(dataset)=}, {BATCH_SIZE=}, {len(dataloader)=}')

# ESTA PARTE SERVE APENAS PARA PEGAR O last_epoch
checkpoint = torch.load(f'./models/Generator_{N_CAMADAS_MAX}_layer.pt', map_location=torch.device('cpu'))
last_epoch = checkpoint['epochs']

if (N_CAMADAS > N_CAMADAS_MAX):
    print (f'Treinamento novo com {N_CAMADAS} camadas, pois o último treinamento tem somente {N_CAMADAS_MAX} camadas.')
    last_epoch = 0
else:
    print (f'Continuação de treinamento com {N_CAMADAS}, pois já existem checkpoints de {N_CAMADAS_MAX} camadas treinados.')

print (f'O treinamento vai começar com época número: {last_epoch}')
print (f'Preparação das variáveis, modelos e otimizadores ok')

generator.train()
discriminator.train()

print ('Iniciando treinamento.')
time.sleep(10) # apenas para ler os prints...

for epoch in range(last_epoch + 1, last_epoch + 11, 1):

    os.system('clear')
    print (f'{epoch=}')

    for real_imgs in tqdm(dataloader):
        real_imgs = real_imgs.to(device)
        b_size = len(real_imgs)

        # Treinando o discriminator
        for _ in range(TAXA_TREINAMENTO_DISCRIMINATOR):
            noise = u.get_noise(b_size, NOISE_DIM, device)
            fake_imgs = generator(noise)

            output_real = discriminator(real_imgs)
            output_fake = discriminator(fake_imgs)

            # Cálculo do Gradient-penalty
            gp = gradient_penalty(discriminator, real_imgs, fake_imgs, device)
            loss_discriminator = -(torch.mean(output_real.view(-1)) - torch.mean(output_fake.view(-1))) + LAMBDA_GP*gp
            discriminator.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            optim_discriminator.step()
        
        # Treinando o generator
        output_fake_for_generator = discriminator(fake_imgs)
        loss_generator = -torch.mean(output_fake_for_generator.view(-1))
        generator.zero_grad()
        loss_generator.backward()
        optim_generator.step()

    # Levando as variáveis para o tensorboard
    writer.add_scalar('loss_discriminator', loss_discriminator.item(), epoch)
    writer.add_scalar('loss_generator', loss_generator.item(), epoch)
    
    discriminator.save(epoch)
    generator.save(epoch)

writer.close()

print (f'Fim do treinamento!')
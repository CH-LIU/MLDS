from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch_dataloader import *
from tqdm import tqdm
from model import Generator, Discriminator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Set random seem for reproducibility
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
LAMBDA = 10

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data , BATCH_SIZE):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, 64, 64)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(BATCH_SIZE, 3, 64, 64)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
def train(batchsize, epochs):
    dataset = dset.ImageFolder(root="./data/",
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=1)
    

    nz = 150
    netG = Generator(nz, (64,64,3))
    netG = netG.cuda()
    netG.apply(weights_init)
    netD = Discriminator((64,64,3))
    netD = netD.cuda()
    netD.apply(weights_init)

    optimizerD = optim.RMSprop(netD.parameters(), lr=0.00005, alpha=0.9)
    optimizerG = optim.RMSprop(netG.parameters(), lr=0.00005, alpha=0.9)

    img_list = []
    G_losses = []
    D_losses = []

    netG.train()
    netD.train()
    for epoch in range(epochs):
        d_loss = 0
        g_loss = 0
        count = 0
        fixed_noise = torch.randn(25, nz, 1, 1, device="cuda")
        print("Epoch: "+str(epoch)+"/"+str(epochs))
        is_d = 0
        for data in tqdm(dataloader):
            optimizerD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            #discriminate real image
            D_real = netD(real_cpu).view(-1)
            D_real_loss = torch.mean(D_real)
            #generate fake image from noise vector
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise).detach()
            #discriminate fake image
            D_fake = netD(fake).view(-1)
            D_fake_loss = torch.mean(D_fake)

            gradient_penalty = calc_gradient_penalty(netD, real_cpu, fake , b_size)
            # discriminator loss
            D_loss =  D_fake_loss - D_real_loss + gradient_penalty
            D_loss.backward()
            # Update D
            optimizerD.step()
            d_loss += D_loss.item()
            D_losses.append(D_loss.item())
            is_d+=1
                # weight clipping
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            # update generator every 5 batch
            if is_d%5 == 0:
                is_d = 1
                # freeze discriminator
                for p in netD.parameters():
                    p.requires_grad = False
                optimizerG.zero_grad()
                #generate fake image
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                #to confuse discriminator
                G_fake = netD(fake).view(-1)
                #generator loss
                G_loss = -torch.mean(G_fake)
                G_loss.backward()
            # Update G
                optimizerG.step()
                g_loss += G_loss.item()
                G_losses.append(G_loss.item())
                for p in netD.parameters():
                    p.requires_grad = True
        print("D_real_loss:%.6f, D_fake_loss:%.6f"%(D_real_loss,D_fake_loss))

        # output image every 3 epoch
        if epoch%3 == 0:
            with torch.no_grad():
                test_img = netG(fixed_noise).detach().cpu()
            test_img = test_img.numpy()
            test_img = np.transpose(test_img,(0,2,3,1))
            fig, axs = plt.subplots(5, 5)
            cnt = 0
            for i in range(5):
                for j in range(5):
                    axs[i,j].imshow(test_img[cnt, :,:,:])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("./output_grad/"+str(epoch)+".png")
            plt.close()
        print("d loss: "+str(d_loss)+", g loss: "+str(g_loss))
    torch.save({'g': netG.state_dict(), 'd': netD.state_dict()},"model_best")
if __name__ == '__main__':
    train(64, 200)

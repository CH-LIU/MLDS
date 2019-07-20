from __future__ import print_function
import numpy as np
from tag_preprocess import make_tag
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
from torch_dataloader import *
from tqdm import tqdm
from model import Generator, Discriminator
import matplotlib.pyplot as plt
from load_data import *
# Set random seem for reproducibility
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
nz = 100
LAMBDA = 10

train_label, hair_dic, eye_dic = make_tag("./tags.csv")

tag_size = len(hair_dic) + len(eye_dic)

test_tags  = torch.Tensor(np.load("./test_tags.npy")).float().to(device)

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

    disc_interpolates, tag = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def train(batchsize, epochs):
    imageset = dset.ImageFolder(root="../extra_data/",
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    dataset = CustomDataset(imageset, train_label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=1)

    netG = Generator(nz, tag_size)     #tag_size = 22
    netG = netG.to(device)
    netG.apply(weights_init)
    netD = Discriminator(22)
    netD = netD.to(device)
    netD.apply(weights_init)

    criterion_d = nn.BCELoss()
    criterion_a = nn.CrossEntropyLoss()

    train_tags = torch.Tensor(train_label).long().to(device)
    real_label = 1
    fake_label = 0

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
        fixed_noise = torch.randn(25, nz, device=device)
        print("Epoch: "+str(epoch)+"/"+str(epochs))
        for data in tqdm(dataloader):
            optimizerD.zero_grad()
            for p in netD.parameters():
                    p.requires_grad = True
            real_cpu = data[0][0].to(device)
            tag = data[1].float().to(device)

            b_size = real_cpu.size(0)

            output_real_d, output_real_tag = netD(real_cpu)
            D_tag_loss = F.binary_cross_entropy(output_real_tag, tag)
            D_real_loss = -torch.mean(output_real_d)
            errD_real = D_real_loss + D_tag_loss
            errD_real.backward()
            D_x = output_real_d.mean().item()
            ######### generate generator input noise and tag ######################
            noise = torch.randn(b_size, nz, device=device)
            hair_label = np.random.randint(0, 12, b_size)
            eye_label = np.random.randint(12, 22, b_size)
            aux_label = np.zeros((b_size,22))
            for i in range(b_size):
                aux_label[i][hair_label[i]] = 1
                aux_label[i][eye_label[i]] = 1
            aux_label = torch.Tensor(aux_label).float().to(device)
            fake = netG(noise, aux_label)
           
            output_fake_d, output_fake_tag = netD(fake.detach())
            D_faketag_loss = F.binary_cross_entropy(output_fake_tag, aux_label)
            D_fake_loss = torch.mean(output_fake_d)
            
            errD_fake = D_fake_loss + D_faketag_loss
            errD_fake.backward()
            gradient_penalty = calc_gradient_penalty(netD, real_cpu, fake , b_size)
            gradient_penalty.backward()
            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()
            d_loss += errD.item()
            D_losses.append(errD.item())
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            if (count%5) == 0:
                count = 1
                for p in netD.parameters():
                    p.requires_grad = False
                optimizerG.zero_grad()
                fake = netG(noise, aux_label)
                output_g, output_tag_g = netD(fake)
                G_tag_loss = F.binary_cross_entropy(output_tag_g, aux_label)
                G_real_loss = -torch.mean(output_g)

                errG = G_real_loss + G_tag_loss
                errG.backward()
                optimizerG.step()

                g_loss += errG.item()
                G_losses.append(errG.item())
            
            count += 1
        if epoch%3 == 0:
            with torch.no_grad():
                test_img = netG(fixed_noise, test_tags).detach().cpu()
            test_img = test_img.numpy()                
            test_img = np.transpose(test_img,(0,2,3,1))
            test_img *= 0.5
            test_img += 0.5
            fig, axs = plt.subplots(5, 5)
            cnt = 0
            for i in range(5):
                for j in range(5):
                    axs[i,j].imshow(test_img[cnt, :,:,:])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("./output/"+str(epoch)+".png")
            plt.close()
        print("d loss: %.6f , d_tag loss: %.6f, g loss: %.6f , g_tag loss: %.6f" %(D_real_loss.item(), D_tag_loss.item(), G_real_loss.item(), G_tag_loss.item()))
    torch.save({'g': netG.state_dict(), 'd': netD.state_dict()},"model_best")
if __name__ == '__main__':
    train(64, 300)


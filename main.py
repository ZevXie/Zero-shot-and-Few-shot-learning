# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import opt
from model import CADA_VAE
from tqdm.auto import tqdm
from dataloader import dataloader, classifier_dataloader
from sklearn.preprocessing import MinMaxScaler

device = opt.device
torch.manual_seed = opt.seed
batch_size = opt.batch_size

def Cross_Alignment_Loss(x, sig, recon_x, recon_sig, xDecoder_sig, sigDecoder_x):
  ca_loss = l1_loss(xDecoder_sig, x) + l1_loss(sigDecoder_x, sig)
  return ca_loss

def Distributed_Alignment_Loss(mu_x, mu_sig, logvar_x, logvar_sig):
  sigma_sig = logvar_sig.exp()
  sigma_x = logvar_x.exp()
  mu_loss_W21 = l2_norm(mu_sig,mu_x) + torch.norm((sigma_sig-sigma_x))
  mu_loss_W12 = l2_norm(mu_x,mu_sig) + torch.norm((sigma_x-sigma_sig))
  da_loss = torch.sqrt(mu_loss_W12 + mu_loss_W21)
  return da_loss


def VAE_Loss(beta, recon_x, x, recon_sig, sig, logvar_x, logvar_sig, mu_x, mu_sig):
  reconstruction_1 = l1_loss(recon_x, x)
  reconstruction_2 = l1_loss(recon_sig, sig)
  KLD_1 = 0.5 * torch.sum(1 + logvar_x - mu_x.pow(2) - logvar_x.exp())
  KLD_2 = 0.5 * torch.sum(1 + logvar_sig - mu_sig.pow(2) - logvar_sig.exp())
  vae_loss = reconstruction_1 + reconstruction_2 - beta*(KLD_1+KLD_2)
  return vae_loss


def loss_function(x, sig, recon_x, recon_sig, xDecoder_sig, sigDecoder_x, mu_x, mu_sig, logvar_x, logvar_sig, gamma, beta, delta):
  ca_loss = Cross_Alignment_Loss(x, sig, recon_x, recon_sig, xDecoder_sig, sigDecoder_x)
  da_loss = Distributed_Alignment_Loss(mu_x, mu_sig, logvar_x, logvar_sig)
  vae_loss = VAE_Loss(beta, recon_x, x, recon_sig, sig, logvar_x, logvar_sig, mu_x, mu_sig)
  cada_vae_loss = vae_loss + gamma*ca_loss + delta*da_loss
  return cada_vae_loss


def train(epoch):
    global beta
    global delta
    global gamma

    trainbar = tqdm(trainloader)
    model.train()
    train_loss = 0
    # print(beta, delta, gamma)
    for batch_idx, (x, y, sig) in enumerate(trainbar):
        x = x.float()
        sig = sig.float()

        recon_x, recon_sig, mu_x, mu_sig, sigDecoder_x, xDecoder_sig, logvar_x, logvar_sig = model(x, sig)

        # loss
        loss = loss_function(x, sig, recon_x, recon_sig, xDecoder_sig, sigDecoder_x, mu_x, mu_sig, logvar_x, logvar_sig,
                             gamma, beta, delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        trainbar.set_description('l:%.3f' % (train_loss / (batch_idx + 1)))

    # print(train_loss/(batch_idx+1))
    if epoch > 5 and epoch < 21:
        delta += 0.54
    if epoch > 20 and epoch < 75:
        gamma += 0.044
    if epoch < 91:
        beta += 0.0026

if __name__ == '__main__':
    # 数据
    train_set = dataloader(opt, transform=MinMaxScaler(), split='train')  # train_loc
    test_set = dataloader(opt, transform=MinMaxScaler(), split='test')  #  test_unseen_loc
    val_set = dataloader(opt, transform=MinMaxScaler(), split='val')  # val_loc
    trainval_set = dataloader(opt, transform=MinMaxScaler(), split='trainval')  # trainval_loc
    trainloader = data.DataLoader(trainval_set, batch_size=batch_size, shuffle=True)  # batch_size:50
    testloader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 模型、优化器、损失函数
    input_dim = train_set.__getlen__()
    atts_dim = train_set.__get_attlen__()
    model = CADA_VAE(input_dim,atts_dim,opt.z_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    l2_norm = nn.MSELoss(reduction='sum')
    l1_loss = nn.L1Loss(reduction='sum')
    Kld_loss = nn.KLDivLoss(reduction='sum')

    # 其他一些参数
    gamma = 0
    beta = 0
    delta = 0

    for epoch in range(1, opt.epochs):
        print("epoch:", epoch)
        train(epoch)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:24:54 2017

@author: ptrblck

This code is based on the blog post by John Glover:
http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

The pytorch code was partly taken from:
https://github.com/pytorch/examples/blob/master/dcgan/main.py
"""

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
#%matplotlib


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        m.bias.data.fill_(0.0)


def update_learning_rate(optimizer, epoch, init_lr, decay_rate, lr_decay_epochs):
    lr = init_lr * (decay_rate**(epoch // lr_decay_epochs))
    
    if epoch % lr_decay_epochs == 0:
        print('LR set to {}'.format(lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


def samples(
    models,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    
    Taken from https://github.com/AYLIEN/gan-intro/blob/master/gan.py
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        sample = torch.FloatTensor(np.reshape(xs[batch_size * i : batch_size * (i + 1)], (batch_size, 1)))
        sample = Variable(sample).cuda()
        db[batch_size * i:batch_size * (i + 1)] = models[0](sample).cpu().data.numpy()

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        sample = torch.FloatTensor(np.reshape(zs[batch_size * i:batch_size * (i + 1)], (batch_size, 1)))
        sample = Variable(sample).cuda()
        g[batch_size * i:batch_size * (i + 1)] = models[1](sample).cpu().data.numpy()

    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range, ax, save_img_name):
    ax.clear()
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    #f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig(save_img_name)
    #plt.show()
    #plt.pause(0.05)


class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return np.reshape(samples, (-1, 1))


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        samples = np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
        return samples


class Generator(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Generator, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class Discriminator(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Discriminator, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H * 2)
        self.linear2 = torch.nn.Linear(H * 2, H * 2)
        self.linear3 = torch.nn.Linear(H * 2, H * 2)
        self.linear4 = torch.nn.Linear(H * 2, D_out)

    def forward(self, x):
        h0 = F.tanh(self.linear1(x))
        h1 = F.tanh(self.linear2(h0))
        h2 = F.tanh(self.linear3(h1))
        out = F.sigmoid(self.linear4(h2))
        return out

def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return (line_db, line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_db, line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])




N = 8  # batch size
D_in = 1  # input size of D
H = 4  # numbr of  hidden neurons
D_out = 1
learning_rate = 0.005
epochs = 10000
plot_every_epochs = 1000
current_time = datetime.datetime.now().strftime("20%y%m%d_%H%M_%S")
output_path = '/tmp/{}'.format(current_time)
if not os.path.exists(output_path):
    os.makedirs(output_path)
anim_path = output_path


anim_frames = []

use_cuda = torch.cuda.is_available()

data_dist = DataDistribution()
gen_dist = GeneratorDistribution(range=8)

x = torch.FloatTensor(N, 1)
z = torch.FloatTensor(N, 1)
label = torch.FloatTensor(N)
real_label = 1
fake_label = 0

netD = Discriminator(D_in=D_in, H=H, D_out=D_out)
netG = Generator(D_in=D_in, H=H, D_out=D_out)
netD.apply(weights_init)
netG.apply(weights_init)
        
criterion = torch.nn.BCELoss()

if use_cuda:
    x, z, label = x.cuda(), z.cuda(), label.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()

optimizerD = torch.optim.SGD(netD.parameters(), lr=learning_rate)
optimizerG = torch.optim.SGD(netG.parameters(), lr=learning_rate)

# Create figure
f, ax = plt.subplots(1)

for epoch in range(epochs):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real: maximize log(D(x))
    netD.zero_grad()
    real_cpu = torch.FloatTensor(data_dist.sample(N))
    if use_cuda:
        real_cpu = real_cpu.cuda()
    x.copy_(real_cpu)
    label.fill_(real_label)
    xv = Variable(x)
    labelv = Variable(label)
    
    output = netD(xv)  # D(x)
    errD_real = criterion(output, labelv)
    errD_real.backward()
    D_x = output.data.mean()
    
    # train with fake: maximize log(1 - D(G(z)))
    z = torch.FloatTensor(gen_dist.sample(N))[...,None]  # (N_sample, N_channel)
    if use_cuda:
        z = z.cuda()
    zv = Variable(z)
    fake = netG(zv)   # G(z)
    labelv = Variable(label.fill_(fake_label))
    output = netD(fake.detach())   # D(G(z))
    errD_fake = criterion(output, labelv)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake
    optimizerD.step()
    optimizerD = update_learning_rate(optimizer=optimizerD,
                                      epoch=epoch,
                                      init_lr=learning_rate,
                                      decay_rate=0.95,
                                      lr_decay_epochs=150)

    ############################
    # (2) Update G network: maximize log(D(G(z))): guide D make wrong prediction: G(z) --> real_label(1)
    ###########################
    netG.zero_grad()
    labelv = Variable(label.fill_(real_label))
    output = netD(fake)  # D(G(z))
    errG = criterion(output, labelv)
    errG.backward()
    D_G_z2 = output.data.mean()
    optimizerG.step()
    optimizerG = update_learning_rate(optimizer=optimizerG,
                                      epoch=epoch,
                                      init_lr=learning_rate,
                                      decay_rate=0.95,
                                      lr_decay_epochs=150)
    
    print('[%d/%d] Loss_D: %.4f Loss_G %.4f D(x): %.4f D(G(z)): %.4f / %.4f' \
        % (epoch, epochs, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    if epoch % plot_every_epochs == 0:
        # Plot distribution
        samps = samples([netD, netG], data_dist, gen_dist.range, N)
        anim_frames.append(samps)
        plot_distributions(samps, gen_dist.range, ax, save_img_name = output_path+'/{:06}'.format(epoch))

# save_animation(anim_frames, anim_path, gen_dist.range)

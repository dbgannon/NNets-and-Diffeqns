# This is the GenerativeAdesarialNetwork GAN for solving 
# the non-linear stochastic PDEs.
# actually this example is not full stochastic because it uses the solution
# of a non-linear PDE so we can check output.

import numpy as np
import scipy
from scipy import spatial
import matplotlib.pyplot as plt
import torch
import os
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import grad


torch.manual_seed(1)

# defining generator class

class generator(nn.Module):
    
    def __init__(self):
        
        super(generator, self).__init__()
        self.l1 = nn.Linear(1,300)
        self.l2 = nn.Linear(300,1000)
        self.l3 = nn.Linear(1000,800)
        self.l4 = nn.Linear(800,2)
        self.rl = nn.Tanh()
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #       nn.init.normal_(m.weight, mean=0, std=0.7)
       
        
    def forward(self, x):
        z = self.rl(self.l1(x))
        u = self.rl(self.l2(z))
        u = self.rl(self.l3(u))
        z = self.l4(u)
        return z

class discriminator(nn.Module):
    
    def __init__(self):
        
        super(discriminator, self).__init__()
        self.l1 = nn.Linear(3,300)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(300,300)
        self.l3 = nn.Linear(300,200)
        self.l4  = nn.Linear(200,1)
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        #z = torch.cat((x, y),1)
        u = self.relu(self.l1(z))
        u = self.relu(self.l2(u))
        u = self.relu(self.l3(u))
        out = self.sig(self.l4(u))
 
        return out

def u_true2(x):
    u = np.sin(2*x)**2
    return u/8.0

def u_true(x):
    u = np.sin(x)**2
    return u

def f_true2(x):
    f = np.sin(4.0*x)/4.0+x*np.cos(4*x)
    return f

def f_true(x):
    f = np.sin(2.0*x)+2.0*x*np.cos(2*x)
    #f = 4.0*np.cos(2.0*x)
    return f

def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]

def Du(x,u):
    #print(flat(u))
    u_x = grad(flat(u), x, create_graph=True, allow_unused=True)[0] #nth_derivative(flat(u), wrt=x, n=1)
    xv =x[:,0].reshape(batch_size,1)
    z = u_x*xv
    u_xx = grad(flat(z), x, create_graph=True, allow_unused=True)[0] #nth_derivative(flat(u), wrt=x, n=1)
    f = u_xx
    #ifb = torch.FloatTensor( f_true(xv.detach().numpy())).reshape(batch_size,1)
    #ifb.requires_grad = True
    return f

batch_size = 20
num_batches = 20
LAMBDA = 0.1


vxn  = 2000
vx =np.linspace(0, np.pi, vxn) 
ix = torch.FloatTensor(vx).reshape(vxn,1)

import random
btches = []
ubtches = []
fakebtches = []
real_data_batches = []
for i in range(num_batches):
    b = random.choices(vx,k=batch_size)
    bar = np.array(b)
    ub = u_true(bar)
    fb = f_true(bar)
    ub0 = torch.FloatTensor(ub).reshape(batch_size,1)
    ub0.requires_grad=True
    ib = torch.FloatTensor(b).reshape(batch_size,1)
    ifb = torch.FloatTensor(fb).reshape(batch_size,1)
    ib.requires_grad=True
    ifb.requires_grad = True
    real_data_batches.append(torch.cat((ib, ub0, ifb),1))
    btches.append(ib)
    ubtches.append(torch.cat((ib, ub0),1))
    noise = torch.randn(batch_size*1,1)
    fake = torch.FloatTensor(noise).reshape(batch_size,1)
    fake.requires_grad=True
    fakebtches.append(fake)
print(fakebtches[0].shape)

# the followning adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py 
# by Marvin Cao
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


criterion = nn.BCELoss()

criteriond2 = nn.BCELoss()
#optimizerG = optim.SGD(gen.parameters(), lr=0.001)
printing_steps = 10
true_vals = Variable(torch.ones(batch_size,1)) #torch.ones(batch_size,1)
false_vals = Variable(torch.zeros(batch_size,1)) #torch.zeros(batch_size,1)
epochs = 50

dis = discriminator()
gen = generator()
#gen.load_state_dict(torch.load('generator-tanh8200'))
#torch.save(dis.state_dict(), 'discriminator')
torch.save(gen.state_dict(), 'generator-tanh')

num_epochs = 20000
lr = 0.00001
optimizerD = optim.Adam(dis.parameters(), lr=lr)
optimizerG = optim.Adam(gen.parameters(), lr=lr)
one = torch.FloatTensor([1])
minusone = one * -1

for epoch in range(num_epochs):
    # For each batch
    for q in range(num_batches):
        i = random.randint(0,num_batches-1)
        x = btches[i]
        real_data = real_data_batches[i]
        #noisev = fakebtches[i]
        noise = torch.randn(batch_size*1,1)
        noisev = torch.FloatTensor(noise).reshape(batch_size,1)
        noisev.requires_grad=True
   
        ###########################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        fake  = gen(noisev)
        fout = Du(noisev, fake[:,1])
        to_dis = torch.cat((fake, fout),1)
        output = dis(to_dis)

        # Calculate G's loss based on this output
        errG = criterion(output, Variable(torch.ones(batch_size,1)))


        # Calculate gradients for G
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        optimizerD.zero_grad()
        # Forward pass real batch through D
        real_out = dis(real_data)
        # Calculate loss on all-real batch
        errD_real = criterion(real_out, Variable(torch.ones(batch_size,1)))
        # Calculate gradients for D in backward pass
        #errD_real.backward()
        real_out = real_out.mean()
        #real_out.backward(minusone)
        D_x = real_out.item()
        
        ## Train with all-fake batch
        
        fake  = gen(noisev)
        fout = Du(noisev, fake[:,1])
        
        fake_data = torch.cat((fake, fout),1).detach()
        fake_out = dis(fake_data)
        errD_fake = criterion(fake_out, Variable(torch.zeros(batch_size,1)))
        fake_out = fake_out.mean()
        D_G_z1 = fake_out.item()
        #errD_fake.backward()
        gradient_penalty = calc_gradient_penalty(dis, real_data.data, fake_data.data)
        #gradient_penalty.backward()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake + gradient_penalty
        errD.backward()
        # Update D
        optimizerD.step()

        #
        # Output training stats
    if epoch % 10 == 0:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f %.4f '
                  % (epoch, num_epochs,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, gradient_penalty.item()) )
                     #file=open('./pdegan_out.txt','a'))

    if epoch % 100 == 0:
        #torch.save(dis.state_dict(), 'discriminator'+str(epoch))
        torch.save(gen.state_dict(), 'generator-tanh'+str(epoch))


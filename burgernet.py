# This is a basic PINN for solving burgers equation.
# it uses an exact solution  from https://people.sc.fsu.edu/~jburkardt/py_src/burgers_solution/burgers_solution.html

import torch
import numpy as np
import os
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.autograd import grad

import burgers
from burgers import burgers_viscous_time_exact1

vtn = 40
vxn = 40
nu = 0.01 / np.pi
xlo = -1.0
xhi = +1.0
vx = np.linspace ( xlo, xhi, vxn )

tlo = 0.0
thi = 3.0 / np.pi
thi = 1.0
vt = np.linspace ( tlo, thi, vtn )

u_true = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )


#a very simple torch method to compute derivatives.
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads

#no attempt here to optimize the number or size of layers.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.il  = nn.Linear(2,80)
        self.mi  = nn.Linear(80,80)
        self.mi1  = nn.Linear(80,80)
        self.mi2 = nn.Linear(80,40)
        self.ol  = nn.Linear(40,1)
        self.tn  = nn.Tanh()

    def forward(self,x,t):
        u = torch.cat((x, t), 1)
        hidden1 = self.il(u)
        hidden2 = self.mi(self.tn(hidden1))
        hidden2a = self.mi1(self.tn(hidden2))
        hidden3 = self.mi2(self.tn(hidden2a))
        out =     self.ol(self.tn(hidden3))
        return out

def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]

def f(x,t):
    u = mynet(x,t)
    #u = [v[0], v[1]]
    u_t = nth_derivative(flat(u), wrt=t, n=1)
    u_x = nth_derivative(flat(u), wrt=x, n=1)
    u_xx = nth_derivative(flat(u_x), wrt=x, n=1)
    w = torch.tensor(0.01/np.pi)
    f = u_t + u*u_x - w*u_xx
    return f
#ivx is the set of x axis points as a tensor
#ivt is the set of t axis points as a tensor
ivx = torch.from_numpy(vx).float()
ivx =  ivx.reshape(vxn,1)
ivx.requires_grad = True

ivt = torch.from_numpy(vt).float()
ivt = ivt.reshape(vtn,1)
ivt.requires_grad = True

# we need to create the boundary conditions.  There are three segments
# when x = -1, x = 1 and t=0
#it0 is the t=0 x values
#ix1 is the x=1 is all 1s
#ixm1 is the x=-1 is all -1s
it0 = torch.zeros(vtn, dtype=torch.float, requires_grad=True).reshape(vtn,1)
ix1 = torch.zeros(vxn, dtype=torch.float, requires_grad=True).reshape(vxn,1)+1.0
ixm1  = torch.zeros(vxn, dtype=torch.float, requires_grad=True).reshape(vxn,1)-1.0
#now the values for u on the above boundary segments
xbndr = torch.from_numpy(u_true[:,0]).float() #boundary for t = 0
xbndr = xbndr.reshape(vxn,1)
ixm1t = torch.from_numpy(u_true[0,:]).float().reshape(4,10,1) #boundary u values for x = -1
ix1t  = torch.from_numpy(u_true[vxn-1,:]).float().reshape(4,10,1) #uvals for x = 1

#divide values into four batches of 10.
mt = it0.reshape(4,10,1) #this is all zeros
mx = ivx.reshape(4,10,1) #this is the full x-axis
mu = xbndr.reshape(4,10,1) #this is the u values on t=0 
#now (mx, mt, mu) is the initial condition

mt2 = ivt.reshape(4,10,1) #the t axis in four batches
mx2 = ixm1.reshape(4,10,1) #all -1s 
mx3 = ix1.reshape(4,10,1)  #all 1s
zeros  = torch.zeros(vtn, dtype=torch.float, requires_grad=True).reshape(4,10,1)

m1 = list(zip(mx, mt,mu))
m2 = list(zip(mx2[1:4], mt2[1:4], ixm1t[1:4])) #the x=-1 boundary tripple (-1,t, u(-1,t))
m3 = list(zip(mx3[1:4], mt2[1:4], ix1t[1:4]))  #the x=1 boundary tripple (1,t, u(1,t))
#note this seems to be missing the initial segments.  could be an error.
m = m1+m2+m3
#m is a list of all the boundary tuples

bb = []
ze = zeros[0].reshape(1,10,1)
for i in range(len(ivt)):
    ts = zeros.clone()
    ts[:][:]=ivt[i]
    for j in range(4):        
        xb = mx[j].reshape(1,10,1)
        tb =ts[0].reshape(1,10,1)
        bb.append(list(zip(xb,tb,ze)))   

#bb is a list of all triplex (x, t, 0.0) for all x and t
# so it describes the interior

mynet = Net()
epocs = 200000
btch = m
btch2 = bb
losstot = np.zeros(len(m))
losstot2 = np.zeros(len(bb))
loss_fn = nn.MSELoss()
#use two optimizers.  learing rates seem to work.
optimizer = optim.SGD(mynet.parameters(), lr=0.001)
optimizer2 = optim.SGD(mynet.parameters(), lr=0.0005)

loss_fn = nn.MSELoss()
for epoc in range(1, epocs+1):
    loss2tot =  0.0
    for i in range(len(btch)):
        #pick a random boundary batch
        b = btch[np.random.randint(0, len(m))]
        #pick a random interior batch
        bf = btch2[np.random.randint(0, len(bb))]
        optimizer.zero_grad()
        optimizer2.zero_grad()
        outputs = mynet(b[0], b[1])
        outputsf = f(bf[0][0], bf[0][1])
        loss = loss_fn(outputs,b[2])
        loss2 = loss_fn(outputsf, bf[0][2])
        loss2tot += loss2
        losstot[i]= loss
        losst = loss
        loss.backward(retain_graph=True)
        optimizer.step()
        loss2.backward(retain_graph=True)
        optimizer2.step()
    if epoc % 500 == 0:
        loss = 0.0
        for i in range(len(m)):
            loss+=losstot[i]
        print('epoc %d bndry loss %f, f loss %f'%(epoc, float(loss), float(loss2tot)), file=open('./burger_out.txt','a'))
    if epoc % 5000 == 0:
        torch.save(mynet.state_dict(), 'burgmodel')

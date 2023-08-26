import torch
import torchvision
from torch.utils.data import Dataset, DataLoader # giriş versi bununla yapılır
import torch.optim as opt
import torch.nn.functional as F
import numpy as np
#from model import *
from densenet169_2 import *   # densenet169_2 # new_model # de169
from Dataloader4Deneme import*
from tuils.lrs_scheduler import WarmRestart, warm_restart
import time
import torch.optim as optim

import matplotlib.pyplot as plt




if __name__=="__main__":
    
    lossL = []
    train_Accuracy = []

    n_epochs = 80 # tüm veriyi 80 kez eğitim için kullanıcaz
    batchSizeTrain = 64 # veriyi 64 lü parçalara ayırıp ,her biri için ağırlık güncellemesi yapılır
   
    batchSizeTest = 64 # 1000

   
    #torchvision.models.densenet169()

    device = "cuda" # "cuda"  ,  "cpu"
    
    
    trainLoader = DataLoader(data(), batch_size=batchSizeTrain, shuffle=True) # data() # data3D()
    
    
    
    testLoader = DataLoader(dataT(), batch_size=batchSizeTest, shuffle=True) # dataT() # dataT3D()
    #testLoader = trainLoader
    
    myModel=Network_0_1().to(device) # network # DenseNet169_change_avg # Network # Network2 # Network3

    #print(myModel)
    
    lr_list = []

    lr1 = 0.0005

    optimizer = torch.optim.Adam(myModel.parameters(), lr=lr1, betas=(0.9, 0.999), eps=1e-005, weight_decay=0.00002)# lr=0.0005
    scheduler = WarmRestart(optimizer, T_max=5, T_mult=2, eta_min=0) # new # WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
    #myModel.load_state_dict(torch.load('epoc80_adam.pth'))
    timer_counter = 0
    myModel.train()
    for e in range(n_epochs):
        print(e+1)
        ## deneme
        if timer_counter == 0:
            start = time.time()
        
        
            
        son = 0
        for batch,target in trainLoader:
            
            
            
            optimizer.zero_grad() 
            
            o=myModel.forward(batch.to(device))

            ####
            
            ####

            loss = F.cross_entropy(o,torch.argmax(target, dim=1).to(device) )  # torch.argmax(target, dim=1).to(device)
            # target.to(device)
            



            lossL.append(loss)
            loss.backward()
            optimizer.step() 
            scheduler.step()# new
            optimizer.zero_grad() # # new #modelin gradyanlarını sıfırlamak için kullanılır. Bu işlem, her iterasyonda gradyanların tekrarlanmasını önlemek için yapılır
        
        
            

        hold = lr_update(lr1)   ### lr update deneme

        if hold != "a":     ### lr update deneme
            lr1 = hold
                
        lr_list.append(lr1)

        

        """pred = o.data.max(1, keepdim=True)[1]
           
        correct = pred.eq(torch.argmax(target, dim=1).to(device).data.view_as(pred)).sum()*100. / 64
        train_Accuracy.append(correct)"""


        
        
        correct = 0
        for batch,target in testLoader:
            
                o=myModel.forward(batch.to(device))
                
                pred = o.data.max(1, keepdim=True)[1]
            
                correct += pred.eq(torch.argmax(target, dim=1).to(device).data.view_as(pred)).sum()
                
        #train_Accuracy.append((correct*100)/1200)
        train_Accuracy.append(correct)

    
        #print(time.time() - start)
        
        """y.append(float(son))"""
        if timer_counter == 0:
            print(time.time() - start)
            timer_counter+=1
    
    
    correct_g = []

    myModel.eval() 
    testLoss = 0
    correct = 0
    for batch,target in testLoader:
        
            o=myModel.forward(batch.to(device))
            
            pred = o.data.max(1, keepdim=True)[1]
           
            correct += pred.eq(torch.argmax(target, dim=1).to(device).data.view_as(pred)).sum()
            correct_g.append(pred.eq(torch.argmax(target, dim=1).to(device).data.view_as(pred)).sum()*100. / 1000)
    testLoss /= len(testLoader.dataset)

  
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))
    
    
    
    #torch.save(myModel.state_dict(), 'epoc40_26_06_2023_adam.pth')
    

    """plt.plot(np.array(train_Accuracy),'ro')
    plt.subplot()
    plt.xlabel("Epoc")
    plt.ylabel("Accuracy")
    plt.show(block = True)"""
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    for n_iter in range(len(train_Accuracy)):
    
        writer.add_scalar('Accuracy/train', train_Accuracy[n_iter], n_iter)
    
    for n_iter in range(len(train_Accuracy)):
    
        writer.add_scalar('Loss/train', lossL[n_iter], n_iter)









"""
class DenseNet169_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x
"""

"""
x = torch.randn(28, 28)
x.unsqueeze_(0)
x = x.repeat(3, 1, 1)
x.shape
> torch.Size([3, 28, 28])
"""

"""
import torch
xx = torch.rand(28,28)
y =torch.stack([xx,xx,xx],0)
print(xx.shape)
print(y.shape)
print(torch.norm(y[:,:,0] - xx))
"""


import torch.nn as nn
import torch
import torch.nn.functional as F

## 64,1,224,224
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv2d(30,60,kernel_size=1,stride=2) # 64 , 30 , 56 , 56
        self.bn2 = nn.BatchNorm2d(60)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        

        # transition layer 2 ->
        self.conv3 = nn.Conv2d(60,120,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv3_drop = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        

        # transition layer 3 ->
        self.conv4 = nn.Conv2d(120,240,kernel_size=1)
        self.bn4 = nn.BatchNorm2d(240)
        self.conv4_drop = nn.Dropout2d()
        self.pool4 = nn.MaxPool2d(kernel_size=1,stride=1)

        

        # classification layer->
        self.fc1 = nn.Linear(11760, 1000)
        self.fcDout = nn.Dropout()
        self.fc2 = nn.Linear(1000, 250)
        
        self.fc3 = nn.Linear(250, 5)
        




    def forward(self, x):

        

        # starting layer ->
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]


        

        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]



       



        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)
        


       
        
        

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        x = self.pool4(x)
        x = torch.relu(x)

        


        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 11760) #   64 , 240x7x7 = 11760             64 , 240x3x3 = 2160

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc3(x)
        
        
        
        return torch.log_softmax(x, dim=1)

        

"""
nn.MaxPool2d

__________________________________________________________________________________________________________________________________
opt.SGD
optimizer = opt.SGD(myModel.parameters(), lr=learning_rate, momentum=momentum)
                                        lr      batchSizeTrain   n_epochs    momentum
Test set: Accuracy: 2297/6000 (38%) -  0.0005 -      64       -    80      -   0.99     
Test set: Accuracy: 2000/6000 (33%) -  0.0005 -     256       -    10      -   0.99     
Test set: Accuracy: 2000/6000 (33%) -  0.001  -     256       -    10      -   0.99     
Test set: Accuracy: 2002/6000 (33%) -  0.01   -     256       -    10      -   0.99                      
Test set: Accuracy: 2507/6000 (42%) -  0.001  -      64       -    60      -   0.99 
Test set: Accuracy:      DENE       -  0.01   -      64       -    60      -   0.99
Test set: Accuracy: 3259/6000 (54%) -  0.001  -      64       -    100     -   0.99     
Test set: Accuracy: 2479/6000 (41%) -  0.005  -      64       -    60      -   0.99     
Test set: Accuracy:                 -  0.008  -      64       -    60      -   0.99     
Test set: Accuracy:                 -  0.0001 -      64       -    60      -   0.99  
Test set: Accuracy:                 -  0.003  -      64       -    60      -   0.99   
__________________________________________________________________________________________________________________________________
torch.optim.Adam
optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
                                         lr     batchSizeTrain   n_epochs
Test set: Accuracy: 2262/6000 (38%)  - 0.001   -     64       -    10     
Test set: Accuracy: 2353/6000 (39%)  - 0.001   -     64       -    30  
Test set: Accuracy: 2487/6000 (41%)  - 0.001   -     64       -    60 
__________________________________________________________________________________________________________________________________
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
                                       batchSizeTrain   n_epochs
Test set: Accuracy: 3238/6000 (54%)          64       -    80     - epoc80_adam.pth
Test set: Accuracy: 3202/6000 (53%)          64       -    90     - epoc90_adam.pth
__________________________________________________________________________________________________________________________________
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.0009, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
                                       batchSizeTrain   n_epochs
Test set: Accuracy: 2882/6000 (48%)          64       -    80     - epoc80New10_05_23_adam.pth

__________________________________________________________________________________________________________________________________
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
                                       batchSizeTrain   n_epochs
Test set: Accuracy:  2361/5372 (44%)         64       -    80     - epoc80New12_05_23_adam.pth     - Test datası farklı

__________________________________________________________________________________________________________________________________
optimizer = opt.RMSprop(myModel.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
                                         lr     batchSizeTrain   n_epochs
Test set: Accuracy: 2000/6000 (33%)  - 0.001   -     64       -    10  
__________________________________________________________________________________________________________________________________
                                        lr     batchSizeTrain   n_epochs
2. uzun deneme(kayıtlı)                                        
Test set: Accuracy: 2295/6000 (38%) -  0.001  -      64       -    40     -   0.99
Test set: Accuracy: 2811/6000 (47%) -  0.001  -      64       -    80     -   0.99
Test set: Accuracy: 3142/6000 (52%) -  0.001  -      64       -    100    -   0.99
Test set: Accuracy: 3522/6000 (59%) -  0.001  -      64       -    120    -   0.99
Test set: Accuracy: 3785/6000 (63%) -  0.001  -      64       -    140    -   0.99
Test set: Accuracy: 4290/6000 (72%) -  0.001  -      64       -    160    -   0.99
Test set: Accuracy: 4781/6000 (80%) -  0.001  -      64       -    180    -   0.99
İlk deneme :
Test set: Accuracy: 3259/6000 (54%) -  0.001  -      64       -    100    -   0.99
"""


"""
nn.AvgPool2d
__________________________________________________________________________________________________________________________________
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
                                       batchSizeTrain   n_epochs
Test set: Accuracy:                         64       -    80     -  epoc80_AvgPool2d_adam.pth   - Test datası farklı




"""



class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv2d(30,60,kernel_size=1,stride=2) # 64 , 30 , 56 , 56
        self.bn2 = nn.BatchNorm2d(60)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        # transition layer 1.1 ->
        self.conv2_1 = nn.Conv2d(60,60,kernel_size=1,stride=1) # 64 , 30 , 56 , 56
        self.bn2_1 = nn.BatchNorm2d(60)
        self.conv2_drop_1 = nn.Dropout2d()
        self.pool2_1 = nn.MaxPool2d(kernel_size=1,stride=1)

        

        # transition layer 2 ->
        self.conv3 = nn.Conv2d(60,120,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv3_drop = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        # transition layer 2.1 ->
        self.conv3_1 = nn.Conv2d(120,120,kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(120)
        self.conv3_drop_1 = nn.Dropout2d()
        self.pool3_1 = nn.MaxPool2d(kernel_size=1,stride=1)

        

        # transition layer 3 ->
        self.conv4 = nn.Conv2d(120,240,kernel_size=1)
        self.bn4 = nn.BatchNorm2d(240)
        self.conv4_drop = nn.Dropout2d()
        self.pool4 = nn.MaxPool2d(kernel_size=1,stride=1)

        

        # classification layer->
        self.fc1 = nn.Linear(11760, 1000)
        self.fcDout = nn.Dropout()
        self.fc2 = nn.Linear(1000, 250)
        
        self.fc3 = nn.Linear(250, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, 5)



    def forward(self, x):

        

        # starting layer ->
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]


        

        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]


        # transition layer 1.1 ->
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.conv2_drop_1(x)
        x = self.pool2_1(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]



       



        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)

        # transition layer 2.1 ->
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.conv3_drop_1(x)
        x = self.pool3_1(x)
        x = torch.relu(x)
        


       
        
        

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        x = self.pool4(x)
        x = torch.relu(x)

        


        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 11760) #   64 , 240x7x7 = 11760             64 , 240x3x3 = 2160

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc5(x)
        
        
        
        return torch.log_softmax(x, dim=1)
    


class Network3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv2d(30,60,kernel_size=1,stride=2) # 64 , 30 , 56 , 56
        self.bn2 = nn.BatchNorm2d(60)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)


        
        # transition layer 2 ->
        self.conv3 = nn.Conv2d(60,120,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv3_drop = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        

        # transition layer 3 ->
        self.conv4 = nn.Conv2d(120,240,kernel_size=1)
        self.bn4 = nn.BatchNorm2d(240)
        self.conv4_drop = nn.Dropout2d()
        self.pool4 = nn.MaxPool2d(kernel_size=1,stride=1)

        

        # classification layer->
        self.fc1 = nn.Linear(11760, 5)
        



    def forward(self, x):

        

        # starting layer ->
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]


        

        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]






        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)

 

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        x = self.pool4(x)
        x = torch.relu(x)

        


        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 11760) #   64 , 240x7x7 = 11760             64 , 240x3x3 = 2160

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        
        
        
        
        return torch.log_softmax(x, dim=1)
    


class Network3D(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3)
        self.conv1 = nn.Conv3d(3,30,kernel_size=(7, 7, 7),stride=2,padding=3) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2))
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv3d(30,60,kernel_size=(1,1,1),stride=2) # 64 , 60 , 56 , 56
        self.bn2 = nn.BatchNorm3d(60)
        self.conv2_drop = nn.Dropout3d()
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))


        
        # transition layer 2 ->
        self.conv3 = nn.Conv3d(60,120,kernel_size=(1,1,1))
        self.bn3 = nn.BatchNorm3d(120)
        self.conv3_drop = nn.Dropout3d()
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))

        

        # transition layer 3 ->
        self.conv4 = nn.Conv3d(120,240,kernel_size=(1,1,1))
        self.bn4 = nn.BatchNorm3d(240)
        self.conv4_drop = nn.Dropout3d()
        self.pool4 = nn.MaxPool3d(kernel_size=(1,1,1),stride=(1,1,1))

        

        # classification layer->
        self.fc1 = nn.Linear(11760, 5)
        



    def forward(self, x):

        
        
        # starting layer ->
        x = self.conv1(x)
        
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]


        

        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]






        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)

 

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        x = self.pool4(x)
        x = torch.relu(x)

        

        
        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 11760) #   64 , 240x7x7 = 11760             64 , 240x3x3 = 2160

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        
        
        
        
        return torch.log_softmax(x, dim=1)
    

    """
    Bu hatanın nedeni, girdinizin ve ağırlığınızın boyutlarının uyumsuz olmasıdır. Girdinizin şekli [1, 64, 3, 224, 224] ise, bu demektir ki:
                                                                                                    [64, 3, 1, 224, 224] olmalı [DENE]
    Girdiniz 5 boyutludur.
    Girdinizin ilk boyutu 1’dir, yani tek bir örnek içerir.
    Girdinizin ikinci boyutu 64’tür, yani her örnekte 64 kanal vardır.
    Girdinizin üçüncü boyutu 3’tür, yani her kanalda 3 derinlik vardır.
    Girdinizin dördüncü ve beşinci boyutları 224’tür, yani her derinlikte 224x224 piksel vardır.
    Ağırlığınızın şekli [30, 3, 7, 7, 7] ise, bu demektir ki:

    Ağırlığınız 5 boyutludur.
    Ağırlığınızın ilk boyutu 30’dur, yani çıktı kanallarının sayısıdır.
    Ağırlığınızın ikinci boyutu 3’tür, yani girdi kanallarının sayısıdır.
    Ağırlığınızın üçüncü, dördüncü ve beşinci boyutları 7’dir, yani çekirdek (kernel) boyutudur.
    Bu durumda, girdinizin ikinci boyutu (64) ile ağırlığınızın ikinci boyutu (3) eşleşmediği için hata alırsınız. Bu sorunu çözmek için, girdinizin ikinci boyutunu ağırlığınızın ikinci boyutuna eşit yapmanız gerekir. Bunun için girdinizin şeklini [64, 3, 1, 224, 224] olarak değiştirebilirsiniz. Böylece:

    Girdinizin ilk boyutu 64 olur, yani toplam örnek sayısıdır.
    Girdinizin ikinci boyutu 3 olur, yani her örnekteki kanal sayısıdır.
    Girdinizin üçüncü boyutu 1 olur, yani her kanaldaki derinlik sayısıdır.
    Girdinizin dördüncü ve beşinci boyutları aynı kalır.
    Girdinizin şeklini değiştirmek için PyTorch’un unsqueeze veya permute gibi fonksiyonlarını kullanabilirsiniz
    """



    


#              Network3D
#                              Accuracy
# 0.005   return lr - 0.01       %41
# 0.0005   return lr - 0.01      %47
# 0.0005   return lr - 0.01      %48
# 0.0005   return lr + 0.01      %48   #####
# 0.0005   return lr + 0.02      %48
# 0.0005   return lr + 0.05      %43
# 0.0005   return lr - 0.05      %46




class Network3D_New(nn.Module):  # fazla bellek tüketiyor , hatalı
    def __init__(self):
        super().__init__()
        # nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3)
        self.conv1 = nn.Conv3d(3,30,kernel_size=(1, 2, 2)) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv3d(30,60,kernel_size=(1,1,1)) # 64 , 60 , 112 , 112
        self.bn2 = nn.BatchNorm3d(60)
        self.conv2_drop = nn.Dropout3d()
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))  # 64 , 60 , 56 , 56 


        
        # transition layer 2 ->
        self.conv3 = nn.Conv3d(60,120,kernel_size=(1,1,1)) # 64 , 120 , 56 , 56 
        self.bn3 = nn.BatchNorm3d(120)
        self.conv3_drop = nn.Dropout3d()
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2)) # 64 , 120 , 28 , 28 

        
        
        

        

        # classification layer->
        self.fc1 = nn.Linear(60840, 5)
        
        



    def forward(self, x):

        
        
        # starting layer ->
        x = self.conv1(x)
        
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]

        
        
        
        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]

        




        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)

        print(x.shape)

        # transition layer 3 ->
        

        
        
        
        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 60840) #   torch.Size([64, 240, 3, 14, 14])

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        
        
        
        
        
        return torch.log_softmax(x, dim=1)


class Network_0_1(nn.Module):  # # %63 2D model
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3) # 64 , 30 , 112 , 112
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv2d(30,60,kernel_size=1,stride=2) # 64 , 30 , 56 , 56
        self.bn2 = nn.BatchNorm2d(60)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        #self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        

        # transition layer 2 ->
        self.conv3 = nn.Conv2d(60,120,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv3_drop = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(kernel_size=1,stride=1)
        #self.pool3 = nn.AvgPool2d(kernel_size=1,stride=1)
        

        # transition layer 3 ->
        self.conv4 = nn.Conv2d(120,240,kernel_size=1)
        self.bn4 = nn.BatchNorm2d(240)
        self.conv4_drop = nn.Dropout2d()
        self.pool4 = nn.MaxPool2d(kernel_size=1,stride=1)
        #self.pool4 = nn.AvgPool2d(kernel_size=1,stride=1)
        

        # classification layer->
        self.fc1 = nn.Linear(47040, 1000)
        self.fcDout = nn.Dropout()
        self.fc2 = nn.Linear(1000, 250)
        
        self.fc3 = nn.Linear(250, 5)
        




    def forward(self, x):

        

        # starting layer ->
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        #y1 = x   # [64, 30, 55, 55]


        

        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = torch.relu(x)  # [64, 3840, 14, 14]



       



        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        x = torch.relu(x)
        


       
        
        

        # transition layer 3 ->
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        x = self.pool4(x)
        x = torch.relu(x)

        


        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 47040) #   64 , 240x7x7 = 11760             64 , 240x3x3 = 2160

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fcDout(x)
        x = self.fc3(x)
        
        
        
        return torch.log_softmax(x, dim=1)



#  optimizer = torch.optim.Adam(myModel.parameters(), lr=lr1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)# lr=0.0005
#  scheduler = WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)
#              2D
#   LR           LR +            Accuracy         model                          scheduler                                                     optimizer                                 time 
# 0.0005    return lr + 0.001       49%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.005       49%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr - 0.001       47%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.01        48%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.06        47%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002            4.7 sn
# 0.0005    return lr + 0.08        50%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.089       47%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.085       48%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.0825      49%         Network_0_1        WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.08        50%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002
# 0.0005    return lr + 0.08        47%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.002
# 0.0005    return lr + 0.08        48%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0002
# 0.0005    return lr + 0.08        48%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000002   ->  lr değiştir ve dene
# 0.0005    return lr + 0.08        50%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-005, weight_decay=0.00002   ->  lr değiştir ve dene
# 0.0005    return lr + 0.08        49%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-009, weight_decay=0.00002
# 0.00005   return lr + 0.08        49%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000002 

# 0.0009    return lr + 0.08        50%         Network_0_1        WarmRestart(optimizer, T_max=5 , T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-005, weight_decay=0.00002



#  AÇIKLAMA
#  WarmRestart(optimizer, T_max=10, T_mult=2, eta_min=0)   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002    

"""
Bu kod, WarmRestart adlı bir program ile optimizatörün öğrenme oranını planlar. Bu program, öğrenme oranını belirli aralıklarla yeniden başlatarak kozinüs eğrisi şeklinde değiştiren bir öğrenme oranı programlayıcısıdır. Bu, modelin yerel minimumlardan kaçınmasına ve daha iyi bir genelleme performansına ulaşmasına yardımcı olabilir. Kodda belirtilen parametreler şunlardır:

T_max: ilk yeniden başlatma aralığının uzunluğudur. Bu kodda 10 olarak ayarlanmıştır.
T_mult: her yeniden başlatmadan sonra yeniden başlatma aralığının çarpılacağı bir faktördür. Bu kodda 2 olarak ayarlanmıştır, yani aralık her seferinde iki katına çıkacaktır.
eta_min: öğrenme oranının alabileceği minimum değerdir. Bu kodda 0 olarak ayarlanmıştır, yani öğrenme oranı sıfıra yaklaşabilir.
Bu kod ayrıca optimizatörün diğer parametrelerini de belirtir:

betas: gradyanların hareketli ortalamalarını hesaplamak için kullanılan iki katsayıdır. İlk katsayı gradyanın momentumunu, ikinci katsayı ise gradyanın RMSProp’unu kontrol eder. Bu kodda sırasıyla 0.9 ve 0.999 olarak ayarlanmıştır.
eps: sayısal kararlılık için paydada eklenen bir terimdir. Bu kodda 1e-08 olarak ayarlanmıştır.
weight_decay: ağırlık azaltma veya L2 cezası olarak da bilinen bir düzenlileştirme terimidir. Bu, aşırı uyumu önlemek ve modelin karmaşıklığını azaltmak için parametreleri her adımda bir miktar azaltır. Bu kodda 0.00002 olarak ayarlanmıştır.
Bu kodun amacı, optimizatörün öğrenme oranını dinamik bir şekilde ayarlamak ve modelin eğitim sürecini iyileştirmektir.

: SGDR: Stochastic Gradient Descent with Warm Restarts

"""

def lr_update(lr):
    
    
    return lr + 0.08 # -

def lr_update_y(lr):
    
    
    return lr**2  - 0.08 # -

def lr_update(lr):
    
    
    return lr + 0.08 # -

def lr_update_y(lr,n_epochs):

    if n_epochs < 80:      # % 66
        return lr + 0.00 # 0.00
    
    elif n_epochs < 60 :
        return lr + 0.04 # 0.04
    
    elif n_epochs < 30 :
        return lr + 0.08 # 0.008

def lr_update_y_TOP(lr,n_epochs):  # TOP % 65

    if n_epochs < 80:
        return lr + 0.00  # 0.00
    
    elif n_epochs < 60 :
        return lr + 0.04
    
    elif n_epochs < 30 :
        return lr + 0.08 # -
    




class Network3D_New_dicom(nn.Module):  # %72 3D model
    def __init__(self):
        super().__init__()
        # nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3)
        self.conv1 = nn.Conv3d(3,30,kernel_size=(1, 3, 3)) # 64 , 30 , 112 , 112
        self.pool1 = nn.AvgPool3d(kernel_size=(1,3,3),stride=(1,3,3))
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv3d(30,60,kernel_size=(1,3,3)) # 64 , 60 , 112 , 112
        self.bn2 = nn.BatchNorm3d(60)
        self.conv2_drop = nn.Dropout3d()
        self.pool2 = nn.AvgPool3d(kernel_size=(1,3,3),stride=(1,3,3))  # 64 , 60 , 56 , 56 


        
        # transition layer 2 ->
        self.conv3 = nn.Conv3d(60,120,kernel_size=(1,3,3)) # 64 , 120 , 56 , 56 
        self.bn3 = nn.BatchNorm3d(120)
        self.conv3_drop = nn.Dropout3d()
        self.pool3 = nn.AvgPool3d(kernel_size=(1,1,1),stride=(1,1,1)) # 64 , 120 , 28 , 28 



        self.conv4 = nn.Conv3d(120,240,kernel_size=(1,1,1)) # 64 , 120 , 56 , 56 
        self.bn4 =nn.BatchNorm3d(240)
        self.conv4_drop = nn.Dropout3d()
        self.pool4 = nn.AvgPool3d(kernel_size=(1,1,1),stride=(1,1,1)) # 64 , 120 , 28 , 28 

        
        
        

        

        # classification layer->
        self.fc1 = nn.Linear(116160, 1000)  #  5880
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 5)
        
        
        



    def forward(self, x):

        
        
        # starting layer ->
        x = self.conv1(x)
        
        x = self.pool1(x)
        x = torch.relu(x)
        #x = torch.softmax(x)
        #y1 = x   # [64, 30, 55, 55]

        
        
        
        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        #x = torch.relu(x)  # [64, 3840, 14, 14]
        x = torch.nn.functional.relu(x)
        #x = torch.softmax(x)
        




        # transition layer 2 ->
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv3_drop(x)
        x = self.pool3(x)
        #x = torch.relu(x)
        x = torch.nn.functional.relu(x)
        #x = torch.softmax(x)
        #print(x.shape)

        # transition layer 3 ->

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv4_drop(x)
        x = self.pool4(x)
        #x = torch.relu(x)
        x = torch.nn.functional.relu(x)
        
        
        
        
        #print(x.shape)
        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 116160) #   torch.Size([64, 240, 3, 14, 14])

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        
        
        
        
        
        
        
        
        return torch.log_softmax(x, dim=1)




class Network3D_New_dicom2(nn.Module):  # %72 3D model
    def __init__(self):
        super().__init__()
        # nn.Conv2d(1,30,kernel_size=7,stride=2,padding=3)
        self.conv1 = nn.Conv3d(3,30,kernel_size=(1, 3, 3)) # 64 , 30 , 112 , 112
        self.pool1 = nn.AvgPool3d(kernel_size=(1,3,3),stride=(1,3,3))
        
        

        # transition layer 1 ->
        self.conv2 = nn.Conv3d(30,60,kernel_size=(1,2,2)) # 64 , 60 , 112 , 112
        self.bn2 = nn.BatchNorm3d(60)
        self.conv2_drop = nn.Dropout3d()
        self.pool2 = nn.AvgPool3d(kernel_size=(1,2,2),stride=(1,2,2))  # 64 , 60 , 56 , 56 




        # classification layer->
        self.fc1 = nn.Linear(77760, 200)  #  5880
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(200, 5)
        
        


    def forward(self, x):

        
        
        # starting layer ->
        x = self.conv1(x)
        
        x = self.pool1(x)
        x = torch.relu(x)
        #x = torch.softmax(x)
        #y1 = x   # [64, 30, 55, 55]

        
        
        
        
        

        # transition layer 1 ->
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        #x = torch.relu(x)  # [64, 3840, 14, 14]
        x = torch.nn.functional.relu(x)
        #x = torch.softmax(x)
        

        #print(x.shape)


        
        
        
        
        
        #print(x.shape)
        # Convolüsyon çıkışını vektör haline getir
        x = x.view(-1, 77760) #   torch.Size([64, 240, 3, 14, 14])

        # Lineer katmanların çıkışını hesapla
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        

        
        return torch.log_softmax(x, dim=1)
    
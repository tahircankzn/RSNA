import os
import csv
from PIL import Image
import pandas as pd
import skimage
import numpy as np

# resimlerin bulunduğu klasörün yolu
folder_path = 'Train224/'

# dönüştürülmüş verilerin kaydedileceği CSV dosyası adı
csv_file_name = 'veri_seti_new_Test_chanel3.csv'


#verilerim
df1 = pd.read_csv('epiduralT.csv')
df2 = pd.read_csv('intraparenchymalT.csv')
df3 = pd.read_csv('intraventricularT.csv')
df4 = pd.read_csv('noneT.csv')
df5 = pd.read_csv('subarachnoidT.csv')
df6 = pd.read_csv('subduralT.csv')

counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0
counter6 = 0
index = 0

# CSV dosyasını oluştur
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # tüm resim dosyalarını klasörden al
    resim_dosya_adlari = os.listdir(folder_path)

    while True:
        index+=1
        # resim dosyasının yolunu oluştur
        if index >= len(resim_dosya_adlari) - 1:
            break
        dosya_adi = resim_dosya_adlari[index]
        resim_yolu = os.path.join(folder_path, dosya_adi)

        # resmi aç ve boyutunu yeniden boyutlandır
        resim = Image.open(resim_yolu)
        

        # resimdeki piksellerin gri tonlarını al
        #gri_tonlar = list(resim.getdata())
        gri_tonlar = []

        filter1 = skimage.filters.prewitt # skimage.filters.gaussian 2.   ,  skimage.filters.prewitt 3.
        
        #gri_tonlar = [int(sum(piksel)/3) for piksel in pikseller]
        
        # resmin adını gri tonlar listesinin başına ekle
        if dosya_adi[:-4] in list(df1["epidural"]) and counter1 < 200:
            #gri_tonlar.insert(0,[1,0,0,0,0,0])
            gri_tonlar = list(resim.getdata())
            gri_tonlar = list(filter1(np.array(gri_tonlar)))
            #gri_tonlar.insert(0,dosya_adi[:-4]+"/epidural")
            gri_tonlar.insert(0,1)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            csv_writer.writerow(gri_tonlar)
            counter1+=1
            
        elif dosya_adi[:-4] in list(df2["intraparenchymal"]) and counter2 < 200:
            gri_tonlar = list(resim.getdata())
            gri_tonlar = list(filter1(np.array(gri_tonlar)))
            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,1)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            csv_writer.writerow(gri_tonlar)
            counter2+=1
            
        elif dosya_adi[:-4] in list(df3["intraventricular"]) and counter3 <200:
            #gri_tonlar.insert(0,[0,0,1,0,0,0])
            gri_tonlar = list(resim.getdata())
            gri_tonlar = list(filter1(np.array(gri_tonlar)))
            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraventricular")
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,1)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            csv_writer.writerow(gri_tonlar)
            counter3+=1
            
        elif dosya_adi[:-4] in list(df4["none"]) and counter4 < 200:
            #gri_tonlar.insert(0,[0,0,0,0,0,0])
            gri_tonlar = list(resim.getdata())
            gri_tonlar = list(filter1(np.array(gri_tonlar)))
            #gri_tonlar.insert(0,dosya_adi[:-4]+"/none")
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            csv_writer.writerow(gri_tonlar)
            counter4+=1
            
        elif dosya_adi[:-4] in list(df5["subarachnoid"]) and counter5 < 200:
            #gri_tonlar.insert(0,[0,0,0,1,0,0])
            gri_tonlar = list(resim.getdata())
            gri_tonlar = list(filter1(np.array(gri_tonlar)))
            #gri_tonlar.insert(0,dosya_adi[:-4]+"/subarachnoid")
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,1)
            gri_tonlar.insert(4,0)
            gri_tonlar.insert(5,0)
            csv_writer.writerow(gri_tonlar)
            counter5+=1
            
        elif dosya_adi[:-4] in list(df6["subdural"]) and counter6 < 200:
            #gri_tonlar.insert(0,[0,0,0,0,1,0])
            gri_tonlar = list(resim.getdata())
            gri_tonlar = list(filter1(np.array(gri_tonlar)))
            #gri_tonlar.insert(0,dosya_adi[:-4]+"/subdural")
            gri_tonlar.insert(0,0)
            gri_tonlar.insert(1,0)
            gri_tonlar.insert(2,0)
            gri_tonlar.insert(3,0)
            gri_tonlar.insert(4,1)
            gri_tonlar.insert(5,0)
            csv_writer.writerow(gri_tonlar)
            counter6+=1

        elif counter1 == 200 and counter2 == 200 and counter3 == 200 and counter4 == 200 and counter5 == 200 and counter6 == 200 :
            break
        
            

        # gri tonları csv dosyasına yazdır
        
       


"""

epidural         [1,0,0,0,0,0]
intraparenchymal [0,1,0,0,0,0]
intraventricular [0,0,1,0,0,0]
subarachnoid     [0,0,0,1,0,0]
subdural         [0,0,0,0,1,0]
none             [0,0,0,0,0,0]

"""
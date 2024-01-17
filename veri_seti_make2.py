import os
import csv
from PIL import Image
import pandas as pd
import skimage
import numpy as np

# resimlerin bulunduğu klasörün yolu
folder_path = 'archive/full_data'




# validasyon
df1_v = pd.read_csv('epidural_full_valid.csv')
df2_v = pd.read_csv('intraparenchymal_full_valid.csv') # _valid
df3_v = pd.read_csv('intraventricular_full_valid.csv')
df4_v = pd.read_csv('any_full_valid.csv')
df5_v = pd.read_csv('subarachnoid_full_valid.csv')
df6_v = pd.read_csv('subdural_full_valid.csv')

# train

df1_t = pd.read_csv('epidural_full.csv')
df2_t = pd.read_csv('intraparenchymal_full.csv') 
df3_t = pd.read_csv('intraventricular_full.csv')
df4_t = pd.read_csv('any_full.csv')
df5_t = pd.read_csv('subarachnoid_full.csv')
df6_t = pd.read_csv('subdural_full.csv')

counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0
counter6 = 0

index = 0

hatalı = []


veriler = {
                "epidural" : df1_v,
                "intraparenchymal" : df2_v,
                "intraventricular" : df3_v,
                "any" : df4_v,
                "subarachnoid" : df5_v,
                "subdural" : df6_v
               }

veriler_t = {
                "epidural" : df1_t,
                "intraparenchymal" : df2_t,
                "intraventricular" : df3_t,
                "any" : df4_t,
                "subarachnoid" : df5_t,
                "subdural" : df6_t
               }

modes = ["valid","train"]

for mode in modes:
    # dönüştürülmüş verilerin kaydedileceği CSV dosyası adı
    
    csv_file_name = f'__data_{mode}__.csv'
    if mode == "valid":
        veriler_ = veriler
    else:
        veriler_ = veriler_t

    counter1,counter2,counter3,counter4,counter5,counter6 = 0,0,0,0,0,0

    with open(csv_file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
    

        for kanama in veriler_.keys():
            for img in veriler_[kanama][kanama]:
                #print(f"D:/derin_ogrenme/python/Tomogrofi_proje/proje_kodlarim/archive/train224/{img}.png")
                if img not in veriler_.keys():
                    try:
                        resim = Image.open(f"{folder_path}/{img}.png") # D:/derin_ogrenme/python/Tomogrofi_proje/proje_kodlarim/
                        """if [counter2,counter3,counter4,counter5,counter6] == [800,800,800,800,800]: #[7000,7000,7000,7000,7000]
                            break"""
                        if kanama == "intraparenchymal" : # and counter2 < 800
                            gri_tonlar = list(resim.getdata())
                            
                            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
                            # intraparenchymal [0,1,0,0,0,0]
                            gri_tonlar.insert(0,0)
                            gri_tonlar.insert(1,1)
                            gri_tonlar.insert(2,0)
                            gri_tonlar.insert(3,0)
                            gri_tonlar.insert(4,0)
                            gri_tonlar.insert(5,0)
                            
                            csv_writer.writerow(gri_tonlar)
                            counter2+=1

                        elif kanama == "intraventricular":
                            gri_tonlar = list(resim.getdata())
                            
                            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
                            # intraventricular [0,0,1,0,0,0]
                            gri_tonlar.insert(0,0)
                            gri_tonlar.insert(1,0)
                            gri_tonlar.insert(2,1)
                            gri_tonlar.insert(3,0)
                            gri_tonlar.insert(4,0)
                            gri_tonlar.insert(5,0)
                            
                            csv_writer.writerow(gri_tonlar)
                            counter3+=1

                        elif kanama == "any" : # and counter4 < 50000
                            gri_tonlar = list(resim.getdata())
                            
                            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
                            # none [0,0,0,0,0,1]
                            gri_tonlar.insert(0,0)
                            gri_tonlar.insert(1,0)
                            gri_tonlar.insert(2,0)
                            gri_tonlar.insert(3,0)
                            gri_tonlar.insert(4,0)
                            gri_tonlar.insert(5,1)
                            
                            csv_writer.writerow(gri_tonlar)
                            counter4+=1

                        elif kanama == "subarachnoid":
                            gri_tonlar = list(resim.getdata())
                            
                            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
                            # subarachnoid  [0,0,0,1,0,0]
                            gri_tonlar.insert(0,0)
                            gri_tonlar.insert(1,0)
                            gri_tonlar.insert(2,0)
                            gri_tonlar.insert(3,1)
                            gri_tonlar.insert(4,0)
                            gri_tonlar.insert(5,0)
                            
                            csv_writer.writerow(gri_tonlar)
                            counter5+=1

                        elif kanama == "subdural":
                            gri_tonlar = list(resim.getdata())
                            
                            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
                            # subdural  [0,0,0,0,1,0]
                            gri_tonlar.insert(0,0)
                            gri_tonlar.insert(1,0)
                            gri_tonlar.insert(2,0)
                            gri_tonlar.insert(3,0)
                            gri_tonlar.insert(4,1)
                            gri_tonlar.insert(5,0)
                            
                            csv_writer.writerow(gri_tonlar)
                            counter6+=1
                        elif kanama == "epidural":
                            gri_tonlar = list(resim.getdata())
                            
                            #gri_tonlar.insert(0,dosya_adi[:-4]+"/intraparenchymal")
                            gri_tonlar.insert(0,1)
                            gri_tonlar.insert(1,0)
                            gri_tonlar.insert(2,0)
                            gri_tonlar.insert(3,0)
                            gri_tonlar.insert(4,0)
                            gri_tonlar.insert(5,0)
                            csv_writer.writerow(gri_tonlar)
                            counter1+=1
                        """else:
                            print(img)"""
                    except:
                        print(f"{kanama}-{img}")
                        hatalı.append(f"{kanama}-{img}")
                        pass
                    
                    

                    print(counter1,counter2,counter3,counter4,counter5,counter6)

            
if len(hatalı) != 0:
    with open(csv_file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in hatalı:
            csv_writer.writerow(i)
        
        
       


"""

epidural         [1,0,0,0,0,0]
intraparenchymal [0,1,0,0,0,0]
intraventricular [0,0,1,0,0,0]
subarachnoid     [0,0,0,1,0,0]
subdural         [0,0,0,0,1,0]
none             [0,0,0,0,0,1]

"""
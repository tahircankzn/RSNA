import os
import csv
from PIL import Image

# resimlerin bulunduğu klasörün yolu
folder_path = 'test1/'

# dönüştürülmüş verilerin kaydedileceği CSV dosyası adı
csv_file_name = 'test.csv'

# dönüştürülecek resimlerin boyutu
resim_boyutu = (48, 48)

# CSV dosyasını oluştur
with open(csv_file_name, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # tüm resim dosyalarını klasörden al
    resim_dosya_adlari = os.listdir(folder_path)

    for dosya_adi in resim_dosya_adlari:
        # resim dosyasının yolunu oluştur
        resim_yolu = os.path.join(folder_path, dosya_adi)

        # resmi aç ve boyutunu yeniden boyutlandır
        resim = Image.open(resim_yolu)
        resim = resim.resize(resim_boyutu)

        # resimdeki piksellerin gri tonlarını al
        pikseller = list(resim.getdata())
        gri_tonlar = [int(sum(piksel)/3) for piksel in pikseller]

        # resmin adını gri tonlar listesinin başına ekle
        if "cat" in dosya_adi:
            gri_tonlar.insert(0,1)
        else:
            gri_tonlar.insert(0,0)

        # gri tonları csv dosyasına yazdır
        csv_writer.writerow(gri_tonlar)

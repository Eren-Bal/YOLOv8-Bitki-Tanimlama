import cv2
from ultralytics import YOLO
from google.colab import files
from google.colab.patches import cv2_imshow

print("Sadece YOLO (Göz ve Beyin) Sisteme Yükleniyor...")

# Sadece best.pt dosyasını yüklüyoruz
yolo_model = YOLO('/content/best.pt')

print("\n Lütfen test edilecek tarladan bir resim yükleyin:")
uploaded = files.upload()

for filename in uploaded.keys():
    img = cv2.imread(filename)

    # YOLO resmi inceliyor (Hem yerini buluyor hem de biliyorsa sınıfını tahmin ediyor)
    print("\nYOLO analiz ediyor...")
    results = yolo_model.predict(img, conf=0.25)

    # YOLO'nun kendi otomatik çizim motoruyla sonucu resmin üzerine basıyoruz
    sonuc_resmi = results[0].plot()

    print("\n YOLO TEST SONUCU:")
    cv2_imshow(sonuc_resmi)
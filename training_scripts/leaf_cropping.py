import os
import cv2
import yaml

# 1. Yolları belirliyoruz
base_dir = '/content/plantdoc-1'
yaml_path = f'{base_dir}/data.yaml'
out_dir = '/content/CNN_Dataset'

# 2. Sınıf (Hastalık/Bitki) isimlerini okuyoruz
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
classes = data['names']

# 3. CNN için tertemiz klasör yapısını kuruyoruz
for split in ['train', 'valid']:
    for cls_name in classes:
        os.makedirs(f"{out_dir}/{split}/{cls_name}", exist_ok=True)

# 4. YOLO koordinatlarına göre makasla kesme işlemi
def crop_and_save(split):
    img_dir = f"{base_dir}/{split}/images"
    lbl_dir = f"{base_dir}/{split}/labels"

    if not os.path.exists(img_dir): return

    for img_name in os.listdir(img_dir):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.rsplit('.', 1)[0] + '.txt')

        if not os.path.exists(lbl_path): continue

        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5: continue
            cls_id, x_c, y_c, bw, bh = map(float, parts)
            cls_name = classes[int(cls_id)]

            # YOLO oranlarını gerçek piksellere çevirme
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            # Resim dışına taşmayı engelleme
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]

            # Kesilen yaprağı ilgili hastalık klasörüne kaydetme
            if crop.size != 0:
                out_path = f"{out_dir}/{split}/{cls_name}/{img_name.split('.')[0]}_crop_{i}.jpg"
                cv2.imwrite(out_path, crop)

print(" YOLO koordinatları kullanılarak yapraklar kesiliyor...")
crop_and_save('train')
print(" Test (Valid) yaprakları kesiliyor...")
crop_and_save('valid')
print(f"\n Bitti! CNN için %100 izole edilmiş yapraklar '{out_dir}' klasöründe hazır.")
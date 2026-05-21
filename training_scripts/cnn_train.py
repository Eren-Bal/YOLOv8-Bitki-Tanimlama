import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os

# 1. LABORATUVAR İÇİN ÖZEL VERİ ARTIRMA
# Beyaz/temiz arka plan olduğu için resmi her açıdan bozup modele ezber bozduruyoruz
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,      # Daha geniş açılı döndürme
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,         # Hastalık lekelerini yakından görmek için zoom artırıldı
    horizontal_flip=True,
    vertical_flip=True,     # Yaprak ters de çevrilebilir
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim verilerini YOLO'nun zaten temizlediği klasörden çekiyoruz
train_generator = train_datagen.flow_from_directory(
    '/content/CNN_Dataset/train', target_size=(224, 224), batch_size=32, class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    '/content/CNN_Dataset/valid', target_size=(224, 224), batch_size=32, class_mode='categorical'
)
num_classes = len(train_generator.class_indices)

# 2. LABORATUVAR MİMARİSİ (Agresif Havuzlama ile Odaklanmış CNN)
model = Sequential([
    Input(shape=(224, 224, 3)),

    # 1. Blok
    Conv2D(32, (3, 3), activation='relu'), BatchNormalization(),
    MaxPooling2D(2, 2),

    # 2. Blok
    Conv2D(64, (3, 3), activation='relu'), BatchNormalization(),
    MaxPooling2D(2, 2),

    # 3. Blok
    Conv2D(128, (3, 3), activation='relu'), BatchNormalization(),
    MaxPooling2D(2, 2),

    # 4. Blok (Agresif Sıkıştırma)
    Conv2D(256, (3, 3), activation='relu'), BatchNormalization(),
    MaxPooling2D(2, 2),

    # 5. Blok (Zirve - Damar ve Leke Analizi)
    Conv2D(512, (3, 3), padding='same', activation='relu'), BatchNormalization(),
    GlobalAveragePooling2D(),

    # Karar Katmanı
    Dense(512, activation='relu'),
    Dropout(0.5), # Sert unutma oranı
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# 3. MOTOR VE AKILLI SİSTEMLER
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Maraton Modu için Sabrı Artırılmış Fren Sistemleri
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n Laboratuvar Modeli (Saf CNN) Eğitime Başlıyor...")

# 4. EĞİTİM (50 Epoch Maratonu)
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr]
)

# 5. OTOMATİK KAYIT
model.save('/content/KitapCNN_Laboratuvar_Beyni.h5')
print("\n Laboratuvar Modeli başarıyla kaydedildi! Sol menüden indirebilirsiniz.")
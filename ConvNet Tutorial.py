import pandas as pd
import os
import numpy
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
def load_image(folder, label_value, image_size=64, max_image=None):
    images, labels = [], []
    
    filenames = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    if max_image:
        filenames = filenames[:max_image]
    
    total_files = len(filenames)
    if total_files == 0:
        print(f"Không tìm thấy file .jpg nào trong thư mục {folder}")
        return images, labels

    for i, filename in enumerate(filenames):
        print(f"Đang xử lý thư mục '{os.path.basename(folder)}': Ảnh {i+1}/{total_files}", end='\r')
        
        try:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = img.resize((image_size, image_size))
            labels.append(label_value)
            images.append(numpy.array(img))
            
        except Exception as e:
            print(f"\nLỗi khi tải ảnh: {filename}. Lỗi: {e}")
            continue

    print(f"\nHoàn tất tải {len(images)} ảnh từ thư mục '{os.path.basename(folder)}'.")
    return images, labels


cat_images, cat_labels = load_image("F:/Data Analysic/PetImages/Cat", 0 )
dog_images, dog_labels = load_image("F:/Data Analysic/PetImages/Dog", 1)
X = numpy.array(cat_images + dog_images)
y = numpy.array(cat_labels + dog_labels)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=3,verbose=1)
early_stopping = EarlyStopping(monitor="val_loss",patience=5,verbose=1,restore_best_weights=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nBắt đầu quá trình huấn luyện mô hình...")
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32, validation_data=(X_test, to_categorical(y_test)), callbacks=[reduce_lr, early_stopping])

print("\nĐánh giá mô hình trên tập dữ liệu kiểm tra...")
loss, accuracy = model.evaluate(X_test, to_categorical(y_test))
print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")
# Nếu không có ảnh nào được tải, dừng chương trình

   
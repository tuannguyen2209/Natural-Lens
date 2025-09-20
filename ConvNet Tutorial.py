import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import tensorflow as tf
print(tf.__version__)
# In đoạn mã này ngay sau khi import tensorflow
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Chỉ định TensorFlow chỉ sử dụng GPU đầu tiên nếu có nhiều GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Thông báo lỗi nếu có vấn đề
    print(e)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
DATA_DIR = "F:/Data Analysic/Convolutional-Neural-Networks/PetImages" 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

print("Tạo luồng dữ liệu training...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', 
    subset='training'         
)

print("Tạo luồng dữ liệu validation...")
validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  
)
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

print("\nBắt đầu quá trình huấn luyện mô hình...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30, 
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)
print("\nĐánh giá mô hình trên tập dữ liệu validation...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Độ chính xác trên tập validation: {accuracy * 100:.2f}%")
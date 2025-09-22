import tensorflow as tf
import numpy as np
loadedmodel = tf.keras.models.load_model('pet_classifier_model.h5')

print("Mô hình đã được tải thành công.")
loadedmodel.summary()
X_new = np.random.rand(5,10)
predictions = loadedmodel.predict(X_new)
print("Dự đoán cho 5 mẫu mới:")
print(predictions)
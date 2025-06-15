import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

img_width, img_height = 48, 48
batch_size = 64
num_classes = 7
epochs = 50

train_data_dir = 'fer2013/train'
validation_data_dir = 'fer2013/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

print(f"Класи: {train_generator.class_indices}")

from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

input_tensor = Input(shape=(img_width, img_height, 3))

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_path = "../Models/emotion_resnet50_model.keras"
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[model_checkpoint, early_stopping]
)

from tensorflow.keras.models import load_model
model = load_model(checkpoint_path)

loss, accuracy = model.evaluate(validation_generator)
print(f"Loss на валідаційному наборі: {loss:.4f}")
print(f"Accuracy на валідаційному наборі: {accuracy*100:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

Y_pred = model.predict(validation_generator, validation_generator.samples // batch_size + 1)
y_pred_classes = np.argmax(Y_pred, axis=1)

y_true = validation_generator.classes

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred_classes))

print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import cv2

# saved_model_path = "emotion_recognition_model_resnet50_fer2013.keras"
# emotion_model = load_model(saved_model_path)

# emotion_labels = {v: k for k, v in train_generator.class_indices.items()}

# def preprocess_image(img_path, target_size=(48, 48), color_mode='rgb'):
#     img = image.load_img(img_path, target_size=target_size, color_mode=color_mode)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# def predict_emotion(img_path):
#     processed_img = preprocess_image(img_path, target_size=(img_width, img_height), color_mode='rgb')

#     predictions = emotion_model.predict(processed_img)
#     predicted_class_index = np.argmax(predictions[0])
#     predicted_emotion = emotion_labels[predicted_class_index]
#     confidence = np.max(predictions[0])

#     return predicted_emotion, confidence

# image_path_to_test = 'path/to/your/test_image.jpg'
# try:
#     emotion, conf = predict_emotion(image_path_to_test)
#     print(f"Прогнозована емоція: {emotion} з впевненістю: {conf:.2f}")

#     img_display = cv2.imread(image_path_to_test)
#     if img_display is not None:
#         cv2.putText(img_display, f"{emotion} ({conf:.2f})", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.imshow("Emotion Recognition", img_display)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print(f"Не вдалося завантажити зображення: {image_path_to_test}")

# except FileNotFoundError:
#     print(f"Файл не знайдено: {image_path_to_test}")
# except Exception as e:
#     print(f"Сталася помилка: {e}")
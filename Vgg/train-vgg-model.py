import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMG_WIDTH, IMG_HEIGHT = 48, 48
TRAIN_DATA_DIR = 'fer-2013/train'
VALIDATION_DATA_DIR = 'fer-2013/test'
BATCH_SIZE = 64
NUM_CLASSES = 7
EPOCHS = 50
MODEL_SAVE_PATH = "Models/emotion_vgg16_model.keras"

print("Створення генераторів даних...")
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
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

print(f"Знайдено класи: {train_generator.class_indices}")

print("Створення моделі...")
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

print("Початок навчання моделі...")
model_checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
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
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping]
)

print("Навчання завершено. Найкращу модель збережено у файл:", MODEL_SAVE_PATH)

print("\nОцінка найкращої збереженої моделі на валідаційних даних...")
best_model = load_model(MODEL_SAVE_PATH)

Y_pred = best_model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
y_pred_classes = np.argmax(Y_pred, axis=1)

y_true = validation_generator.classes

print('\nConfusion Matrix:')
print(confusion_matrix(y_true, y_pred_classes))

print('\nClassification Report:')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=target_names))


print("\nВідображення графіків навчання. Закрийте вікно, щоб завершити роботу.")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Графік точності (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Графік втрат (Loss)')

plt.show()
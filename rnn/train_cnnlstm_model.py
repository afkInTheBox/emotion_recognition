import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

IMG_WIDTH, IMG_HEIGHT = 48, 48
TRAIN_DATA_DIR = 'fer-2013/train'
VALIDATION_DATA_DIR = 'fer-2013/test'
BATCH_SIZE = 32
NUM_CLASSES = 7
EPOCHS = 50
MODEL_SAVE_PATH = "../Models/emotion_cnnlstm_model.keras"
TIMESTEPS = 1

def sequence_data_generator(generator):
    while True:
        x, y = next(generator)
        yield np.expand_dims(x, axis=1), y

print("Створення генераторів даних...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator_base = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical'
)
validation_generator_base = validation_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

train_sequence_generator = sequence_data_generator(train_generator_base)
validation_sequence_generator = sequence_data_generator(validation_generator_base)

print("Створення моделі CNN-LSTM за допомогою Sequential API...")

cnn_base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)
cnn_base.trainable = False

model = Sequential([
    Input(shape=(TIMESTEPS, IMG_WIDTH, IMG_HEIGHT, 3)),
    TimeDistributed(cnn_base),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(128),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

print("Початок навчання моделі...")
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    train_sequence_generator,
    steps_per_epoch=train_generator_base.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_sequence_generator,
    validation_steps=validation_generator_base.samples // BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping]
)

print("Навчання завершено. Найкращу модель збережено у файл:", MODEL_SAVE_PATH)

print("\nОцінка найкращої збереженої моделі на валідаційних даних...")
best_model = load_model(MODEL_SAVE_PATH)

Y_pred = best_model.predict(validation_sequence_generator, steps=validation_generator_base.samples // BATCH_SIZE + 1)
y_pred_classes = np.argmax(Y_pred, axis=1)

y_true = validation_generator_base.classes

print('\nConfusion Matrix:')
print(confusion_matrix(y_true, y_pred_classes))

print('\nClassification Report:')
target_names = list(validation_generator_base.class_indices.keys())
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
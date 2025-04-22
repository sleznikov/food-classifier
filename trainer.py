import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os

# Determine the environment and set up paths accordingly
try:
    # For Google Colab
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/food41_model'
    DATASET_PATH = '/content/drive/MyDrive/food41/images'  
except:
    # For Kaggle or local environment
    SAVE_DIR = './food41_model'
    DATASET_PATH = '/kaggle/input/food41/images'  # Kaggle dataset path

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 256
EPOCHS = 20

# GPU Optimization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = valid_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator

def train_model():
    train_generator, validation_generator = prepare_data()
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    TRAIN_STEPS = train_generator.samples // BATCH_SIZE
    VAL_STEPS = validation_generator.samples // BATCH_SIZE

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(SAVE_DIR, 'logs')
        )
    ]

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        callbacks=callbacks
    )
    return model, history, train_generator.class_indices

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def predict_image(model, image_path, class_indices):
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_name, confidence

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Starting training...")

    # Train the model
    model, history, class_indices = train_model()

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('food41_classifier_model.keras')  # Local save
    model.save(os.path.join(SAVE_DIR, 'food41_classifier_model.keras'))  # Save to specified directory

    # Test prediction on a sample image
    sample_image_path = os.path.join(
        DATASET_PATH,
        os.listdir(DATASET_PATH)[0],
        os.listdir(os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0]))[0]
    )
    predicted_class, confidence = predict_image(model, sample_image_path, class_indices)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

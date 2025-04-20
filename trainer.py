import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold

# Constants
IMG_SIZE = 380
BATCH_SIZE = 32  # Adjust as needed based on GPU memory
EPOCHS = 50
NUM_FOLDS = 5

# Path setup
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/food41_model'
    DATASET_PATH = '/content/drive/MyDrive/food41/images'
except:
    SAVE_DIR = './food41_model'
    DATASET_PATH = '/kaggle/input/food41/images'

os.makedirs(SAVE_DIR, exist_ok=True)

# GPU Setup - MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

def mixup(x, y, alpha=0.2):
    """Performs mixup on the input batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    
    return mixed_x, mixed_y

def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, initial_lr=0.001):
    """Implements cosine decay schedule with warmup"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs * initial_lr
    return initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

def create_model(num_classes, trainable_layers=0):
    """Creates an EfficientNetB4 based model with enhanced architecture"""
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Initially freeze all layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Unfreeze specified number of layers from the end
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Enhanced dense layers with BatchNormalization
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)

def prepare_data():
    """Prepares data with enhanced augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

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

def train_model_with_folds():
    train_generator, validation_generator = prepare_data()
    num_classes = len(train_generator.class_indices)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, 'best_model_{epoch:02d}_{val_accuracy:.3f}.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cosine_decay_with_warmup(epoch, EPOCHS)
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(SAVE_DIR, 'logs')
        )
    ]

    with strategy.scope():
        model = create_model(num_classes, trainable_layers=0)
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        history1 = model.fit(
            train_generator,
            epochs=EPOCHS // 2,
            validation_data=validation_generator,
            callbacks=callbacks
        )

        model = create_model(num_classes, trainable_layers=30) # Recreate for fine-tuning
        model.compile(
            optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        history2 = model.fit(
            train_generator,
            epochs=EPOCHS // 2,
            validation_data=validation_generator,
            callbacks=callbacks
        )

    return model, history1, history2

def plot_training_history(history1, history2):
    """Plots training history for both phases"""
    plt.figure(figsize=(15, 5))
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
    plt.show()

def predict_image(model, image_path, class_indices):
    """Makes prediction on a single image"""
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

    model, history1, history2 = train_model_with_folds()

    plot_training_history(history1, history2)

    model.save(os.path.join(SAVE_DIR, 'final_model.keras'))

    train_generator, _ = prepare_data()
    sample_image_path = os.path.join(
        DATASET_PATH,
        os.listdir(DATASET_PATH)[0],
        os.listdir(os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0]))[0]
    )
    predicted_class, confidence = predict_image(model, sample_image_path, train_generator.class_indices)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Added for ImageDataGenerator

def set_all_seeds(seed=42):
    """
    Sets seeds for reproducibility.
    """
    backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed}.")

def create_lr_reducer(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001):
    """
    Creates and returns a ReduceLROnPlateau callback.
    """
    return ReduceLROnPlateau(monitor=monitor,
                                patience=patience,
                                verbose=1,
                                factor=factor,
                                min_lr=min_lr)

def build_cnn_model1(input_shape=(224, 224, 3), num_classes=12):
    """
    Builds the basic CNN model (model1).
    """
    print("Building CNN Model 1...")
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("CNN Model 1 built.")
    model.summary()
    return model

def build_cnn_model2(input_shape=(224, 224, 3), num_classes=12):
    """
    Builds the improved CNN model (model2) with BatchNormalization.
    """
    print("Building CNN Model 2 with BatchNormalization...")
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding = 'same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    opt = Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("CNN Model 2 built.")
    model.summary()
    return model

def build_vgg16_model(input_shape=(224, 224, 3), num_classes=12, trainable_blocks=None):
    """
    Builds a VGG16-based model with a custom classification head.
    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        trainable_blocks (list or None): List of VGG16 block names to unfreeze for fine-tuning.
                                          If None, all VGG16 layers are frozen initially.
    Returns:
        tf.keras.Model: Compiled VGG16 model.
    """
    print(f"Setting up VGG16 model (trainable blocks: {trainable_blocks})...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    if trainable_blocks is None:
        # Freeze all layers of the base model initially
        for layer in base_model.layers:
            layer.trainable = False
    else:
        # Unfreeze specified blocks for fine-tuning
        for layer in base_model.layers:
            if any(layer.name.startswith(block) for block in trainable_blocks):
                layer.trainable = True
            else:
                layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print("VGG16 model built.")
    model.summary()
    return model

def build_vgg16_bn_aug_model(input_shape=(224, 224, 3), num_classes=12, trainable_blocks=None):
    """
    Builds a VGG16-based model with BatchNormalization in the head and optional fine-tuning.
    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        trainable_blocks (list or None): List of VGG16 block names to unfreeze for fine-tuning.
                                          If None, all VGG16 layers are frozen initially.
    Returns:
        tf.keras.Model: Compiled VGG16 model with BatchNormalization.
    """
    print(f"Setting up VGG16 model with BatchNormalization (trainable blocks: {trainable_blocks})...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    if trainable_blocks is None:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            if any(layer.name.startswith(block) for block in trainable_blocks):
                layer.trainable = True
            else:
                layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print("VGG16 model with BatchNormalization built.")
    model.summary()
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compiles the given Keras model with Adam optimizer and categorical crossentropy loss.
    """
    print(f"Compiling model with learning rate: {learning_rate}")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, callbacks=None, data_generator=None, steps_per_epoch=None):
    """
    Trains a given Keras model. Can accept an ImageDataGenerator for augmented training.
    """
    print("Starting model training...")
    if data_generator:
        print("Training with ImageDataGenerator...")
        # Use flow method to get batches from generator
        train_generator = data_generator.flow(X_train, y_train, batch_size=batch_size)
        if steps_per_epoch is None:
            steps_per_epoch = X_train.shape[0] // batch_size

        history = model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks
        )
    else:
        print("Training without ImageDataGenerator...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )
    print("Model training complete.")
    return history

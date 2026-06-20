import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def split_data(images, labels, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.
    Args:
        images (np.array): Processed image data.
        labels (pd.Series): DataFrame of labels.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split (from the remaining data after test split).
        random_state (int): Seed for random number generation for reproducibility.
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    print("Splitting data into training, validation, and test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_temp)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_labels(y_train, y_val, y_test):
    """
    Encodes target labels using LabelBinarizer.
    Args:
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Test labels.
    Returns:
        tuple: y_train_encoded, y_val_encoded, y_test_encoded, and the LabelBinarizer instance.
    """
    print("Encoding target labels...")
    enc = LabelBinarizer()
    y_train_encoded = enc.fit_transform(y_train)
    y_val_encoded = enc.transform(y_val)
    y_test_encoded = enc.transform(y_test)
    print(f"y_train_encoded shape: {y_train_encoded.shape}, y_val_encoded shape: {y_val_encoded.shape}, y_test_encoded shape: {y_test_encoded.shape}")
    return y_train_encoded, y_val_encoded, y_test_encoded, enc

def normalize_images(X_train, X_val, X_test):
    """
    Normalizes image pixel values to a 0-1 range.
    Args:
        X_train (np.array): Training image data.
        X_val (np.array): Validation image data.
        X_test (np.array): Test image data.
    Returns:
        tuple: X_train_normalized, X_val_normalized, X_test_normalized.
    """
    print("Normalizing image pixel values...")
    X_train_normalized = X_train.astype('float32') / 255.0
    X_val_normalized = X_val.astype('float32') / 255.0
    X_test_normalized = X_test.astype('float32') / 255.0
    print("Normalization complete.")
    return X_train_normalized, X_val_normalized, X_test_normalized

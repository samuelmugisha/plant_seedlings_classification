import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report

def evaluate_model_performance(model, X_test, y_test_encoded, enc, model_name):
    """
    Evaluates model performance and displays accuracy, classification report, and confusion matrix.
    Args:
        model (tf.keras.Model): The trained Keras model.
        X_test (np.array): Test image data.
        y_test_encoded (np.array): One-hot encoded true labels for test data.
        enc (sklearn.preprocessing.LabelBinarizer): LabelBinarizer instance for inverse transformation.
        model_name (str): Name of the model for display purposes.
    """
    print(f'\n--- Evaluating {model_name} ---\n')
    loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=2)
    print(f'{model_name} Test Accuracy: {accuracy:.4f}')
    print(f'{model_name} Test Loss: {loss:.4f}')

    y_pred = model.predict(X_test)
    y_pred_arg = np.argmax(y_pred, axis=1)
    y_test_arg = np.argmax(y_test_encoded, axis=1)

    # Classification Report
    print(f'\nClassification Report for {model_name}:\n')
    cr = classification_report(y_test_arg, y_pred_arg, target_names=enc.classes_)
    print(cr)

    # Confusion Matrix
    confusion_matrix_val = tf.math.confusion_matrix(y_test_arg, y_pred_arg)
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        confusion_matrix_val,
        annot=True,
        linewidths=.4,
        fmt="d",
        square=True,
        ax=ax
    )
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.xaxis.set_ticklabels(list(enc.classes_), rotation=40)
    ax.yaxis.set_ticklabels(list(enc.classes_), rotation=20)
    plt.show()

def plot_training_history(history, model_name):
    """
    Plots the training and validation accuracy from the model's history.
    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit.
        model_name (str): Name of the model for display purposes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.show()

def visualize_predictions(model, X_raw, X_normalized, y_test_encoded, enc, num_samples=4, model_name='Model'):
    """
    Visualizes sample predictions from the model.
    Args:
        model (tf.keras.Model): The trained Keras model.
        X_raw (np.array): Raw image data (for display).
        X_normalized (np.array): Normalized image data (for prediction).
        y_test_encoded (np.array): One-hot encoded true labels for test data.
        enc (sklearn.preprocessing.LabelBinarizer): LabelBinarizer instance.
        num_samples (int): Number of random samples to visualize.
        model_name (str): Name of the model for display purposes.
    """
    print(f"Visualizing predictions for {model_name}...")
    random_indices = np.random.choice(len(X_raw), num_samples, replace=False)

    for i, idx in enumerate(random_indices):
        plt.figure(figsize=(2, 2))
        plt.imshow(X_raw[idx]) # Display original image
        plt.title(f'Sample {idx}')
        plt.show()

        # Reshape for single prediction
        single_image = X_normalized[idx].reshape(1, *X_normalized.shape[1:])
        
        predicted_label_encoded = model.predict(single_image, verbose=0)
        predicted_label = enc.inverse_transform(predicted_label_encoded)
        true_label = enc.inverse_transform(y_test_encoded[idx].reshape(1, -1))

        print(f'Predicted Label: {predicted_label[0]}')
        print(f'True Label: {true_label[0]}')
        print('\n---\n')

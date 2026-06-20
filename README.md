# plant_seedlings_classification
A Convolutional Neural Network to classify plant seedlings into their respective categories.


# Plant Seedlings Classification

## Project Overview

This project aims to build and evaluate Convolutional Neural Networks (CNNs) for classifying plant seedlings into 12 distinct species. Accurate and efficient plant identification is crucial in agriculture to reduce manual labor, improve crop yields, and promote sustainable practices.

## Problem Statement

The agricultural sector faces significant challenges in manual plant and weed recognition, demanding extensive time and effort. This project seeks to leverage Artificial Intelligence and Deep Learning to automate and streamline the identification of plant seedlings, thereby enhancing efficiency and effectiveness beyond human capabilities.

## Objective

To develop a robust Convolutional Neural Network model capable of classifying images of plant seedlings into their respective categories.

## Data Description

The dataset, provided by the Aarhus University Signal Processing group and the University of Southern Denmark, comprises images of 12 unique plant species. The data is provided in two files:
- `images.npy`: Contains image data.
- `Labels.csv`: Contains the corresponding labels for each image.

**List of Species:**
- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common Wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet

## Methodology

The project involved the following key steps:

1.  **Data Preprocessing:**
    -   Image resizing from 128x128 to 224x224 pixels to accommodate transfer learning models.
    -   Conversion of images from BGR to RGB format.
    -   Splitting the dataset into training (80%), validation (10%), and test (10%) sets.
    -   One-hot encoding of target labels using `LabelBinarizer`.
    -   Normalization of pixel values to a 0-1 range.

2.  **Model Development & Evaluation:**
    -   **Model 1 (Basic CNN):** A custom sequential CNN model was built and trained. Performance was evaluated using accuracy, loss, and confusion matrices.
    -   **Model 2 (Improved CNN with Augmentation):** This model enhanced Model 1 by incorporating `BatchNormalization`, `ImageDataGenerator` for data augmentation (rotation, shifts, shear, zoom, horizontal flip), and `ReduceLROnPlateau` for learning rate scheduling.
    -   **VGG16 Transfer Learning:**
        -   **Initial VGG16 (Frozen Layers):** The VGG16 model, pre-trained on ImageNet, was used as a feature extractor by freezing its convolutional base and adding a custom classification head.
        -   **Fine-tuned VGG16 (Unfrozen Block 5):** The VGG16 model was further fine-tuned by unfreezing its last convolutional block (Block 5) and retraining with a very low learning rate.
        -   **Fine-tuned VGG16 with Batch Normalization and Data Augmentation:** This final iteration combined fine-tuning with Batch Normalization in the classification head and extensive data augmentation for robust performance.

## Results

| Model                                                        | Test Accuracy | Key Enhancements                                                                           |
| :----------------------------------------------------------- | :------------ | :----------------------------------------------------------------------------------------- |
| Model 1 (Basic CNN)                                          | 64.63%        | -                                                                                          |
| Model 2 (Improved CNN with Augmentation)                     | 70.74%        | Batch Normalization, ImageDataGenerator, ReduceLROnPlateau                                 |
| VGG16 (Frozen Layers)                                        | 62.74%        | Transfer Learning with VGG16 base                                                          |
| Fine-tuned VGG16 (Unfrozen Block 5)                          | 80.00%        | Fine-tuning (unfreezing Block 5) with low learning rate                                    |
| **Fine-tuned VGG16 (with BN and Augmentation)** (Final Model) | **83.00%**    | Fine-tuning, Batch Normalization in head, comprehensive Data Augmentation, LR Scheduling |

The **Fine-tuned VGG16 model with Batch Normalization and Data Augmentation** achieved the highest test accuracy of **83.00%**, demonstrating superior generalization and reduced overfitting.

## Actionable Insights

*   **Transfer Learning and Fine-tuning are Highly Effective:** The significant performance improvement observed with the VGG16 models, especially after fine-tuning and integrating data augmentation, highlights the power of these techniques for image classification tasks.
*   **Targeted Improvement Areas:** Despite the high overall accuracy, classes like 'Black-grass' and 'Shepherds Purse' still exhibit lower recall and f1-scores. This indicates persistent challenges in distinguishing these species, possibly due to visual similarities or data representation issues.
*   **Robustness Across Most Classes:** The final model demonstrates strong performance for the majority of classes, with many achieving F1-scores above 0.80, suggesting effective feature learning.

## Business Recommendations

1.  **Prioritize Deployment:** The top-performing Fine-tuned VGG16 model (83.00% accuracy) is ready for real-world deployment, promising substantial reductions in manual labor and increased efficiency in plant identification for well-performing species.
2.  **Data Enhancement for Challenging Classes:** To improve performance on 'Black-grass' and 'Shepherds Purse', consider acquiring more diverse data for these species, applying specialized data augmentation, or exploring alternative loss functions. A dedicated sub-model for these classes could also be explored.
3.  **Continuous Monitoring with Human Oversight:** Implement the model within a system that includes continuous performance monitoring. For predictions with lower confidence scores or those related to less accurate classes, integrate human expert review to ensure overall reliability.
4.  **Explore Further Architectures and Ensemble Methods:** Investigate other advanced CNN architectures (e.g., EfficientNet) or ensemble techniques to potentially achieve even higher robustness and accuracy.
5.  **Integration with Agricultural Systems:** Develop pathways to integrate this classification model with automated farming technologies, such as drones or robotic weeding systems, to facilitate real-time identification and intervention, thereby boosting agricultural productivity and sustainability.

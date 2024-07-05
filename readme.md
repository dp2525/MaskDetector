# Face Mask Detection using MobileNetV2

This repository contains the code and resources for training a face mask detection model using the MobileNetV2 architecture.

## Dataset

The dataset used for training the model consists of images of people wearing and not wearing face masks. It is a balanced dataset with annotations for mask/no-mask labels.

## Model Architecture

The MobileNetV2 architecture is used as the base model for this project. It is a lightweight and efficient convolutional neural network that performs well on mobile and embedded devices.

## Training

The model is trained using transfer learning, where the pre-trained MobileNetV2 model is fine-tuned on the face mask dataset. The training process involves optimizing the model's parameters using the Adam optimizer and minimizing the cross-entropy loss.

## Evaluation

The trained model is evaluated on a separate test set to measure its performance. The evaluation metrics include accuracy, precision, recall, and F1 score.

## Usage

To use the trained model for face mask detection, you can run the provided inference script. The script takes an input image as an argument and outputs the predicted labels for each detected face.

## Results

The trained model achieves an accuracy of over 95% on the test set. It performs well in detecting both masked and unmasked faces, with a low false positive rate.

## Future Work

In the future, we plan to improve the model's performance by collecting a larger and more diverse dataset. We also aim to explore other architectures and techniques to further enhance the accuracy and efficiency of the face mask detection system.



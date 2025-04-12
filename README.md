# Flower Image Classification

A deep learning project to classify different species of flowers using convolutional neural networks (CNN). This project utilizes a dataset of flower images and builds a model that can accurately predict the type of flower in a given image.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Dataset Structure](#dataset-structure)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## About the Project

This project aims to build an image classification system for recognizing various flower species using deep learning techniques. The main goal is to leverage a Convolutional Neural Network (CNN) for extracting features and classifying images into predefined categories.

## Dataset

- The dataset contains images of flowers from different classes such as roses, tulips, sunflowers, daisy, and dandelions.
- Source: [Insert dataset source or link](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)
- Number of classes: 5
- Number of images: ~3700

## Dataset Structure
This is the structure of the dataset used. Note that the there is a sample folder, created additionally.
![data_structure](other/flower_dataset_flow_chart.png)
## Technologies Used

- Python
- TensorFlow / Keras or PyTorch
- NumPy, Pandas, Matplotlib
- Jupyter Notebook / Google Colab
- Scikit-learn

## Model Architecture

- Input layer: Resized images to 128x128
- 3 Convolutional layers with ReLU and MaxPooling
- Fully connected Dense layers
- Softmax output for multi-class classification

## Results
### Evaluation Metrics
| Metric | Value % |
|--------|-------|
| Validation Loss | 74.35 |
| Validation Accuracy | 72.83 |
| Model Loss | 69.81 |
| Model Accuracy | 73.11 |

![output](other/model_evaluation.png)

### Output
This is how the output will be shown after running main.py file
![output](other/output.png)
## Future Work
- Use transfer learning with pretrained models (e.g., VGG16, ResNet50)
- Deploy as a web application
- Optimize model for mobile devices

## Acknowledgements
- TensorFlow Tutorials
- [Public Flower Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset)
- Kaggle Community
- ChatGPT
- Anaconda Distribution

NOTE:- THIS PROJECT WAS CREATED FOR ONLY PRACTICE AND A LOT OF TECHNIQUES AND ALGORITHMS ARE NOT YET USED. THANK YOU! :)

# Lettuce Malnutrition Detection using CNN

This project is a deep learning-based system for detecting malnutrition in lettuce plants caused by deficiencies in Nitrogen (N), Phosphorus (P), and Potassium (K). It utilizes a Convolutional Neural Network (CNN) model to classify images of lettuce leaves into four categories: Healthy, Nitrogen Deficiency, Phosphorus Deficiency, and Potassium Deficiency.

---

## Features
- **Image Classification**: Detects NPK deficiencies in lettuce leaves.
- **Dataset**: Organized into training, validation, and testing sets for better model performance.
- **Deep Learning Model**: Built using TensorFlow/Keras with Inception ResNet V2 for transfer learning.
- **GPU Acceleration**: Optimized for fast training using Google Colab Pro.

---

## Dataset Structure

The dataset is structured as follows:
```
Dataset/
    train/
        Healthy/
        N/
        P/
        K/
    valid/
        Healthy/
        N/
        P/
        K/
    test/
        Healthy/
        N/
        P/
        K/
```
Each folder contains images corresponding to its category.

---

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Google Colab Pro (recommended for GPU acceleration)
- Required Python libraries:
  - numpy
  - pandas
  - matplotlib
  - tensorflow
  - scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone this repository:
```bash
git clone https://github.com/orlyihan/CNN-Lettuce-NPK-Malnutrition-Detection.git
cd lettuce-malnutrition-detection
```

2. Prepare the dataset:
   - Upload the dataset to the `Dataset/` directory.
   - Ensure the dataset structure matches the one specified above.

3. Run the training script:
   - Open `train_model.ipynb` in Google Colab.
   - Ensure GPU is enabled in Colab runtime settings.
   - Execute the cells to train the model.

4. Test the model:
   - Use the `test_model.ipynb` notebook to evaluate the model's accuracy on the test dataset.

---

## Model

The model is based on **Inception ResNet V2**, a pre-trained architecture used for transfer learning. Fine-tuning is performed to adapt the model to classify lettuce deficiencies effectively.

---

## Results

The trained model achieves the following performance:
  precision    recall  f1-score   support

          FN       0.85      0.91      0.88        32
           K       0.96      0.93      0.95        85
           N       0.94      0.81      0.87        77
           P       0.58      0.90      0.70        21

    accuracy                           0.88       215
   macro avg       0.83      0.89      0.85       215
weighted avg       0.90      0.88      0.88       215
---

## Usage

1. Place images in the `test/` directory.
2. Run the notebook to classify new images.
3. View the results in the output folder.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes or improvements.

---


## Acknowledgments

- **Google Colab** for providing GPU resources.
- **TensorFlow/Keras** for simplifying the deep learning process.
- **Inception ResNet V2** for transfer learning architecture.


# Facial Expression Recognition with CNN

This project implements a Convolutional Neural Network (CNN) for facial expression recognition using grayscale images (48x48) and the FER-2013 dataset structure.

## 📁 Project Structure

- `deep_learning.py` – Preprocesses data, trains CNN, saves model and graphs.
- `test_model.py` – Loads the saved model and tests on random image from test set.
- `train/` – Folder containing training images (structured as class subfolders).
- `test/` – Folder containing testing images (structured the same way).
- `requirements.txt` - Python package dependencies
- `.gitignore` - Files/folders to ignore in Git

## 🧠 Model Architecture

- 3 Conv2D layers with ReLU activation
- MaxPooling2D + Dropout layers
- Fully connected Dense layers
- Final output layer with softmax activation

## 🧪 Evaluation

- Training and validation accuracy/loss graphs are generated using `matplotlib`.

## 🚀 Getting Started

1. Clone the repo:
   
   git clone https://github.com/YOUR_USERNAME/facial-recognition-cnn.git
   cd facial-recognition-cnn
   
2. Create a virtual environment and install dependencies:

   pip install -r requirements.txt

3. Run the training script:

   Run the training script:

4. Test the model:

   python test_model.py
   

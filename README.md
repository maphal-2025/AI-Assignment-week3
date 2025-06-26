# AI-Assignment-week3
Machine Learning Workbench
This repository presents a collection of applied machine learning and deep learning models across three major domains:

-Classical ML with Scikit-learn (Iris classification)
-Deep Learning with PyTorch (MNIST digit recognition using CNN)
-NLP with spaCy (Named Entity Recognition and simple sentiment analysis)
# Projects

-iris_classifier.py        # Scikit-learn Decision Tree on Iris Dataset
-mnist_cnn.py              # PyTorch CNN for MNIST classification
-spacy_sentiment.py        # spaCy NER + simple sentiment scoring

# 1. Iris Dataset Classification (Scikit-learn)
Goal: Predict the species of Iris flowers based on petal and sepal measurements.

Model: Decision Tree Classifier

Steps:
-Data cleaning and encoding
-Train-test split (80-20)
-Model training and evaluation (Accuracy, Precision, Recall)

# 2. MNIST Digit Classification (PyTorch CNN)
Goal: Classify handwritten digits (0–9) using a Convolutional Neural Network.

Model Architecture:

Conv2D → ReLU → Conv2D → ReLU → MaxPool → Dropout → Flatten → FC → Dropout → FC
-Training: 5 epochs, Adam optimizer, CrossEntropyLoss
-Evaluation: Reports test set accuracy
-Extras: Visualization of model predictions on sample digits

# 3. Named Entity Recognition + Sentiment (spaCy)
Goal: Analyze a product review to:
-Extract named entities (e.g. product names, organizations)
-Determine sentiment via simple rule-based scoring
Components:
-spaCy's en_core_web_sm model
-Custom word-based sentiment scoring using keyword matching

# Required libraries before running the code
pip install pandas scikit-learn torch torchvision matplotlib spacy
python -m spacy download en_core_web_sm


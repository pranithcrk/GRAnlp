# GRAnlp


# NLP Task README

This repository contains code and instructions for various NLP tasks, including binary text classification, multi-class text classification, ensemble modeling, and fine-tuning a BERT model.

## Files

- `train.py`: Python script containing functions to clean data, build, and train models for binary and multi-class text classification.
- `evaluate.py`: Python script to load and run multiple models as an ensemble, calculate precision, recall, F1-score, and save results locally.
- `bert_fine_tuning.py`: Python script for fine-tuning a BERT model, calculating precision, recall, F1-score, and saving results.
- `utils.py`: Utility functions used across the scripts.

## Instructions

### Binary Text Classification

1. **Data Preparation**: Load and preprocess the data using `train.py`. Use `create_binary_model` to train binary classification models for each label (A1, A2, A3, B1, B2, B3, B4).

   ```bash
   python train.py
Evaluation: Evaluate the ensemble of binary models and save results.


python evaluate.py
Multi-Class Text Classification
Data Preparation: Use train.py to split the data, preprocess text, and create a multi-class text classifier using the build_multi_class_model function.

python train.py --multiclass
Training: Train the multi-class model using the train_multi_class_model function.


python train.py --multiclass
Ensemble Modeling
Ensemble Evaluation: Load and run all 7 models as an ensemble, calculate precision, recall, F1-score, and save results locally.

python evaluate.py --ensemble
BERT Model Fine-Tuning
Fine-Tuning: Fine-tune a BERT model using the bert_fine_tuning.py script.

python bert_fine_tuning.py
Evaluation: Calculate precision, recall, F1-score, and save results for the fine-tuned BERT model.

python evaluate.py --bert
Dependencies
Python 3.7+
TensorFlow 2.x
NLTK
Scikit-learn
Transformers (for BERT fine-tuning)
Install dependencies using:


pip install -r requirements.txt
Conclusion
This repository provides a comprehensive set of scripts for NLP tasks, from building and training traditional models to fine-tuning advanced models like BERT.



This README template assumes that you have a `requirements.txt` file with the necessary dependencies for your project. Adjust the file paths, script names, and instructions based on your specific project structure and needs.

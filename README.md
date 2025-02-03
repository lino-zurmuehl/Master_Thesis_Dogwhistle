# Dog Whistle Detection with Confidence Learning (Ongoing)

This project enhances the detection of dog whistles—coded language with covert meanings—by cleaning mislabeled data and training a classifier using **BERT embeddings** and **Confident Learning**.

**Note: The data of this project is in part not published yet by Kruk et al. For this reason the 0 instances are not accessible through this repo.**

## Overview

Dog whistles are subtle and context-dependent, often evading conventional classifiers. Using the **Silent Signals** dataset (Kruk et al., 2024) and the `cleanlab` library (Northcutt et al., 2021), this project identifies mislabeled data and improves classification performance.

Key results:  
- **Original Model**: Accuracy = 67%, F1-Score (class 1) = 0.51  
- **Updated Model**: Accuracy = 83%, F1-Score (class 1) = 0.76  

## Methodology

1. **Dataset Cleaning**: Identify label errors using a logistic regression model trained on BERT embeddings.
2. **Model Training**: Train a logistic regression classifier with corrected labels.
3. **Evaluation**: Compare performance metrics before and after label correction.

## Future Plans
Finetuning RobBERTa Model for detection or Small Language Model from HF. 

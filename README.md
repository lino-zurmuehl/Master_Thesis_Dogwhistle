# Dog Whistle Detection with Fine-Tuned Language Models

**Note: The data of this project is in part not published yet by Kruk et al. For this reason the 0 instances are not accessible through this repo.**

This repository contains the code, data processing scripts, and evaluation tools used in the thesis project:

**"Reading Between the Lines: Fine-Tuning Language Models for Context-Aware Dog Whistle Detection"**

The project explores the use of RoBERTa and Gemma 2 models for identifying context-dependent coded language—commonly known as *dog whistles*—in political and online discourse.

---

## Project Structure

- **`finetuning_scripts/`**  
  Scripts and configurations for fine-tuning RoBERTa and Gemma 2 on the combined dog whistle dataset.  
  Includes hyperparameter settings, model checkpoints, and training logs.

- **`eval/`**  
  Evaluation scripts for computing classification metrics, generating confusion matrices, and ROC curves.  
  Also contains the external benchmark test set and evaluation pipeline used for out-of-distribution testing.

- **`preprocessing/`**  
  Scripts for cleaning, transforming, and labeling the raw data.  
  Includes confident learning checks, GPT-4-based disambiguation logic, and train/validation split scripts.

- **`plots/`**  
  All figures used in the thesis, including performance metrics, label distributions, and interpretability visualizations.  
  Useful for understanding dataset characteristics and model behavior.

---

# Acknowledgments
Dataset and initial annotations based on the work of [Kruk et al. (2024)].

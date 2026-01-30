## UCS654 Sampling Assignment
### Name: Khushveer Kaur
### Roll number: 102303327 

# Sampling Techniques on Imbalanced Credit Card Dataset

## Overview
This project demonstrates the importance of **sampling techniques** in handling **imbalanced datasets** and analyzes how different sampling strategies affect the performance of multiple Machine Learning models.

The dataset used represents **credit card transactions**, where fraudulent transactions are significantly fewer than normal ones. Such imbalance can lead to biased models, which is addressed using various sampling methods.

---

## Objective

- Convert an imbalanced dataset into a balanced one  
- Apply different sampling techniques  
- Train multiple ML models  
- Compare model performance under each sampling method  
- Identify the best sampling technique for each model  

---

## Dataset

The dataset contains credit card transactions labeled as:

- **Class 0** → Normal Transaction  
- **Class 1** → Fraud Transaction  

Initially, the dataset was highly imbalanced, with very few fraud cases.

---

## Sampling Techniques Used

| Sampling Name | Description |
|--------------|-------------|
| **Random Oversampling** | Increases minority class samples |
| **Random Undersampling** | Reduces majority class samples |
| **SMOTE** | Synthetic data generation for minority class |
| **Stratified Sampling** | Maintains class distribution |
| **Simple Random Sampling** | Random selection of data points |

---

## Machine Learning Models Used

| Model Code | Model Name |
|-----------|------------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors |
| M5 | Support Vector Machine |

---

## Experiment Procedure

1. Loaded dataset  
2. Checked class imbalance  
3. Applied **Random Oversampling** to balance data  
4. Created **5 random samples**  
5. Applied **5 sampling techniques**  
6. Trained **5 ML models**  
7. Evaluated performance using **Accuracy**  
8. Compared results  

---

## Results Table

### Accuracy table over different sampling techniques and ML models

<img width="1149" height="250" alt="Screenshot 2026-01-30 162942" src="https://github.com/user-attachments/assets/58ff4336-ef02-44aa-aef5-ab6b1ad5a4a7" />

### Best sampling technique for each ML model

<img width="395" height="209" alt="Screenshot 2026-01-30 162950" src="https://github.com/user-attachments/assets/2f1329c6-3567-4907-9ce1-1f84e4d3efce" />

### Graphical Representation 

<img width="880" height="611" alt="Screenshot 2026-01-30 163127" src="https://github.com/user-attachments/assets/a0b2e659-782e-4959-9c0a-8a29fc662c48" />

<img width="850" height="606" alt="Screenshot 2026-01-30 163137" src="https://github.com/user-attachments/assets/5c0cd8d5-bc24-4217-a412-c70466d0d3a1" />

## How to Run the Project

This project was developed and tested using **Google Colab**.

### Step 1: Open Notebook
1. Download the file **Sampling_Assignment(1).ipynb** from this repository.
2. Go to **https://colab.research.google.com**
3. Click **Upload Notebook**
4. Upload the downloaded `.ipynb` file.

---

### Step 2: Upload Dataset
1. In Colab, click the **folder icon** on the left sidebar.
2. Click **Upload**.
3. Upload the file **Creditcard_data.csv**.

---

### Step 3: Install Required Library
Run the first cell to install dependencies:

```python
!pip install imbalanced-learn

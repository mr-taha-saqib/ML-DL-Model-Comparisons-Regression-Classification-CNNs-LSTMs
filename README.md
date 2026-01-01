# Machine Learning & Deep Learning Model Comparisons

This repository contains a collection of Machine Learning and Deep Learning projects focused on **model comparison, performance evaluation, and practical insights** across regression, classification, image recognition, and time-series forecasting tasks.

Each project implements multiple models, compares their performance using appropriate metrics, and provides analytical reasoning behind the observed results.

---

## 📌 Projects Overview

### **1. House Price Prediction (Linear Regression vs MLP)**
- Built a baseline **Linear Regression** model with standardized features.
- Implemented a **Multi-Layer Perceptron (MLP)** with tuned hyperparameters.
- Evaluated models using **RMSE** and **R² score**.
- **Result:** MLP outperformed Linear Regression by capturing non-linear relationships.

---

### **2. Corporate Credit Rating Classification (Logistic Regression vs Neural Network)**
- Converted corporate credit ratings into a binary **investment-grade** classification.
- Trained **Logistic Regression** as a baseline and a **Neural Network** classifier.
- Evaluated using **accuracy, precision, recall, F1-score**, and confusion matrices.
- **Result:** Logistic Regression achieved perfect accuracy, indicating strong linear separability.

---

### **3. Tennis Play Prediction (Random Forest vs Naïve Bayes)**
- Preprocessed categorical data using imputation and encoding.
- Trained an optimized **Random Forest** using GridSearchCV.
- Compared against **Gaussian Naïve Bayes**.
- **Result:** Random Forest significantly outperformed Naïve Bayes.

---

### **4. Image Classification using CNN (Animal Recognition)**
- Built a **Convolutional Neural Network (CNN)** to classify animal images.
- Performed exploratory data analysis and dataset balancing analysis.
- Evaluated training, validation, and test performance.
- **Result:** Model showed overfitting, highlighting the need for data augmentation and transfer learning.

---

### **5. JPM Stock Price Prediction using Stacked LSTM**
- Downloaded historical stock data using **yfinance**.
- Built a **Stacked LSTM** network for time-series forecasting.
- Evaluated performance using **Root Mean Squared Error (RMSE)**.
- **Result:** Model successfully captured price trends with strong predictive performance.

---

## 🛠️ Technologies Used
- **Python**
- **NumPy, Pandas**
- **Scikit-learn**
- **TensorFlow / Keras**
- **Matplotlib, Seaborn**
- **yFinance**

---

## 📊 Evaluation Metrics
- RMSE
- R² Score
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

---

## 🚀 Key Learnings
- Simpler models can outperform complex ones when data is linearly separable.
- Neural networks excel in capturing non-linear patterns but require careful regularization.
- Ensemble models like Random Forest provide strong performance on categorical data.
- Deep learning models are prone to overfitting without sufficient data or augmentation.
- Time-series forecasting benefits from sequential models like LSTMs.

---

## 📌 Disclaimer
These projects are developed for **educational and academic purposes**.  
Stock price predictions are not financial advice.

---

## 👤 Author
**Taha Saqib**  
Data Scientist | Machine Learning & Deep Learning Enthusiast



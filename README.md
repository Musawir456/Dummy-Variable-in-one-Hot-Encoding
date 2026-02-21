# 🔢 Dummy Variables & One-Hot Encoding in Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A practical guide to converting categorical variables into numerical features for ML models.**

*Covers dummy variables, one-hot encoding, and the dummy variable trap — with real car price data.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why One-Hot Encoding?](#-why-one-hot-encoding)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Concepts Covered](#-concepts-covered)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🧠 Overview

Machine learning models only understand **numbers** — but real-world data is full of categories like car brands, cities, or fuel types. This project teaches you how to properly convert categorical data into numerical features using **Dummy Variables** and **One-Hot Encoding**, demonstrated on a real car prices dataset.

**What you'll learn:**
- What dummy variables are and why they matter
- How to apply `pd.get_dummies()` in pandas
- How to use `OneHotEncoder` from scikit-learn
- How to avoid the **Dummy Variable Trap**
- How encoded features improve ML model performance

---

## 💡 Why One-Hot Encoding?

| Raw Data | Problem | Solution |
|---|---|---|
| `"Toyota"`, `"Honda"`, `"BMW"` | ML models can't read text | Convert to 0s and 1s |
| Label Encoding (1, 2, 3) | Implies false ordering | One-Hot Encoding |
| Too many dummy columns | Multicollinearity | Drop one column (dummy trap) |

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Data Handling** | pandas, numpy |
| **Encoding** | `pd.get_dummies()`, `sklearn.OneHotEncoder` |
| **ML Model** | scikit-learn (Linear Regression) |
| **Environment** | Jupyter Notebook |

---

## 📂 Dataset

**File:** `carprices.csv`

A dataset of car prices with categorical features like car brand/model used to demonstrate encoding techniques.

| Column | Type | Description |
|---|---|---|
| `Car Model` | Categorical | Name/brand of the car |
| `Mileage` | Numerical | Mileage of the car |
| `Sell Price($)` | Numerical | Selling price of the car |
| `Age(yrs)` | Numerical | Age of the car in years |

---

## 🗂 Project Structure

```
Dummy-Variable-in-one-Hot-Encoding/
│
├── 📓 one_hot_encoding.ipynb              # Main notebook — dummy variables & encoding
├── 📓 Another_eg_one_hot_encoding.ipynb   # Additional example with different approach
├── 📊 carprices.csv                        # Car prices dataset
└── 📄 README.md                            # Project documentation
```

---

## 📚 Concepts Covered

### 1. 🏷️ Dummy Variables with Pandas
```python
import pandas as pd

df = pd.get_dummies(df, columns=['Car Model'])
# Converts each category into a separate binary column (0 or 1)
```

### 2. ⚠️ The Dummy Variable Trap
```python
# Drop one column to avoid multicollinearity
df = pd.get_dummies(df, columns=['Car Model'], drop_first=True)
```

### 3. 🤖 One-Hot Encoding with scikit-learn
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['Car Model']])
```

### 4. 📈 Training a Model on Encoded Features
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_encoded, y)
model.predict(X_new)
```

---

## ⚙️ How It Works

```
carprices.csv (with categorical columns)
          │
          ▼
┌──────────────────────────────┐
│   Identify Categorical Cols  │
│   e.g. "Car Model" column    │
└──────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│     Apply One-Hot Encoding   │
│  pd.get_dummies() OR         │
│  sklearn OneHotEncoder       │
└──────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Avoid Dummy Variable Trap   │
│  drop_first=True             │
└──────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Train Linear Regression     │
│  on fully numerical data     │
└──────────────────────────────┘
          │
          ▼
     Predict Car Prices 🚗
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Musawir456/Dummy-Variable-in-one-Hot-Encoding.git
cd Dummy-Variable-in-one-Hot-Encoding
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn jupyter
```

### 3. Run the Notebooks

```bash
jupyter notebook one_hot_encoding.ipynb
# OR
jupyter notebook Another_eg_one_hot_encoding.ipynb
```

---

## 👨‍💻 Author

<div align="center">

**Abdul Musawir**
*AI/Machine Learning Engineer & Data Science*
📍 Lahore, Pakistan
🎓 Superior University, Lahore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdul-musawir-a9713a20b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Musawir456)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/abmusawir)

</div>

---

<div align="center">

⭐ **Found this helpful? Give it a star!** ⭐

*Made with ❤️ by Abdul Musawir — Lahore, Pakistan*

</div>

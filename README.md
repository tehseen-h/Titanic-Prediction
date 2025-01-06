# 🛳️ Titanic Dataset Analysis and Prediction

This repository contains a comprehensive Python-based analysis and machine learning approach to the Titanic dataset. The project explores the dataset, performs Exploratory Data Analysis (EDA), handles missing values, and builds predictive models to estimate passenger survival. Below is a detailed explanation of the code structure and methodology.

---

## 📋 Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Part 1: Exploratory Data Analysis (EDA)](#part-1-exploratory-data-analysis-eda)
3. [Part 2: Machine Learning for Filling Missing Values](#part-2-machine-learning-for-filling-missing-values)
4. [Model Building and Evaluation](#model-building-and-evaluation)
5. [Installation and Usage](#installation-and-usage)
6. [Dataset](#dataset)
7. [Results](#results)

---

## 📊 Dataset Overview

The dataset used in this project is the Titanic dataset, which provides information about passengers aboard the Titanic, such as age, gender, fare, class, and survival status. The dataset contains missing values, particularly in the `Age` column, which are addressed using two approaches.

---

## 🔍 Part 1: Exploratory Data Analysis (EDA)

### Goals of EDA:

1. Understand the distribution of data.
2. Visualize survival rates based on key features such as age and gender.
3. Identify and handle missing values in the dataset.

### Key Steps:

1. **Data Loading**: The dataset is loaded using `pandas`.
2. **Initial Inspection**:
   - Summary statistics using `.info()` and `.describe()`.
   - Count and distribution of survival (`Survived` column).
3. **Visualizations**:
   - Distribution of age by survival status using `seaborn.histplot`.
   - Bar plot of survival rates by gender using `seaborn.barplot`.
4. **Handling Missing Values**:
   - Missing values in the `Age` column were filled with the median value initially.
   - Missing values in the `Embarked` column were filled using the mode.

---

## 🤖 Part 2: Machine Learning for Filling Missing Values

### Objective:

To use a machine learning model to predict and fill missing values in the `Age` column instead of relying on statistical measures such as mean or median.

### Steps:

1. Splitting the data into:
   - **Train Data**: Rows with non-null `Age` values.
   - **Test Data**: Rows with null `Age` values.
2. **Feature Selection**:
   - Features used for prediction: `Pclass`, `Fare`, `SibSp`, `Parch`, and `Sex`.
3. **Model Training**:
   - A `RandomForestRegressor` model was trained on the train data.
   - Missing `Age` values were predicted and updated in the dataset.
4. **Visualization**:
   - The age distribution after filling missing values was visualized to confirm a natural-looking distribution.

---

## 🛠️ Model Building and Evaluation

### Models Used:

1. **Logistic Regression**:
   - Achieved a score of ~77.7%.
2. **Random Forest Classifier**:
   - Outperformed logistic regression with a score of ~83.2%.

### Steps:

1. Features such as `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, and `Sex` were used for training.
2. Data was split into training and testing sets using an 80-20 ratio.
3. Two models were trained and evaluated, with Random Forest being selected for final predictions due to its superior performance.

---

## 💻 Installation and Usage

### Prerequisites:

- Python 3.x
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `joblib`

### Steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/titanic-analysis.git

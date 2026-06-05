# Insurance Cost Prediction

## Project Overview

Insurance Cost Prediction is a machine learning regression project that predicts medical insurance charges based on customer demographic and lifestyle information.

The project uses a medical insurance dataset containing features such as age, sex, BMI, number of children, smoking status, and region. The target variable is `charges`, which represents the medical insurance cost charged to each customer.

The main purpose of this project is to explore the dataset, clean and prepare the data, build multiple regression models, compare their performance, and create a simple Streamlit web application for making predictions.

---

## Problem Statement

Medical insurance charges can vary based on several personal, demographic, and health-related factors. Insurance companies and analysts can use machine learning models to estimate expected insurance costs based on customer information.

This project aims to answer the question:

**Can we predict a customer's medical insurance charge using demographic and lifestyle features?**

Since the target variable `charges` is a continuous numerical value, this is a **regression problem**.

---

## Dataset

* **Source:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance) — originally from Brett Lantz's *Machine Learning with R*
* **Samples:** 1,338 records
* **Features:** 6 input features + 1 numeric target
* **Target variable:** `charges`

---

## Dataset Description

The dataset contains medical insurance customer records.

| Column     | Description                                        |
| ---------- | -------------------------------------------------- |
| `age`      | Age of the insurance customer                      |
| `sex`      | Gender of the customer                             |
| `bmi`      | Body Mass Index                                    |
| `children` | Number of children/dependents covered by insurance |
| `smoker`   | Whether the customer is a smoker or non-smoker     |
| `region`   | Residential region of the customer                 |
| `charges`  | Medical insurance cost charged to the customer     |

---

## Project Objectives

The main objectives of this project are:

* Perform exploratory data analysis
* Understand relationships between features and insurance charges
* Identify important predictors of insurance cost
* Clean the dataset and remove duplicate records
* Engineer useful new features
* Encode categorical variables for machine learning
* Train and evaluate regression models
* Compare model performance using suitable metrics
* Tune the best-performing model
* Analyze feature importance
* Build a simple Streamlit web app for prediction

---

## Project Workflow

The project follows a complete machine learning workflow:

1. Exploratory Data Analysis
2. Data Cleaning and Feature Engineering
3. Model Building
4. Model Evaluation and Comparison
5. Cross-Validation
6. Hyperparameter Tuning
7. Feature Importance Analysis
8. Streamlit Web App Development

---

## Exploratory Data Analysis

The EDA notebook analyzes the dataset to understand patterns and relationships before model building.

Main EDA tasks include:

* Dataset overview
* Missing value and duplicate record checking
* Target variable distribution
* Numerical feature analysis
* Categorical feature analysis
* Bivariate analysis with insurance charges
* Smoker and BMI interaction analysis
* Correlation analysis
* Pair plot analysis
* Key EDA findings

### Key EDA Findings

* Smoking status is the strongest predictor of insurance charges.
* Smokers have significantly higher charges than non-smokers.
* Customers who are both smokers and obese have the highest average charges.
* Age has a positive relationship with insurance charges.
* BMI has a weak positive relationship with charges, but its effect becomes stronger when combined with smoking status.
* The target variable `charges` is right-skewed.
* The dataset has no missing values.
* One duplicate row was identified and removed during data cleaning.

---

## Data Cleaning and Feature Engineering

The data cleaning notebook prepares the dataset for machine learning.

### Cleaning Steps

* Checked missing values
* Checked duplicate rows
* Removed one duplicate record
* Verified cleaned dataset shape
* Saved the cleaned dataset

### Feature Engineering

New features were created to improve model performance and interpretability.

| Feature        | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `bmi_category` | Groups BMI values into categories such as normal, overweight, and obese |
| `age_group`    | Groups customers into young, middle, and senior age groups              |
| `smoker_obese` | Identifies customers who are both smokers and obese                     |

The `smoker_obese` feature was added because the EDA showed that smokers with high BMI values usually have much higher insurance charges.

### Encoding

Categorical variables were converted into numerical format using one-hot encoding.

Encoded features include:

* `sex`
* `smoker`
* `region`
* `bmi_category`
* `age_group`

The final processed dataset was saved as:

```text
data/insurance_cleaned.csv
```

---

## Machine Learning Models

The following regression models were trained and evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. KNN Regressor

### Model Descriptions

| Model                       | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| Linear Regression           | Simple baseline regression model                      |
| Decision Tree Regressor     | Tree-based model that captures non-linear patterns    |
| Random Forest Regressor     | Ensemble model that combines multiple decision trees  |
| Gradient Boosting Regressor | Boosting model that improves predictions step by step |
| KNN Regressor               | Distance-based model using nearest neighbors          |

---

## Model Evaluation Metrics

The models were evaluated using the following regression metrics:

| Metric   | Meaning                                                                    |
| -------- | -------------------------------------------------------------------------- |
| R² Score | Measures how much variation in insurance charges is explained by the model |
| RMSE     | Root Mean Squared Error                                                    |
| MAE      | Mean Absolute Error                                                        |
| MAPE     | Mean Absolute Percentage Error                                             |

A good model should have:

* Higher R² score
* Lower RMSE
* Lower MAE
* Lower MAPE

---

## Model Performance Summary

The model comparison showed that Gradient Boosting and Linear Regression performed strongly on this dataset.

Tree-based ensemble models such as Random Forest and Gradient Boosting were useful because they can capture non-linear relationships and feature interactions.

Gradient Boosting was also tuned using GridSearchCV to check whether the model performance could be improved.

---

## Cross-Validation

5-fold cross-validation was used to check model stability.

Cross-validation helps evaluate whether the model performs consistently across different subsets of the training data.

The models were compared using mean cross-validation R² scores and standard deviation values.

A reliable model should have:

* High mean cross-validation R² score
* Low standard deviation across folds

---

## Hyperparameter Tuning

Gradient Boosting Regressor was tuned using GridSearchCV.

The following hyperparameters were tested:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_samples_leaf`

The goal of tuning was to find the best parameter combination based on cross-validation R² score.

---

## Feature Importance

Feature importance analysis was performed using tree-based models.

The most important features included:

* `smoker_obese`
* `smoker_yes`
* `age`
* `bmi`
* `children`

The results confirmed that smoking status and obesity-related features have a strong impact on medical insurance charges.

---

## Streamlit Web App

A simple Streamlit web application was created to test the model locally.

The app allows users to enter customer details such as:

* Age
* Sex
* BMI
* Number of children
* Smoker status
* Region

The app then predicts the estimated medical insurance charge.

---

## Project Structure

```text
insurance-cost-prediction/
│
├── app.py
│
├── data/
│   ├── insurance.csv
│   └── insurance_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_building.ipynb
│
├── src/
│   └── utils.py
│
├── models/
│
├── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## File Descriptions

| File / Folder                       | Description                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| `app.py`                            | Streamlit web application for insurance cost prediction                       |
| `data/insurance.csv`                | Original raw dataset                                                          |
| `data/insurance_cleaned.csv`        | Cleaned and processed dataset                                                 |
| `notebooks/01_eda.ipynb`            | Exploratory data analysis notebook                                            |
| `notebooks/02_data_cleaning.ipynb`  | Data cleaning and feature engineering notebook                                |
| `notebooks/03_model_building.ipynb` | Model training, evaluation, cross-validation, tuning, and prediction notebook |
| `src/utils.py`                      | Helper functions for loading data, preprocessing, evaluation, and plotting    |
| `models/`                           | Folder reserved for saved trained models                                      |
| `outputs/`                          | Folder reserved for output files, charts, or reports                          |
| `requirements.txt`                  | Required Python libraries                                                     |
| `.gitignore`                        | Files and folders ignored by Git                                              |
| `README.md`                         | Project documentation                                                         |

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* GitHub
* VS Code

---

## How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Hirun-Rajapaksha/credit-card-fraud-detection
```

```bash
cd insurance-cost-prediction
```

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

### 3. Activate the Virtual Environment

For Windows PowerShell:

```bash
.venv\Scripts\activate
```

---

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 5. Run Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebooks in this order:

```text
notebooks/01_eda.ipynb
notebooks/02_data_cleaning.ipynb
notebooks/03_model_building.ipynb
```

---

### 6. Run the Streamlit Web App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

Usually it will be:

```text
http://localhost:8501
```

---

## Requirements

The required libraries are listed in `requirements.txt`.

```text
jupyter
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

---

## Example Prediction Inputs

### Example 1: Non-Smoker

```text
Age: 30
Sex: male
BMI: 28
Children: 1
Smoker: no
Region: southeast
```

This example is expected to produce a lower insurance cost.

### Example 2: Smoker with High BMI

```text
Age: 45
Sex: male
BMI: 35
Children: 2
Smoker: yes
Region: southeast
```

This example is expected to produce a higher insurance cost.

---

## Results and Insights

The project found that:

* Smoking status is the most influential factor in predicting insurance charges.
* Customers who are both smokers and obese usually have the highest charges.
* Age is also an important predictor.
* BMI alone has a weaker effect, but becomes more important when combined with smoking status.
* Gradient Boosting and Linear Regression produced strong model performance.
* Feature engineering improved the interpretability of the model.

---

## Future Improvements

Possible future improvements include:

* Save the trained model as a `.pkl` or `.joblib` file
* Load the saved model in the Streamlit app instead of training inside the app
* Try advanced models such as XGBoost or LightGBM
* Add more user-friendly design to the Streamlit app
* Deploy the app using Streamlit Cloud
* Add more input validation
* Add model explanation using SHAP or feature contribution methods
* Add prediction confidence or error range

---

## Conclusion

This project successfully demonstrates the full machine learning workflow for a regression problem.

It includes data exploration, data cleaning, feature engineering, model training, model evaluation, cross-validation, hyperparameter tuning, feature importance analysis, and web app development.

The final result is a complete beginner-to-intermediate level machine learning project that can be included in a data science portfolio.

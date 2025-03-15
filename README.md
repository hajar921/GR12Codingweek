# 🏥 Obesity Prediction Using Machine Learning


## Project Description

This project aims to develop an advanced clinical decision support tool that enables doctors to accurately estimate the risk of obesity in their patients based on their lifestyle and physical condition. The application is powered by a machine learning model with explainability provided through SHAP (SHapley Additive exPlanations).

## Project Objectives

- Develop a robust and explainable machine learning model.
- Ensure transparency in predictions using SHAP explainability.
- Design an intuitive user interface with Streamlit or Flask.
- Apply best software development practices (GitHub, CI/CD).
- Document the prompts used for AI prompt engineering.

## Dataset Used

We use the following dataset:\
[Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

## Preliminary Data Analysis

In the file notebooks/eda.ipynb, we conducted the following analysis:

- Missing Values: Identification and strategies for handling missing values.
- Outliers: Detection and treatment of outliers.
- Class Imbalance:
  - The dataset contains seven obesity levels, with class distributions ranging between 12.88% and 16.63%, making it relatively balanced.
  However, to ensure all classes had sufficient representation, SMOTE (Synthetic Minority Over-sampling Technique) was applied.
Impact: SMOTE improved recall for minority classes without significantly affecting overall accuracy.
  - Documentation of strategies to handle imbalance (oversampling, undersampling, class weighting).
- Correlation Between Variables: Identification and treatment of highly correlated variables.
- Imbalanced data was addressed using SMOTE (Synthetic Minority Over-sampling Technique) to improve the model’s ability to detect fraud.
- Memory Optimization: Unnecessary columns were dropped, data types were optimized, and categorical values were encoded efficiently to reduce memory usage and improve training speed. 



## Machine Learning Models

We tested and compared the performance of the following models:

- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

### Evaluation Metrics
Best Performing Model:

The Random Forest Classifier achieved the highest accuracy (86.29%), followed by XGBoost (86.1%) and LightGBM (85.3%).
Justification:
Random Forest provided the best balance between performance and generalization.
Despite requiring more memory, it handled imbalanced data better and had stronger predictive accuracy.
Models are compared using:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Accuracy

### LightGBM: 
Achieved an accuracy of 85.3%, but struggled with imbalanced data and had higher memory efficiency.

### XGBClassifier : 
Performed slightly better at 86.1% accuracy but had overfitting issues and moderate memory consumption.

### Random Forest Classifier:
 Provided the best results with an accuracy of 86.29%, demonstrating strong generalization capabilities, but required higher computational resources due to multiple decision trees.

### Justification:
The Random Forest Classifier was chosen as the final model due to its superior performance in terms of accuracy and its robustness in handling imbalanced datasets. While it requires more memory compared to Logistic Regression, its predictive power makes it the best choice for fraud detection.

## Memory Optimization

To optimize memory usage, a function optimize_memory(df) has been developed in data_processing.py. This function adjusts data types (float64 → float32, int64 → int32) to improve processing efficiency.
 As a result, memory usage was reduced from 1.2 MB to 87.9 KB, significantly enhancing efficiency without compromising accuracy.

## Explainability with SHAP
Top Medical Features Affecting Predictions (SHAP Analysis):

Weight & BMI (strongest predictor)
Caloric intake & eating frequency
Physical activity levels
Family history of obesity
Water consumption (CH2O variable)
We integrate SHAP explanations to interpret the model:

- Generation of SHAP visualizations.
- Visualization of feature impacts on predictions.

## Interface Development (Streamlit/Flask)

The interface allows:

- Input of patient data (diet, physical activity, etc.).
- Display of obesity level predictions.
- Access to SHAP explanations.

## GitHub Workflow & CI/CD

- Code is organized in a well-structured GitHub repository.
- A CI/CD workflow is implemented via GitHub Actions to automate testing and continuous integration.

## Selected Task for Documentation: Memory Optimization Function
Insights from AI-Powered Prompt Engineering
Initially, we wrote a simple function to convert float64 → float32 and int64 → int32, but it lacked:

Logging to show before/after memory usage.
Handling for categorical variables and missing values.
Flexibility in precision levels for conversions.
To refine it, we used ChatGPT & Copilot, iterating with different prompts.
### Prompts Used

We utilized AI tools (Copilot, ChatGPT) to generate and refine the memory optimization function. Below are the exact prompts used:

1. *Initial Prompt:*\
   "Generate a Python function to optimize memory usage in a Pandas DataFrame by converting large data types (float64, int64) to smaller ones. Ensure it preserves numerical precision."

   *Result:*\
   A basic function that converts float64 → float32 and int64 → int32, but lacked logging and exception handling.

2. *Refinement Prompt:*\
   "Improve the function by adding logging to display memory savings, handling missing values properly, and ensuring compatibility with categorical variables."

   *Result:*\
   A more robust function with logging and better error handling, preserving categorical variables.

How AI Helped Improve the Function:

Better Type Handling: ChatGPT suggested converting categorical variables properly.
Logging: AI recommended adding before/after memory tracking.
Precision Control: ChatGPT provided ideas for precision-based conversions.
Error Handling: AI ensured missing values wouldn't cause crashes.
### Effectiveness and Potential Improvements

- The prompts effectively generated a useful function but required refinements for better usability.
- Future improvements:
  - Automatically detect and convert datetime columns.
  - Provide a summary report of before/after memory usage.
  - Allow user-defined precision levels for conversions.

By iterating on prompts, we enhanced the function’s efficiency, usability, and transparency.

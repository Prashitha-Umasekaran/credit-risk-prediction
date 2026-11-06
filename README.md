# Credit Risk Prediction

## Project Overview
This project explores credit risk assessment by analysing borrower-related data and identifying patterns associated with potential loan defaults.

## Objectives
- Understand key factors associated with credit risk.
- Analyse borrower information to identify risk patterns.
- Explore machine learning classification models to analyse loan delinquency patterns and evaluate their predictive performance.
- Understand how predictive insights can support lending decisions.

## Tools & Technologies
- Python
- Pandas
- Scikit-learn
- Data Analysis
- Machine Learning

## Requirements

The project uses the following Python libraries:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

## Business Relevance

This project demonstrates how data analytics can support financial risk management and lending decisions.

Key business applications include:

- **Borrower Risk Identification:** Identify patterns associated with borrowers who may have a higher likelihood of default.
- **Lending Decision Support:** Provide data-driven insights to support loan approval and credit assessment processes.
- **Risk Exposure Management:** Help financial institutions understand potential default risks and improve risk monitoring.
- **Data-Driven Decision Making:** Translate borrower data and predictive model outputs into actionable business insights.
- **Model Interpretability:** Examine influential borrower-related features to understand the factors contributing to predicted credit risk.

## Model Results

The project includes the following model evaluation and analysis outputs:

- **Class Distribution:** Examined the original dataset to understand class imbalance.
- **SMOTE Analysis:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance in the training data.
- **Confusion Matrix:** Evaluated the model's classification performance by examining correct and incorrect predictions.
- **Feature Importance:** Identified the features that contributed to the model's predictions.

These results provide insights into model behaviour and demonstrate how predictive analytics can support credit risk assessment.

## Limitations & Future Improvements

### Limitations
- Model performance depends on the quality and representativeness of the dataset.
- Class imbalance can affect the reliability of predictions.
- Feature importance indicates model relationships but does not establish causation.
- Predictions should support, not replace, financial risk assessment and human decision-making.

### Future Improvements
- Evaluate additional classification algorithms.
- Compare model performance using precision, recall, F1-score and ROC-AUC.
- Explore explainable AI techniques to improve model transparency.
- Develop an interactive credit risk monitoring dashboard.

## Project Repository
This project demonstrates the application of data analytics and machine learning to a practical financial risk management problem.

## Project Visualizations

### 1. Original Class Distribution
![Original Class Distribution](./visuals/class_distribution.png)

### 2. Class Distribution After SMOTE
![Class Distribution After SMOTE](./visuals/smote_distribution.png)

### 3. Confusion Matrix
![Confusion Matrix](./visuals/confusion_matrix.png)

### 4. Feature Importance
![Feature Importance](./visuals/feature_importance.png)

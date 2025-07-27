# MSCS_634_Lab_4 - Multiple Regression and Regularization

## Overview
This lab explores various regression techniques on the Diabetes dataset from scikit-learn, implementing and comparing simple linear regression, multiple regression, polynomial regression, and regularization methods (Ridge and Lasso). The goal is to understand model implementation, performance evaluation, and the impact of regularization on preventing overfitting.

## Dataset
- **Source**: Diabetes dataset from `sklearn.datasets`
- **Target Variable**: Disease progression (continuous)
- **Features**: 10 health indicators including BMI, blood pressure, age, sex, etc.
- **Sample Size**: 442 observations
- **Data Split**: 80% training, 20% testing with random_state=42

## Implementation Steps

### Step 1: Data Preparation
- Loaded the Diabetes dataset using `sklearn.datasets.load_diabetes()`
- Converted to pandas DataFrame for features and Series for target variable
- Displayed descriptive statistics to understand data distributions

### Step 2: Simple Linear Regression
- **Feature Used**: BMI only
- **Model**: `LinearRegression` from scikit-learn
- **Performance Metrics**:
  - MAE: 52.26
  - MSE: 4061.83
  - RMSE: 63.73
  - R²: 0.23
- **Visualization**: Scatter plot with regression line showing BMI vs. disease progression

### Step 3: Multiple Regression
- **Features Used**: All 10 features
- **Model**: `LinearRegression` with full feature set
- **Performance Metrics**:
  - MAE: 42.79
  - MSE: 2900.19
  - RMSE: 53.85
  - R²: 0.45
- **Visualization**: Actual vs. predicted values scatter plot

### Step 4: Polynomial Regression
- **Feature Used**: BMI with polynomial transformations
- **Degrees Tested**: 2, 3, and 4
- **Performance Results**:
  - Degree 2: MSE: 4085.03, R²: 0.23
  - Degree 3: MSE: 4064.44, R²: 0.23
  - Degree 4: MSE: 4226.14, R²: 0.20

### Step 5: Regularization Methods
- **Ridge Regression** (α=1.0):
  - MSE: 3077.42
  - R²: 0.42
- **Lasso Regression** (α=0.1):
  - MSE: 2798.19
  - R²: 0.47

## Key Findings and Insights

### Model Performance Comparison
1. **Simple Linear Regression**: Provides baseline performance with limited predictive power due to using only one feature
2. **Multiple Linear Regression**: Significantly improves performance by leveraging all available features
3. **Polynomial Regression**: Shows minimal improvement over simple linear, with higher degrees potentially leading to overfitting
4. **Regularization Methods**: 
   - Ridge regression provides stable performance with controlled coefficient magnitudes
   - Lasso regression achieves the best R² score while potentially performing feature selection

### Technical Insights
- **Feature Importance**: Multiple features significantly improve predictive power over single-feature models
- **Overfitting Prevention**: Regularization methods help control model complexity and improve generalization
- **Non-linear Relationships**: Limited evidence of strong non-linear patterns in the dataset
- **Model Selection**: Lasso regression with α=0.1 provides the best balance of performance and interpretability

### Challenges and Decisions
- **Data Splitting**: Consistent 80/20 train-test split with fixed random_state ensures reproducible results
- **Hyperparameter Tuning**: Alpha values for Ridge (1.0) and Lasso (0.1) were selected based on experimentation
- **Polynomial Degree Selection**: Tested degrees 2-4 to observe overfitting patterns
- **Evaluation Metrics**: Used comprehensive metrics (MAE, MSE, RMSE, R²) for thorough model comparison

## Technologies Used
- **Python Libraries**: scikit-learn, pandas, numpy, matplotlib
- **Models**: LinearRegression, Ridge, Lasso, PolynomialFeatures
- **Evaluation**: mean_absolute_error, mean_squared_error, r2_score
- **Visualization**: matplotlib for regression plots and performance comparisons

## Files
- `lab4.ipynb`: Jupyter notebook containing the complete implementation
- `README.md`: This documentation file

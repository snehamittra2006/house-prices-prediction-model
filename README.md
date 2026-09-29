# House Price Prediction

A machine learning project that predicts house prices using property characteristics. The notebook covers exploratory data analysis, data preprocessing, regression model training, evaluation, and prediction for a new house.

## Project Overview

House prices depend on multiple factors, including the property's area, number of bedrooms and bathrooms, number of stories, parking availability, amenities, and furnishing status.

This project uses the **Housing dataset** to explore these relationships and train regression models to estimate house prices.

## Objectives

- Explore and visualize the housing dataset.
- Inspect missing values, duplicate records, and feature distributions.
- Prepare numerical and categorical features for machine learning.
- Train and compare regression models.
- Evaluate predictions using regression metrics and visualizations.
- Generate a price prediction for a new house using the trained model.

## Dataset

The notebook loads a file named `Housing.csv`.

The dataset contains the following input features:

| Feature | Description |
|---|---|
| `area` | Area of the property |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the property is connected to the main road |
| `guestroom` | Whether a guestroom is available |
| `basement` | Whether the property has a basement |
| `hotwaterheating` | Whether hot-water heating is available |
| `airconditioning` | Whether air conditioning is available |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the property is in a preferred area |
| `furnishingstatus` | Furnishing category of the property |

**Target variable:** `price`

The notebook checks for missing values and duplicate rows and explores numerical and categorical feature distributions. Confirm the dataset's source and licensing before redistributing it.

## Technologies and Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The notebook uses descriptive statistics and visualizations to understand the data:

- Dataset shape, column names, data types, and summary statistics
- Missing-value and duplicate-row checks
- Mean, median, standard deviation, skewness, and kurtosis for numerical columns
- Frequency distributions for categorical columns
- Histograms and boxplots to inspect distributions and potential outliers
- Count plots for categorical features
- Correlation heatmap for numerical variables
- Scatter plot of house price against area, grouped by air-conditioning availability

The notebook's EDA notes that `area` and `bathrooms` have relatively strong positive correlations with `price` in the dataset.

### 2. Data Preprocessing

The target variable is transformed using the natural logarithm:

```python
y = np.log1p(df["price"])
```

The data is split into training and testing sets using an 80:20 split and `random_state=42`.

Categorical columns are encoded using `OneHotEncoder` with `drop="first"` and `handle_unknown="ignore"`. Numerical columns are passed through without scaling, using a `ColumnTransformer`.

The encoder is fitted on the training data only, then applied to the test data. This helps prevent information from the test set leaking into preprocessing.

### 3. Model Training

The notebook trains and compares these regression models:

| Model | Configuration |
|---|---|
| Linear Regression | Scikit-learn `LinearRegression` |
| Ridge Regression | `Ridge(alpha=1.0)` |
| Random Forest Regressor | 100 trees, `random_state=42` |

Each model is trained using the transformed training features and the log-transformed target.

### 4. Model Evaluation

Predictions are converted back to the original price scale using `np.expm1` and evaluated with:

| Metric | Meaning |
|---|---|
| RMSE | Root Mean Squared Error; penalizes larger errors more heavily |
| MAE | Mean Absolute Error; average absolute prediction error |
| R² | Measures the proportion of target variance explained by the model |
| Accuracy within 10%, 15%, and 20% | Percentage of predictions whose absolute percentage error falls within the specified threshold |

The notebook displays a side-by-side comparison table and plots actual versus predicted prices, residual distributions, and metric comparisons.

### 5. Feature Analysis

The notebook retrieves transformed feature names from the fitted `ColumnTransformer` and includes a feature-analysis section for model coefficients or tree-based feature importances.

**Note:** In the current notebook, `best_model` is explicitly assigned to the Linear Regression model. Therefore, the feature-analysis cell uses Linear Regression coefficients rather than Random Forest feature importances unless `best_model` is changed.

### 6. Predicting a New House Price

A sample house record is created with the required input columns. The fitted transformer converts it into the same feature representation used during training, and the trained model predicts its log price. `np.expm1` converts the prediction back to the original price scale.

The sample input values in the notebook are illustrative. Replace them with valid property details to estimate a different house price.

## Installation

1. Clone or download this repository.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

2. Install the required libraries.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

3. Place `Housing.csv` in the location expected by the notebook, or update the dataset path.

The notebook currently reads the dataset from:

```python
df = pd.read_csv("/content/Housing.csv")
```

For local execution, change this to the appropriate path, for example:

```python
df = pd.read_csv("Housing.csv")
```

4. Launch Jupyter Notebook.

```bash
jupyter notebook
```

5. Open `house-price-prediction.ipynb` and run the cells in order.

## Repository Structure

```text
.
├── house-price-prediction.ipynb
├── Housing.csv
└── README.md
```

The dataset is listed as a separate file because the notebook expects it to be available when executed. Include it in the repository only if its license permits redistribution.

## Key Concepts Demonstrated

- Exploratory data analysis and visualization
- Numerical and categorical data preprocessing
- One-hot encoding
- Train-test splitting
- Log transformation of a regression target
- Linear, regularized, and ensemble regression
- Regression evaluation and residual analysis
- Applying a fitted preprocessing pipeline to new data

## Limitations

- Predictions depend on the dataset's coverage, quality, and original price units.
- The model is trained on the available dataset and may not generalize to other locations or housing markets.
- The notebook evaluates models on a single train-test split.
- The sample prediction is a demonstration, not a verified market valuation.
- The notebook's evaluation code labels prices with `$`; confirm the dataset's currency and units before interpreting or reporting the results.

## Future Improvements

- Use cross-validation and hyperparameter tuning.
- Add a reusable prediction function that accepts property details.
- Save the fitted preprocessing transformer and model for reuse.
- Add further regression models and compare them under the same evaluation setup.
- Improve validation and handling of invalid or out-of-range user inputs.
- Build a simple web interface for interactive price estimation.

## Author

**Sneha Mittra**

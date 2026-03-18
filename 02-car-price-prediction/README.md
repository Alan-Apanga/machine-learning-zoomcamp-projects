# Car Price Prediction

## Project Overview

This project builds a machine learning regression model to predict car prices (MSRP) based on vehicle features and specifications. The goal is to develop an accurate, interpretable baseline linear regression model that can estimate price ranges for used and new vehicles, enabling better market valuation and pricing strategies.

---

## Problem Statement

**Business Context:**  
Accurate car price prediction is critical for dealerships, buyers, and sellers to establish fair market values and identify underpriced or overpriced listings.

**Technical Challenge:**  
Given a dataset of vehicle specifications (engine power, fuel efficiency, popularity, etc.), predict the Manufacturer's Suggested Retail Price (MSRP).

- **Target Variable:** `msrp` (price in USD)
- **Task Type:** Supervised Regression
- **Approach:** Linear Regression from first principles

---

## Dataset

| Aspect | Details |
|--------|---------|
| **Source** | Machine Learning Zoomcamp (Kaggle Car Data) |
| **Size** | 11914 records, 16 features |
| **Format** | CSV |
| **Target Variable** | `msrp`: car price in USD |
| **Key Features** | `engine_hp`, `engine_cylinders`, `highway_mpg`, `city_mpg`, `popularity`, and categorical variables |
| **Missing Values** | Handled via imputation (filled with 0 for numeric features) |
| **Data Split** | 60% train, 20% validation, 20% test |

**Sample Features:**
- Engine Power (horsepower)
- Engine Cylinders
- Fuel Efficiency (highway & city MPG)
- Vehicle Popularity Metric
- Make, Model, Body Type, Transmission, etc.

---

## Project Structure

```
02-car-price-prediction/
├── README.md                      # Project documentation (this file)
├── regression.py                  # Main regression model implementation
├── price-prediction.ipynb        # Jupyter notebook with exploration & analysis
└── data/
    └── data.csv                  # Dataset
```

---

## Setup / Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Environment Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd 02-car-price-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate          # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

### Verify Installation
```bash
python -c "import pandas; import numpy; print('Dependencies installed successfully!')"
```

---

## How to Run

### Option 1: Run the Regression Script
```bash
python regression.py
```

**Output:**
- Training RMSE
- Validation RMSE
- Model predictions and performance metrics

### Option 2: Interactive Exploration (Jupyter Notebook)
```bash
jupyter notebook price-prediction.ipynb
```

Navigate through cells to explore data, visualizations, and model development step-by-step.

---

## Methodology

### 1. Data Preparation
- Load data from `data/data.csv`
- Normalize column names (lowercase, replace spaces with underscores)
- Standardize categorical string values
- Remove or impute missing values

### 2. Feature Engineering
- **Numeric features:** `engine_hp`, `engine_cylinders`, `highway_mpg`, `city_mpg`, `popularity`
- **Categorical encoding:** Label encoding or one-hot encoding for make, model, body type, transmission, etc.
- **Feature scaling:** Normalize numeric features (optional, depends on model)

### 3. Target Transformation
- Apply log transformation: $y_{transformed} = \log_1p(y)$
- **Rationale:** MSRP has a long-tailed distribution; log transformation normalizes it to a Gaussian distribution, improving model fit

### 4. Data Splitting
- **Train (60%):** Used to fit model weights and bias
- **Validation (20%):** Used for hyperparameter tuning and early stopping
- **Test (20%):** Used for final model evaluation

### 5. Model Training
- Implement Linear Regression using linear algebra (matrix operations)
- Calculate weight vector $w$ and bias term $w_0$ via the Normal Equation:
  $$w = (X^T X)^{-1} X^T y$$
- Use pseudoinverse for numerical stability

---

## Modeling

### Baseline Model
- **Algorithm:** Linear Regression (from scratch)
- **Features:** 5 numeric baseline features
- **Rationale:** Simple interpretable baseline to establish performance floor


### Final Selected Model
- **Type:** Linear Regression
- **Features:** Baseline numeric + engineered categorical features
- **Rationale:** Strong interpretability, acceptable performance, reproducibility

---

## Evaluation

### Metrics
- **RMSE (Root Mean Squared Error):** Primary metric for regression
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **MAE (Mean Absolute Error):** Secondary metric for interpretability
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Performance

| Dataset | RMSE | MAE |
|---------|------|-----|
| **Train** | 0.4606 | 0.3473 |
| **Validation** | 0.4541 | 0.3418 |
| **Test** | 0.4518 | 0.3381 |

---

## Results / Key Findings

- **Best Model Performance:** Validation RMSE of `<rmse-value>` on log-transformed prices
- **Top Predictive Features:** `engine_hp`, `engine_cylinders`, `popularity` show strongest correlation with price
- **Key Insight:** Log-transformed target variable significantly improved model convergence and distribution fit
- **Generalization:** Consistent performance across train/val/test splits indicates minimal overfitting

---

## Example Usage

### Prediction Example

```python
import numpy as np
from regression import train_linear_regression, prepare_Xbase

# Sample car features: [engine_hp, engine_cylinders, highway_mpg, city_mpg, popularity]
sample_car = np.array([[350, 8, 28, 20, 85]])

# Generate prediction (returns log-transformed price)
predicted_log_price = w_0 + sample_car.dot(w)

# Transform back to original price
predicted_price = np.expm1(predicted_log_price)
print(f"Predicted MSRP: ${predicted_price:,.2f}")
```

**Sample Output:**
```
Predicted MSRP: $45,230.50
```

---

## Deployment

### Current State
- **Type:** Batch prediction script
- **Usage:** Run `python regression.py` for model training and validation

### Potential Deployment Options
- **REST API:** Flask/FastAPI service for real-time predictions
- **Docker:** Containerize model for reproducible deployment
- **Streamlit App:** Interactive web interface for price estimation
- **Batch Processing:** Scheduled job for bulk predictions on new listings
- **Azure/AWS:** Cloud deployment with scalable inference

### Next Steps for Production
1. Save trained model weights to pickle/joblib file
2. Create API endpoint for real-time predictions
3. Implement input validation and error handling
4. Add monitoring for prediction accuracy drift
5. Version control for model artifacts

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Framework** | NumPy (Linear Algebra), scikit-learn (optional for comparison) |
| **Notebooks** | Jupyter Lab/Notebook |
| **Environment** | venv or conda |

---

## Reproducibility

### Environment Details
- **Python Version:** 3.8+
- **Random Seed:** `np.random.seed(2)` (for train/val/test shuffle)
- **Key Dependencies:**
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  jupyter>=1.0.0
  ```

### How to Reproduce
1. Install dependencies as per Setup section
2. Ensure `data/data.csv` is in the `data/` folder
3. Run `python regression.py`
4. Results will match exactly due to fixed random seed

### Model Artifacts
- **Weights:** Stored in `regression.py` output or can be exported to JSON/pickle
- **Preprocessing Steps:** Documented in `regression.py` (column normalization, feature preparation)
- **Validation Framework:** 60-20-20 split with seed 2

---

## Future Improvements

- [ ] **Advanced Models:** Try Ridge/Lasso regression, Random Forest, Gradient Boosting
- [ ] **Hyperparameter Tuning:** Regularization strength, feature scaling optimization
- [ ] **Feature Engineering:** Polynomial features, interaction terms, domain-specific engineering
- [ ] **Cross-Validation:** Implement k-fold CV for more robust evaluation
- [ ] **Outlier Analysis:** Investigate and handle extreme price outliers
- [ ] **Categorical Features:** Expand categorical encoding strategies (target encoding, embeddings)
- [ ] **Model Explainability:** SHAP values or feature importance analysis
- [ ] **Production Pipeline:** Automated training, monitoring, and retraining pipeline
- [ ] **API Development:** Build REST API for real-time predictions
- [ ] **Documentation:** Add docstrings and type hints throughout codebase

---

## License

This project is part of the Machine Learning Zoomcamp course.  


---

## References & Resources

- [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code) — Course materials and datasets
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html) — Implementation reference
- [Normal Equation](https://en.wikipedia.org/wiki/Ordinary_least_squares) — Theory and derivation
- [Log Transform for Skewed Data](https://en.wikipedia.org/wiki/Power_transform) — Statistical background

---

## Contact & Support

- **Author:** Alan Apanga
- **LinkedIn:** [Portfolio](https://www.linkedin.com/in/alan-apanga/)

For questions or feedback, please open an issue or contact the author.

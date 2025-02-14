# Financial Time Series Regression Analysis App

## Overview
This Streamlit application provides comprehensive regression analysis tools for financial time series data, offering both simple and multiple regression capabilities. The app is designed to analyze Brazilian financial system data from March 2006 to December 2016, featuring key financial indicators and economic metrics.

## Features

### Simple Linear Regression Analysis
- Interactive variable selection for dependent and independent variables
- Visual representation of the regression line with slope and intercept
- Comprehensive statistical summary including R² and MSE
- Point-specific analysis showing actual values, predictions, and errors
- Residual analysis and diagnostic plots

### Multiple Linear Regression Analysis
- Support for multiple independent variables
- Advanced statistical metrics (R², Adjusted R², F-statistic)
- Multicollinearity analysis using VIF (Variance Inflation Factor)
- Comprehensive residual diagnostics
- Multiple normality tests (D'Agostino-Pearson, Shapiro-Wilk, Kolmogorov-Smirnov)
- Confidence bands for Q-Q plots

### Available Variables
- IR: Interest Rate
- GDP: Gross Domestic Product
- CRED: Credit to GDP Ratio
- ROE: Return on Equity
- LEV: Capital to Assets Ratio
- NPL: Non-Performing Loans
- INFLA: Inflation
- TARG: Brazilian Monetary Policy Inflation Targeting
- EMBI: JP Morgan's Emerging Markets Bonds Index

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-regression-analysis.git
cd financial-regression-analysis
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- streamlit
- pandas
- numpy
- plotly
- statsmodels
- scipy

## Usage

1. Place your data file (series_temporais.csv) in the app directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the app through your web browser (typically http://localhost:8501)

### Using Simple Regression
1. Select a dependent variable (Y)
2. Choose an independent variable (X)
3. Analyze the regression line and statistics
4. Examine residual plots and diagnostics

### Using Multiple Regression
1. Select a dependent variable
2. Choose multiple independent variables
3. Analyze the regression results, including:
   - Coefficient analysis
   - VIF statistics for multicollinearity
   - Residual diagnostics
   - Normality tests

## Data Format
The application expects a CSV file named 'series_temporais.csv' with the following structure:
- First column: Date (formatted as YYYY-MM-DD)
- Subsequent columns: Financial variables (IR, GDP, CRED, etc.)

## Model Assumptions and Diagnostics

### Linear Regression Assumptions
1. Linearity of variables
2. Independence of residuals
3. Normal distribution of residuals
4. Equal variance of residuals (homoscedasticity)

### Diagnostic Tools
- Residuals vs Fitted Values plot
- Normal Q-Q plot with confidence bands
- Multiple normality tests
- VIF analysis for multicollinearity
- Statistical summaries and coefficients

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
[MIT License](LICENSE)

## Authors
[Your Name]

## Acknowledgments
- Brazilian Financial System data providers
- Streamlit community
- Statistical analysis libraries contributors

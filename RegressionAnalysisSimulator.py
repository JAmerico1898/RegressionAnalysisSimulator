import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import statsmodels.api as sm

# Center the heading using HTML
st.markdown(
    "<h5 style='text-align: center;'>Make your choice!</h5>",
    unsafe_allow_html=True
)

options = ["Simple Linear Regression", "Multiple Linear Regression"]

# Create three columns, putting the widest space (or equal space) at the edges
col1, col2, col3 = st.columns([1, 2, 1])

# Place the radio button inside the middle column
with col2:
    geral = st.radio(
        label="", 
        options=options, 
        index=None, 
        disabled=False, 
        horizontal=True
    )
    
if geral == "Multiple Linear Regression":


    # Set page configuration

    # Title and introduction
    st.title("Multiple Linear Regression Analysis Tool")

    # Explanation of Multiple Linear Regression and Assumptions
    st.header("Multiple Linear Regression Overview")
    st.write("""
    Multiple linear regression extends simple linear regression by incorporating multiple independent variables 
    to predict a dependent variable. The model is represented as:
    Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε

    Where:
    - β₀ is the intercept
    - βᵢ are the coefficients for each independent variable
    - Xᵢ are the independent variables
    - ε is the error term
    """)

    st.subheader("Key Assumptions:")
    st.write("""
    1. **Linearity**: The relationship between independent variables and the dependent variable should be linear
    2. **Independence**: The residuals should be independent
    3. **Normality**: The residuals should be normally distributed
    4. **Homoscedasticity**: The residuals should have constant variance
    5. **No Multicollinearity**: Independent variables should not be highly correlated with each other
    """)

    # Load and prepare data
    @st.cache_data
    def load_data():
        df = pd.read_csv('series_temporais.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df

    try:
        df = load_data()
        
        # Variable selection
        variables = ['CRED', 'IR', 'GDP', 'ROE', 'LEV', 'NPL', 'INFLA', 'TARG', 'EMBI']
        
        # Select dependent variable
        dependent_var = st.selectbox(
            'Select Dependent Variable (Y)',
            variables
        )
        
        # Select multiple independent variables
        remaining_vars = [var for var in variables if var != dependent_var]
        independent_vars = st.multiselect(
            'Select Independent Variables (X)',
            remaining_vars,
            default=remaining_vars[:2]  # Default select first two variables
        )
        
        if len(independent_vars) < 1:
            st.warning("Please select at least one independent variable.")
        else:
            # Prepare data for regression
            X = df[independent_vars]
            y = df[dependent_var]
            
            # Add constant to X for statsmodels
            X_with_const = sm.add_constant(X)
            
            # Fit regression model
            model = sm.OLS(y, X_with_const).fit()
            
            # Display regression results
            st.header("Regression Analysis")
            
            # Model Statistics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Overall Model Fit:**")
                st.write(f"R-squared: {model.rsquared:.4f}")
                st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                st.write(f"F-statistic: {model.fvalue:.4f}")
                st.write(f"F-statistic p-value: {model.f_pvalue:.4f}")
            
            with col2:
                st.write("**Error Metrics:**")
                st.write(f"Mean Square Error: {np.mean(model.resid**2):.4f}")
                st.write(f"Root Mean Square Error: {np.sqrt(np.mean(model.resid**2)):.4f}")
                st.write(f"AIC: {model.aic:.4f}")
                st.write(f"BIC: {model.bic:.4f}")
            
            # Coefficient Analysis
            st.subheader("Coefficient Analysis")
            coef_df = pd.DataFrame({
                'Coefficient': model.params.round(4),
                'Std Error': model.bse.round(4),
                't-value': model.tvalues.round(4),
                'P-value': model.pvalues.round(4)
            })
            st.write(coef_df.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                't-value': '{:.4f}',
                'P-value': '{:.4f}'
            }))
            
            # Multicollinearity Analysis
            st.subheader("Multicollinearity Analysis")
            
            st.write("""
            VIF stands for **Variance Inflation Factor**. It is a numerical measure used in regression analysis to quantify 
            how much the variance of a regression coefficient is inflated due to collinearity (i.e., correlation) among 
            the explanatory variables.

            Here's the key idea:
            * For each predictor xₖ in the model, you regress xₖ on the other predictors to obtain R²ₖ, the 
            coefficient of determination from that auxiliary regression.
            * The Variance Inflation Factor for xₖ is calculated as VIFₖ = 1/(1 - R²ₖ).

            If R²ₖ is high (meaning xₖ can be well-explained by the other predictors), then 1 - R²ₖ is small, so VIFₖ 
            will be large, indicating a potential multicollinearity problem.

            A common rule of thumb is that:
            * VIF > 5 (sometimes 10, depending on the field and context) may indicate problematic levels of 
            multicollinearity that could bias the regression results or make coefficient estimates unstable.
            """)
            
            # Calculate and display VIF values
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_data["VIF"] = vif_data["VIF"].round(4)  # Round to 4 decimal places
            st.write(vif_data)
            
            # Actual vs Predicted Plot
            st.subheader("Actual vs Predicted Values")
            fig_pred = px.scatter(
                x=y, y=model.fittedvalues,
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                title='Actual vs Predicted Values'
            )
            
            # Add 45-degree line
            fig_pred.add_trace(
                go.Scatter(
                    x=[y.min(), y.max()],
                    y=[y.min(), y.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                )
            )
            
            # Update layout for better visualization
            fig_pred.update_layout(
                legend=dict(
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    orientation="h"
                )
            )
            
            st.plotly_chart(fig_pred, use_container_width=True, key='pred_plot')
            
            # Residual Analysis
            st.subheader("Residual Analysis")

            # Calculate and display mean of residuals
            mean_residuals = np.mean(model.resid)
            st.write(f"Mean of Residuals: {mean_residuals:.4f}")
            
            st.write("""
            A good "Residuals vs Fitted Values" plot for a valid linear regression generally looks like an unstructured 
            "cloud" of points around the horizontal zero line, with no obvious pattern or trend. In particular:

            1. **Mean of Residuals is Zero**
            The residuals should cluster symmetrically around zero, indicating that on average, the regression 
            doesn't systematically over- or under-predict.

            2. **No Systematic Pattern**
            You do not want to see curves, funnels (i.e., the spread getting wider as fitted values increase), or 
            any other obvious structure. Any pattern can hint at issues such as nonlinearity, heteroskedasticity, 
            or missing variables.

            3. **Roughly Constant Variance**
            The spread (variance) of residuals should be roughly the same across the range of fitted values 
            (homoscedasticity). If the variance appears to grow (or shrink) in one portion of the plot, this 
            indicates possible heteroskedasticity.

            When these conditions are met—no clear patterns, a roughly constant spread, and a center near zero—
            the linear model assumptions are more likely to hold, making the regression analysis more reliable.
            """)
            
            # Residual vs Fitted Plot
            fig_resid = px.scatter(
                x=model.fittedvalues,
                y=model.resid,
                labels={'x': 'Fitted Values', 'y': 'Residuals'},
                title='Residuals vs Fitted Values'
            )
            
            # Add horizontal line at y=0
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_resid, use_container_width=True, key='resid_plot')
            
            # QQ Plot with confidence bands
            st.subheader("Normal Q-Q Plot Analysis")
            
            # Calculate theoretical quantiles
            n = len(model.resid)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
            observed_quantiles = np.sort(model.resid)
            
            # Calculate confidence bands
            # Using simulation to generate confidence bands
            n_simulations = 1000
            simulated_bands = np.zeros((n_simulations, n))
            for i in range(n_simulations):
                simulated_data = stats.norm.rvs(size=n)
                simulated_bands[i] = np.sort(simulated_data)
            
            confidence_level = 0.90
            lower_band = np.percentile(simulated_bands, (1 - confidence_level) * 100 / 2, axis=0)
            upper_band = np.percentile(simulated_bands, (1 + confidence_level) * 100 / 2, axis=0)
            
            # Create Q-Q plot with confidence bands
            fig_qq = go.Figure()
            
            # Add confidence bands
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=upper_band,
                    mode='lines',
                    line=dict(color='rgba(173, 216, 230, 0.5)'),
                    name=f'{int(confidence_level * 100)}% Confidence Band',
                    showlegend=False
                )
            )
            
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=lower_band,
                    mode='lines',
                    line=dict(color='rgba(173, 216, 230, 0.5)'),
                    fill='tonexty',
                    name=f'{int(confidence_level * 100)}% Confidence Band'
                )
            )
            
            # Add Q-Q line
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=observed_quantiles,
                    mode='markers',
                    name='Sample Quantiles',
                    marker=dict(color='blue')
                )
            )
            
            # Add 45-degree reference line
            fig_qq.add_trace(
                go.Scatter(
                    x=[min(theoretical_quantiles), max(theoretical_quantiles)],
                    y=[min(theoretical_quantiles), max(theoretical_quantiles)],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig_qq.update_layout(
                title='Normal Q-Q Plot with Confidence Bands',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                showlegend=True
            )
            
            st.plotly_chart(fig_qq, use_container_width=True, key='qq_plot')
            
            # Multiple Normality Tests
            st.subheader("Normality Tests")
            
            # D'Agostino-Pearson test
            dagostino_stat, dagostino_p = stats.normaltest(model.resid)
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(model.resid)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(stats.zscore(model.resid), 'norm')
            
            # Create a DataFrame with test results
            normality_tests = pd.DataFrame({
                'Test': ['D\'Agostino-Pearson', 'Shapiro-Wilk', 'Kolmogorov-Smirnov'],
                'Statistic': [dagostino_stat, shapiro_stat, ks_stat],
                'p-value': [dagostino_p, shapiro_p, ks_p]
            })
            
            # Round the values
            normality_tests['Statistic'] = normality_tests['Statistic'].round(4)
            normality_tests['p-value'] = normality_tests['p-value'].round(4)
            
            st.write("""
            **Interpretation of Normality Tests:**
            - H₀ (null hypothesis): The data follows a normal distribution
            - H₁ (alternative hypothesis): The data does not follow a normal distribution
            - If p-value > 0.05, we fail to reject the null hypothesis (data appears normal)
            - If p-value ≤ 0.05, we reject the null hypothesis (data appears non-normal)
            """)
            
            st.write(normality_tests)
            
            # Point Prediction
            st.subheader("Point Prediction")
            st.write("Select values for independent variables to make a prediction:")
            
            # Create input fields for each independent variable
            input_values = {}
            for var in independent_vars:
                min_val = float(df[var].min())
                max_val = float(df[var].max())
                default_val = float(df[var].mean())
                input_values[var] = st.slider(
                    f"Select value for {var}",
                    min_val, max_val, default_val,
                    format="%.2f"
                )
            
            # Make prediction
            # Create a list of values in the same order as independent variables
            input_list = [input_values[var] for var in independent_vars]
            
            # Create the prediction array with explicit shape
            prediction_array = np.array(input_list).reshape(1, -1)
            
            # Create DataFrame with proper column names and shape
            prediction_input = pd.DataFrame(prediction_array, columns=independent_vars)
            
            # Add constant and make prediction
            prediction_input_with_const = sm.add_constant(prediction_input, has_constant='add')
            prediction = model.predict(prediction_input_with_const).iloc[0]
            
            st.write(f"**Predicted {dependent_var}:** {prediction:.4f}")
            
            # Assumption Tests
            st.subheader("Model Assumptions Tests")
            
            # Normality test
            _, normality_p = stats.normaltest(model.resid)
            
            # Heteroscedasticity test
            _, het_p, _, _ = het_breuschpagan(model.resid, X_with_const)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Normality Test (D'Agostino-Pearson):**")
                st.write(f"p-value: {normality_p:.4f}")
                st.write("Interpretation: Residuals are normal if p > 0.05")
            
            with col2:
                st.write("**Heteroscedasticity Test (Breusch-Pagan):**")
                st.write(f"p-value: {het_p:.4f}")
                st.write("Interpretation: Variance is constant if p > 0.05")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please make sure the 'series_temporais.csv' file is available and contains the required columns.")
        
        
elif geral == "Simple Linear Regression":        


    # Title and introduction
    st.title("Linear Regression Analysis Tool")

    # Explanation of Linear Regression and Assumptions
    st.header("Linear Regression Overview")
    st.write("""
    Simple linear regression is a statistical method that allows us to study the relationship between two continuous 
    variables: a dependent variable (Y) and an independent variable (X). The relationship is modeled using a linear equation:
    Y = β₀ + β₁X + ε

    Where:
    - β₀ is the intercept (value of Y when X = 0)
    - β₁ is the slope (change in Y for a one-unit change in X)
    - ε is the error term
    """)

    st.subheader("Key Assumptions:")
    st.write("""
    1. **Linearity**: The relationship between X and Y should be linear
    2. **Independence**: The residuals should be independent
    3. **Normality**: The residuals should be normally distributed
    4. **Homoscedasticity**: The residuals should have constant variance
    """)

    # Load and prepare data
    @st.cache_data
    def load_data():
        df = pd.read_csv('series_temporais.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df

    try:
        df = load_data()
        
        # Variable selection
        variables = ['IR', 'GDP', 'CRED', 'ROE', 'LEV', 'NPL', 'INFLA', 'TARG', 'EMBI']
        
        col1, col2 = st.columns(2)
        
        with col1:
            dependent_var = st.selectbox(
                'Select Dependent Variable (Y)',
                variables
            )
        
        with col2:
            independent_var = st.selectbox(
                'Select Independent Variable (X)',
                [var for var in variables if var != dependent_var]
            )
        
        # Prepare data for regression
        X = df[independent_var].values.reshape(-1, 1)
        y = df[dependent_var].values
        
        # Add constant to X for statsmodels
        X_with_const = sm.add_constant(X)
        
        # Fit regression model
        model = sm.OLS(y, X_with_const).fit()
        
        # Plot regression line
        st.header("Regression Analysis")
        
        fig = px.scatter(df, x=independent_var, y=dependent_var, 
                        title=f'Linear Regression: {dependent_var} vs {independent_var}')
        
        # Add regression line
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_with_const = sm.add_constant(x_range)
        y_pred_line = model.predict(X_range_with_const)
        
        # Add regression line with equation
        equation_text = f'y = {model.params[0]:.2f} + {model.params[1]:.2f}x'
        fig.add_trace(
            go.Scatter(x=x_range.flatten(), y=y_pred_line,
                    mode='lines', name='Regression Line',
                    line=dict(color='red'))
        )
        
        # Update layout to position legend at the top
        fig.update_layout(
            legend=dict(
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                orientation="h"
            ),
            title=dict(
                text=f'Linear Regression: {dependent_var} vs {independent_var}<br><sub>{equation_text}</sub>',
                x=0.5,
                xanchor='center'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Statistics
        st.subheader("Regression Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Summary:**")
            st.write(f"R-squared: {model.rsquared:.4f}")
            st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            st.write(f"Mean Square Error: {np.mean(model.resid**2):.4f}")
            st.write(f"Root Mean Square Error: {np.sqrt(np.mean(model.resid**2)):.4f}")
        
        with col2:
            st.write("**Coefficient Statistics:**")
            st.write(f"Intercept (β₀): {model.params[0]:.4f}")
            st.write(f"Slope (β₁): {model.params[1]:.4f}")
            st.write(f"P-value (slope): {model.pvalues[1]:.4f}")
        
        # Point Prediction
        st.subheader("Point Prediction")
        
        # Allow user to select a specific point
        point_index = st.slider("Select a data point", 0, len(df)-1, len(df)//2)
        
        actual_x = X[point_index][0]
        actual_y = y[point_index]
        predicted_y = model.predict(X_with_const)[point_index]
        
        st.write(f"**Selected Point:**")
        st.write(f"X ({independent_var}): {actual_x:.4f}")
        st.write(f"Actual Y ({dependent_var}): {actual_y:.4f}")
        st.write(f"Predicted Y: {predicted_y:.4f}")
        st.write(f"Prediction Error: {actual_y - predicted_y:.4f}")
        
        # Assumption Tests
        st.subheader("Model Assumptions Tests")
        
        # Normality test
        _, normality_p = stats.normaltest(model.resid)
        
        # Heteroscedasticity test
        _, het_p, _, _ = het_breuschpagan(model.resid, X_with_const)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Normality Test (D'Agostino-Pearson):**")
            st.write(f"p-value: {normality_p:.4f}")
            st.write("Interpretation: Residuals are normal if p > 0.05")
        
        with col2:
            st.write("**Heteroscedasticity Test (Breusch-Pagan):**")
            st.write(f"p-value: {het_p:.4f}")
            st.write("Interpretation: Variance is constant if p > 0.05")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please make sure the 'series_temporais.csv' file is available and contains the required columns.")        

# Footer
st.divider()
st.caption("© 2025 Linear Regression Teaching Tool | Developed for educational purposes")
st.caption("Prof. José Américo – Coppead")

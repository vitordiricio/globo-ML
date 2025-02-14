import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from utils.model_evaluation import ModelEvaluator

# Try importing XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Initialize session state for storing models
if 'models' not in st.session_state:
    st.session_state.models = {}

# Page configuration
st.set_page_config(
    page_title="ML Regression Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header
st.title("🔍 ML Regression Analysis Tool")
st.markdown("""
    Upload your CSV file and perform regression analysis with interactive visualizations.
    This tool helps you understand your data and make predictions.
""")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Column selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox(
                "Select target variable (Y)",
                options=numeric_columns,
                help="Choose the variable you want to predict"
            )

        with col2:
            available_features = [col for col in numeric_columns if col != target_column]

            # Add select all option
            all_option = "Select All Features"
            options = [all_option] + available_features

            selected = st.multiselect(
                "Select feature variables (X)",
                options=options,
                help="Choose the variables to use for prediction"
            )

            # Handle Select All option
            if all_option in selected:
                feature_columns = available_features
            else:
                feature_columns = selected

        if feature_columns and target_column:
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]

            # Handle missing values
            X = X.fillna(X.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Model selection tabs
            st.subheader("Model Analysis")
            tabs = st.tabs([
                "Linear Regression",
                "XGBoost",
                "Model Comparison",
                "CatBoost",
                "LightGBM",
                "Market Mix Modeling",
                "Deep Learning (LSTM)"
            ])

            # Linear Regression Tab
            with tabs[0]:
                st.markdown("""
                    ### Linear Regression
                    Linear regression is a fundamental statistical model that assumes a linear relationship between inputs and outputs.
                    It's useful for:
                    - Understanding direct relationships between variables
                    - Making simple, interpretable predictions
                    - Identifying the strength of feature impacts on the target variable
                """)

                with st.spinner("Training Linear Regression model..."):
                    # Train model
                    lr_model = LinearRegression()
                    lr_model.fit(X_train, y_train)
                    st.session_state.models['Linear Regression'] = lr_model

                    # Make predictions
                    y_pred = lr_model.predict(X_test)

                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)

                    # Display equation
                    st.subheader("Model Equation")
                    equation = f"{target_column} = {lr_model.intercept_:.4f}"
                    for coef, feature in zip(lr_model.coef_, feature_columns):
                        equation += f" + ({coef:.4f} × {feature})"
                    st.code(equation)

                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    metric_cols = st.columns(3)
                    metrics = {
                        'R²': round(r2, 3),
                        'MSE': round(mse, 3),
                        'RMSE': round(rmse, 3)
                    }

                    for i, (metric, value) in enumerate(metrics.items()):
                        with metric_cols[i]:
                            st.metric(metric, value)

                    # Visualization section
                    st.subheader("Visualizations")

                    # Actual vs Predicted plot
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                        title='Actual vs Predicted Values'
                    )

                    # Add perfect prediction line
                    fig.add_trace(
                        go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='#50E3C2', dash='dash')
                        )
                    )

                    fig.update_layout(
                        template='plotly_white',
                        plot_bgcolor='white',
                        width=800,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Feature importance
                    importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': np.abs(lr_model.coef_)
                    })
                    importance = importance.sort_values('Importance', ascending=True)

                    fig_importance = px.bar(
                        importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance'
                    )
                    fig_importance.update_layout(template='plotly_white')
                    st.plotly_chart(fig_importance, use_container_width=True)

            # XGBoost Tab
            with tabs[1]:
                st.markdown("""
                    ### XGBoost Regression
                    XGBoost (eXtreme Gradient Boosting) is a powerful and efficient implementation of gradient boosted trees.
                    Key advantages:
                    - Automatic feature selection and importance ranking
                    - Handles non-linear relationships
                    - Robust to outliers and missing values
                    - High prediction accuracy
                """)

                if not XGBOOST_AVAILABLE:
                    st.warning("""
                        XGBoost is not installed in this environment. 
                        Please install it using: pip install xgboost

                        Until then, you can explore other models like Linear Regression.
                    """)
                else:
                    with st.spinner("Training XGBoost model..."):
                        # Train XGBoost model
                        xgb_model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=100,
                            learning_rate=0.1,
                            random_state=42
                        )
                        xgb_model.fit(X_train, y_train)
                        st.session_state.models['XGBoost'] = xgb_model

                        # Make predictions
                        y_pred = xgb_model.predict(X_test)

                        # Calculate metrics
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)

                        # Display metrics
                        st.subheader("Model Performance Metrics")
                        metric_cols = st.columns(3)
                        metrics = {
                            'R²': round(r2, 3),
                            'MSE': round(mse, 3),
                            'RMSE': round(rmse, 3)
                        }

                        for i, (metric, value) in enumerate(metrics.items()):
                            with metric_cols[i]:
                                st.metric(metric, value)

                        # Visualization section
                        st.subheader("Visualizations")

                        # Actual vs Predicted plot
                        fig = px.scatter(
                            x=y_test,
                            y=y_pred,
                            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                            title='Actual vs Predicted Values'
                        )

                        # Add perfect prediction line
                        fig.add_trace(
                            go.Scatter(
                                x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='#50E3C2', dash='dash')
                            )
                        )

                        fig.update_layout(
                            template='plotly_white',
                            plot_bgcolor='white',
                            width=800,
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Feature importance
                        importance = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': xgb_model.feature_importances_
                        })
                        importance = importance.sort_values('Importance', ascending=True)

                        fig_importance = px.bar(
                            importance,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (XGBoost)'
                        )
                        fig_importance.update_layout(template='plotly_white')
                        st.plotly_chart(fig_importance, use_container_width=True)

            # Model Comparison Tab
            with tabs[2]:
                st.markdown("""
                    ### Model Comparison
                    Compare the performance of different models using cross-validation.
                    This helps us understand:
                    - Which model performs better on average
                    - How stable each model's predictions are
                    - If the performance differences are statistically significant
                """)

                if len(st.session_state.models) > 0:
                    with st.spinner("Performing cross-validation..."):
                        # Initialize model evaluator
                        evaluator = ModelEvaluator(st.session_state.models)

                        # Perform cross-validation
                        evaluator.perform_cross_validation(X, y, cv=5)

                        # Display summary statistics
                        st.subheader("Cross-Validation Summary")
                        summary_stats = evaluator.get_summary_stats()
                        st.dataframe(summary_stats.style.format("{:.4f}"))

                        # Plot comparison
                        st.subheader("Performance Distribution")
                        cv_plot = evaluator.plot_cv_comparison()
                        st.plotly_chart(cv_plot, use_container_width=True)

                        # Statistical comparison
                        st.subheader("Statistical Comparison")
                        st.markdown("""
                            P-values for pairwise model comparisons.
                            Values < 0.05 indicate statistically significant differences.
                        """)
                        p_values = evaluator.perform_statistical_test()
                        st.dataframe(p_values.style.format("{:.4f}"))
                else:
                    st.info("Train at least one model to see the comparison.")

            # Placeholder for other models
            for i, model_name in enumerate([
                "CatBoost", "LightGBM", 
                "Market Mix Modeling", "Deep Learning (LSTM)"
            ], 3):
                with tabs[i]:
                    st.info(f"🚧 {model_name} implementation coming soon!")
                    st.markdown(f"""
                        ### {model_name}
                        This model is currently being implemented. It will include:
                        - Model training and predictions
                        - Performance metrics
                        - Feature importance visualization
                        - Model equation or feature impact analysis
                        - Results export
                    """)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")

# Footer
st.markdown("""
    ---
    Made with ❤️ using Streamlit | Data Science Tool
""")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

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
            feature_columns = st.multiselect(
                "Select feature variables (X)",
                options=[col for col in numeric_columns if col != target_column],
                help="Choose the variables to use for prediction"
            )

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

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

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
                'Importance': np.abs(model.coef_)
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

            # Export results
            st.subheader("Export Results")
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })

            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="regression_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")

# Footer
st.markdown("""
    ---
    Made with ❤️ using Streamlit | Data Science Tool
""")
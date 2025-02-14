#utils/ml_models.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import shap
import plotly.graph_objects as go
import plotly.express as px

class RegressionAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        self.explainer = None
        
    def train_model(self, X_train, y_train):
        """Train the regression model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {
            'R²': round(r2, 3),
            'MSE': round(mse, 3),
            'RMSE': round(rmse, 3)
        }
    
    def calculate_shap_values(self, X):
        """Calculate SHAP values"""
        self.explainer = shap.LinearExplainer(self.model, X)
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def create_prediction_plot(self, y_true, y_pred):
        """Create scatter plot of actual vs predicted values"""
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            labels={'x': 'Valores reais', 'y': 'Valores preditos'},
            title='Reais vs Predições'
        )
        
        # Add perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[y_true.min(), y_true.max()],
                y=[y_true.min(), y_true.max()],
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
        
        return fig
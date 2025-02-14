import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def validate_csv(self, file):
        try:
            df = pd.read_csv(file)
            return df, None
        except Exception as e:
            return None, str(e)
    
    def prepare_data(self, df, target_column, feature_columns, test_size=0.2):
        """Prepare data for ML model"""
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, self.scaler
    
    def get_numeric_columns(self, df):
        """Return numeric columns from dataframe"""
        return df.select_dtypes(include=[np.number]).columns.tolist()

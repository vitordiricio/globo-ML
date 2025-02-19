# models/xgboost_model.py
import xgboost as xgb
from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    Algoritmo avançado de aprendizado de máquina baseado em árvores de decisão.
    Vantagens:
    - Captura relações não lineares
    - Robustez a outliers
    - Alta precisão nas previsões
    """
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'random_state': 42}
        self.params = params  # Save parameters to support get_params/set_params
        self.model = xgb.XGBRegressor(**params)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_

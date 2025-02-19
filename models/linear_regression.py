# models/linear_regression.py
from sklearn.linear_model import LinearRegression
from models.base_model import BaseModel

class LinearRegressionModel(BaseModel):
    """
    Modelo estatístico fundamental que assume uma relação linear entre as variáveis.
    Ideal para:
    - Relações diretas entre variáveis
    - Previsões simples e interpretáveis
    - Identificar força de impacto das variáveis
    """
    def __init__(self, normalize_data=False):
        super().__init__()
        self.normalize_data = normalize_data  # Parameter to allow future normalization
        self.model = LinearRegression()
    
    def train(self, X_train, y_train):
        # Future: implement normalization if self.normalize_data is True
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    @property
    def intercept_(self):
        return self.model.intercept_
    
    @property
    def coef_(self):
        return self.model.coef_
    
    @property
    def feature_importances_(self):
        # For linear regression, the coefficients represent variable importance
        return self.model.coef_

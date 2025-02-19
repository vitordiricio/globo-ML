# ml_models.py
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb

class BaseModel(BaseEstimator):
    """
    Modelo base que deve ser estendido pelos demais modelos.
    As subclasses devem implementar os métodos train e predict, além de definir
    os atributos de descrição e nome do modelo.
    """
    model_name = "BaseModel"
    description = "Modelo base. Substitua os atributos na subclasse."

    def __init__(self):
        pass

    def train(self, X_train, y_train):
        """
        Treina o modelo.
        Deve ser implementado pela subclasse.
        """
        raise NotImplementedError("O método train deve ser implementado pela subclasse.")

    def predict(self, X):
        """
        Faz as previsões.
        Deve ser implementado pela subclasse.
        """
        raise NotImplementedError("O método predict deve ser implementado pela subclasse.")

    def run(self, X, y):
        """
        Separa os dados, treina o modelo e faz as previsões.
        
        Args:
            X (DataFrame): Features.
            y (Series): Variável alvo.
        
        Returns:
            dict: Contém y_test, y_pred, X_test e X_train.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train(X_train, y_train)
        y_pred = self.predict(X_test)
        return {"y_test": y_test, "y_pred": y_pred, "X_test": X_test, "X_train": X_train}

    def fit(self, X, y):
        """
        Alias para train() para compatibilidade com a interface do scikit-learn.
        """
        return self.train(X, y)

class LinearRegressionModel(BaseModel):
    """
    Modelo de Regressão Linear.
    Ideal para:
    - Relações diretas entre variáveis
    - Previsões simples e interpretáveis
    - Identificar força de impacto das variáveis
    """
    model_name = "Regressão Linear"
    description = """
    Modelo estatístico fundamental que assume uma relação linear entre as variáveis.
    Ideal para:
    - Relações diretas entre variáveis
    - Previsões simples e interpretáveis
    - Identificar força de impacto das variáveis
    """
    default_params = {'normalize_data': False}

    def __init__(self, normalize_data=None):
        super().__init__()
        if normalize_data is None:
            normalize_data = self.default_params['normalize_data']
        self.normalize_data = normalize_data  # Parâmetro para permitir normalização futura
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        # Futuro: implementar normalização se self.normalize_data for True
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
        # Para regressão linear, os coeficientes representam a importância das variáveis
        return self.model.coef_

class XGBoostModel(BaseModel):
    """
    Modelo XGBoost.
    Vantagens:
    - Captura relações não lineares
    - Robustez a outliers
    - Alta precisão nas previsões
    """
    model_name = "XGBoost"
    description = """
    Algoritmo avançado de aprendizado de máquina baseado em árvores de decisão.
    Vantagens:
    - Captura relações não lineares
    - Robustez a outliers
    - Alta precisão nas previsões
    """
    default_params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'random_state': 42}

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = self.__class__.default_params.copy()
        self.params = params  # Armazena os parâmetros para suportar get_params/set_params
        self.model = xgb.XGBRegressor(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

# Lista de classes de modelos disponíveis.
AVAILABLE_MODELS = [LinearRegressionModel, XGBoostModel]

# models/base_model.py
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

class BaseModel(BaseEstimator):
    """
    Base model that other models should extend.
    Subclasses should override the description attribute, train, and predict methods.
    """
    description = "Base model. Override description in subclass."
    
    def __init__(self):
        # No parameters by default; subclasses should store their parameters as attributes.
        pass
    
    def train(self, X_train, y_train):
        """
        Trains the model.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("O método train deve ser implementado pela subclasse.")
    
    def predict(self, X):
        """
        Makes predictions.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("O método predict deve ser implementado pela subclasse.")
    
    def run(self, X, y):
        """
        Splits the data, trains the model, and makes predictions.
        
        Args:
            X (DataFrame): Features.
            y (Series): Target variable.
        
        Returns:
            dict: Contains y_test, y_pred, and X_test.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train(X_train, y_train)
        y_pred = self.predict(X_test)
        return {"y_test": y_test, "y_pred": y_pred, "X_test": X_test}
    
    def fit(self, X, y):
        """
        Alias for train() to be compatible with scikit-learn's interface.
        """
        return self.train(X, y)

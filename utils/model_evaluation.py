#utils/model_evaluation.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

class ModelEvaluator:
    def __init__(self, models_dict):
        """
        Initialize with a dictionary of models
        Args:
            models_dict (dict): Dictionary with model names as keys and model objects as values
        """
        self.models = models_dict
        self.cv_results = {}
        self.cv_predictions = {}
        
    def perform_cross_validation(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform k-fold cross-validation for all models
        """
        for name, model in self.models.items():
            cv_scores = cross_val_score(
                model, X, y, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1
            )
            
            if scoring.startswith('neg_'):
                cv_scores = -cv_scores
                
            self.cv_results[name] = cv_scores
            
    def get_summary_stats(self):
        """
        Get summary statistics for each model's CV results
        """
        summary = {}
        for name, scores in self.cv_results.items():
            summary[name] = {
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'Min': np.min(scores),
                'Max': np.max(scores)
            }
        return pd.DataFrame(summary).T
        
    def plot_cv_comparison(self):
        """
        Create a box plot comparing model performances
        """
        fig = go.Figure()
        
        for name, scores in self.cv_results.items():
            fig.add_trace(go.Box(
                y=scores,
                name=name,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
        fig.update_layout(
            title='Cross-Validation Performance Comparison',
            yaxis_title='Mean Squared Error',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
        
    def perform_statistical_test(self, alpha=0.05):
        """
        Perform statistical test to compare models
        Returns: DataFrame with p-values for model comparisons
        """
        model_names = list(self.cv_results.keys())
        n_models = len(model_names)
        p_values = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Perform paired t-test
                t_stat, p_val = stats.ttest_rel(
                    self.cv_results[model_names[i]],
                    self.cv_results[model_names[j]]
                )
                p_values[i,j] = p_val
                p_values[j,i] = p_val
                
        return pd.DataFrame(
            p_values,
            columns=model_names,
            index=model_names
        )
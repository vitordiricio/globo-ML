# utils/model_evaluation.py
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from scipy import stats

class ModelEvaluator:
    def __init__(self, models_dict):
        """
        Inicializa com um dicionário de modelos.
        
        Args:
            models_dict (dict): Dicionário com nomes dos modelos como chaves e objetos dos modelos como valores.
        """
        self.models = models_dict
        self.cv_results = {}
        self.cv_predictions = {}
        
    def perform_cross_validation(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Executa validação cruzada k-fold para todos os modelos que implementam a interface fit.
        """
        for name, model in self.models.items():
            if not hasattr(model, 'fit'):
                # Pula modelos que não implementam a interface esperada
                continue
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
        Retorna estatísticas resumo dos resultados da validação cruzada para cada modelo.
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
        Cria um box plot comparando a performance dos modelos na validação cruzada.
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
        Executa teste estatístico para comparar os modelos.
        
        Returns:
            DataFrame com os valores-p das comparações entre os modelos.
        """
        model_names = list(self.cv_results.keys())
        n_models = len(model_names)
        p_values = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
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

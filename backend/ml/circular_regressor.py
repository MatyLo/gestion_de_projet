"""
Advanced Circular Regression Model for Fire Direction Prediction
Traite correctement les angles circulaires (0° = 360°)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class CircularRegressor(BaseEstimator, RegressorMixin):
    """
    Régression pour angles circulaires.
    
    Au lieu de prédire directement l'angle (0-360°), prédit:
    - sin(angle) et cos(angle)
    
    Avantages:
    - Pas de discontinuité à 0°/360°
    - Meilleure gestion de la circularité
    """
    
    def __init__(self, base_estimator=None, n_estimators=300, max_depth=25):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_sin = None
        self.model_cos = None
    
    def fit(self, X, y):
        """
        Entraîne deux modèles: un pour sin(angle), un pour cos(angle)
        
        Args:
            X: Features (n_samples, n_features)
            y: Angles en degrés (n_samples,)
        """
        # Convertir angles en composantes sin/cos
        y_rad = np.radians(y)
        y_sin = np.sin(y_rad)
        y_cos = np.cos(y_rad)
        
        # Créer les modèles de base
        if self.base_estimator is None:
            self.model_sin = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.model_cos = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=43,  # Seed différent
                n_jobs=-1
            )
        else:
            from copy import deepcopy
            self.model_sin = deepcopy(self.base_estimator)
            self.model_cos = deepcopy(self.base_estimator)
        
        print("   Entraînement modèle sin...")
        self.model_sin.fit(X, y_sin)
        
        print("   Entraînement modèle cos...")
        self.model_cos.fit(X, y_cos)
        
        return self
    
    def predict(self, X):
        """
        Prédit les angles
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Angles en degrés (n_samples,)
        """
        # Prédire sin et cos
        y_sin_pred = self.model_sin.predict(X)
        y_cos_pred = self.model_cos.predict(X)
        
        # Normaliser (sin²+cos²=1)
        magnitude = np.sqrt(y_sin_pred**2 + y_cos_pred**2)
        y_sin_pred = y_sin_pred / (magnitude + 1e-8)
        y_cos_pred = y_cos_pred / (magnitude + 1e-8)
        
        # Convertir en angle
        angles_rad = np.arctan2(y_sin_pred, y_cos_pred)
        angles_deg = np.degrees(angles_rad) % 360
        
        return angles_deg
    
    def get_confidence(self, X):
        """
        Estime la confiance de la prédiction basée sur la magnitude
        
        Returns:
            Confiance (0-1) pour chaque prédiction
        """
        y_sin_pred = self.model_sin.predict(X)
        y_cos_pred = self.model_cos.predict(X)
        
        # Magnitude proche de 1 = haute confiance
        magnitude = np.sqrt(y_sin_pred**2 + y_cos_pred**2)
        
        return magnitude


class WeightedCircularRegressor(BaseEstimator, RegressorMixin):
    """
    Combine régression directe + régression circulaire avec pondération
    """
    
    def __init__(self, alpha=0.6):
        """
        Args:
            alpha: Poids de la régression circulaire (0-1)
                   1.0 = uniquement circulaire
                   0.0 = uniquement directe
        """
        self.alpha = alpha
        self.circular_model = None
        self.direct_model = None
    
    def fit(self, X, y):
        print(f"   Alpha (circulaire): {self.alpha:.2f}")
        
        # Modèle circulaire
        self.circular_model = CircularRegressor(n_estimators=200, max_depth=25)
        self.circular_model.fit(X, y)
        
        # Modèle direct
        self.direct_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        print("   Entraînement modèle direct...")
        self.direct_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        # Prédictions des deux modèles
        pred_circular = self.circular_model.predict(X)
        pred_direct = self.direct_model.predict(X)
        
        # Moyenne circulaire pondérée
        pred_circ_rad = np.radians(pred_circular)
        pred_direct_rad = np.radians(pred_direct)
        
        # Convertir en vecteurs
        sin_circ = np.sin(pred_circ_rad) * self.alpha
        cos_circ = np.cos(pred_circ_rad) * self.alpha
        
        sin_direct = np.sin(pred_direct_rad) * (1 - self.alpha)
        cos_direct = np.cos(pred_direct_rad) * (1 - self.alpha)
        
        # Combiner
        sin_combined = sin_circ + sin_direct
        cos_combined = cos_circ + cos_direct
        
        # Convertir en angle
        angles_rad = np.arctan2(sin_combined, cos_combined)
        angles_deg = np.degrees(angles_rad) % 360
        
        return angles_deg

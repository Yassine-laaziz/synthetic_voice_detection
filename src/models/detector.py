"""
AI Voice Detection Models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, Tuple, Optional


class VoiceDetector:
    """Classifier for detecting AI-generated voices"""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize detector

        Args:
            model_type: Type of classifier ('random_forest', 'svm')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Train the detector

        Args:
            X: Feature DataFrame
            y: Labels (0 for human, 1 for AI)

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

        # Predictions for detailed report
        y_pred = self.model.predict(X_test_scaled)

        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict whether voice is AI-generated

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            # For SVM without probability, use decision function
            probabilities = self.model.decision_function(X_scaled)
            # Normalize to [0, 1]
            probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())

        return predictions, probabilities

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # For SVM, use absolute coefficients
            importances = np.abs(self.model.coef_[0])

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, path: str):
        """
        Save trained model

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }

        joblib.dump(model_data, path)

    def load_model(self, path: str):
        """
        Load trained model

        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']

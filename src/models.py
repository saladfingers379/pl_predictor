import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class XGBoostPredictor:
    def __init__(self, features=None, params=None):
        """
        Initialize the XGBoost Predictor.
        
        Args:
            features (list): List of column names to use as features.
            params (dict): Hyperparameters for the XGBoost model.
        """
        self.features = features
        self.params = params if params else {
            'objective': 'multi:softprob',
            'num_class': 3, # Home, Draw, Away
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.params)
        
    def tune_hyperparameters(self, df, n_iter=10):
        """
        Performs RandomizedSearchCV to find better hyperparameters.
        """
        if not self.features:
            raise ValueError("No features specified for tuning.")
            
        X = df[self.features]
        y = df['Target']
        
        param_dist = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5]
        }
        
        # Use TimeSeriesSplit to respect temporal order
        tscv = TimeSeriesSplit(n_splits=5)
        
        xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42)
        
        random_search = RandomizedSearchCV(
            xgb_model, 
            param_distributions=param_dist, 
            n_iter=n_iter, 
            scoring='neg_log_loss', 
            cv=tscv, 
            verbose=1, 
            n_jobs=-1,
            random_state=42
        )
        
        print("Tuning hyperparameters...")
        random_search.fit(X, y)
        
        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best score: {random_search.best_score_}")
        
        # Update model with best params
        self.params.update(random_search.best_params_)
        self.model = xgb.XGBClassifier(**self.params)
        
    def train(self, df, tune=False):
        """
        Trains the model on the provided DataFrame.
        """
        if not self.features:
            raise ValueError("No features specified for training.")
            
        if tune:
            self.tune_hyperparameters(df)
            
        X = df[self.features]
        y = df['Target']
        
        # Split for validation during training (optional, but good practice)
        # Use a time-based split manually if not using CV
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training on {len(X_train)} samples with features: {self.features}")
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred, target_names=['Home', 'Draw', 'Away']))
        
    def predict_proba(self, df):
        """
        Returns probabilities for the input DataFrame.
        """
        if not self.features:
            raise ValueError("No features specified.")
            
        X = df[self.features]
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        """
        Returns and plots feature importance.
        """
        importance = self.model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': self.features, 'Importance': importance})
        feat_imp = feat_imp.sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feat_imp)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp['Feature'], feat_imp['Importance'])
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('data/feature_importance.png')
        print("Feature importance plot saved to data/feature_importance.png")
        plt.close()

        return feat_imp


class ElasticNetPredictor:
    def __init__(self, features=None, params=None):
        """
        Initialize the Elastic Net Predictor using LogisticRegression with elasticnet penalty.

        Args:
            features (list): List of column names to use as features.
            params (dict): Hyperparameters for the model.
        """
        self.features = features
        self.params = params if params else {
            'penalty': 'elasticnet',
            'solver': 'saga',  # saga supports elasticnet
            'C': 1.0,  # Inverse of regularization strength (smaller = stronger regularization)
            'l1_ratio': 0.5,  # Balance between L1 and L2 (0=L2, 1=L1, 0.5=equal mix)
            'max_iter': 1000,
            'random_state': 42,
            'multi_class': 'multinomial'  # For 3-class problem
        }
        self.model = LogisticRegression(**self.params)
        self.scaler = StandardScaler()  # Elastic Net requires feature scaling

    def tune_hyperparameters(self, df, n_iter=20):
        """
        Performs RandomizedSearchCV to find better hyperparameters.
        Focus on C (regularization strength) and l1_ratio (L1/L2 balance).
        """
        if not self.features:
            raise ValueError("No features specified for tuning.")

        X = df[self.features]
        y = df['Target']

        # Scale features for Elastic Net
        X_scaled = self.scaler.fit_transform(X)

        param_dist = {
            'C': np.logspace(-3, 2, 20),  # Wide range of regularization strengths
            'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # L1/L2 balance
            'max_iter': [1000, 2000, 3000]  # Convergence iterations
        }

        # Use TimeSeriesSplit to respect temporal order
        tscv = TimeSeriesSplit(n_splits=5)

        elastic_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            multi_class='multinomial',
            random_state=42
        )

        random_search = RandomizedSearchCV(
            elastic_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_log_loss',
            cv=tscv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        print("Tuning Elastic Net hyperparameters...")
        random_search.fit(X_scaled, y)

        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best score: {random_search.best_score_}")

        # Update model with best params
        self.params.update(random_search.best_params_)
        self.model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            multi_class='multinomial',
            random_state=42,
            **random_search.best_params_
        )

    def train(self, df, tune=False):
        """
        Trains the model on the provided DataFrame.
        """
        if not self.features:
            raise ValueError("No features specified for training.")

        if tune:
            self.tune_hyperparameters(df)

        X = df[self.features]
        y = df['Target']

        # Split for validation during training
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features (fit on train, transform both)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"Training Elastic Net on {len(X_train)} samples with features: {self.features}")
        print(f"Regularization: C={self.params.get('C', 1.0):.4f}, l1_ratio={self.params.get('l1_ratio', 0.5):.2f}")

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred, target_names=['Home', 'Draw', 'Away']))

    def predict_proba(self, df):
        """
        Returns probabilities for the input DataFrame.
        """
        if not self.features:
            raise ValueError("No features specified.")

        X = df[self.features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        """
        Returns and plots feature coefficients (importance) for the multinomial model.
        For 3-class problem, we'll take the absolute mean across classes.
        """
        # Get coefficients for all classes (shape: n_classes x n_features)
        coefs = self.model.coef_

        # Take absolute mean across classes to get overall feature importance
        importance = np.abs(coefs).mean(axis=0)

        feat_imp = pd.DataFrame({'Feature': self.features, 'Importance': importance})
        feat_imp = feat_imp.sort_values('Importance', ascending=False)

        print("\nElastic Net Feature Importance (Mean Absolute Coefficients):")
        print(feat_imp)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp['Feature'], feat_imp['Importance'])
        plt.xlabel('Importance (Mean |Coefficient|)')
        plt.title('Elastic Net Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('data/feature_importance_elasticnet.png')
        print("Feature importance plot saved to data/feature_importance_elasticnet.png")
        plt.close()

        return feat_imp


class EnsemblePredictor:
    def __init__(self, features=None, weights=None):
        """
        Initialize the Ensemble Predictor combining XGBoost and Elastic Net.

        Args:
            features (list): List of column names to use as features.
            weights (dict): Weights for each model. Default: {'xgboost': 0.5, 'elasticnet': 0.5}
        """
        self.features = features
        self.weights = weights if weights else {'xgboost': 0.5, 'elasticnet': 0.5}

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        # Initialize both models
        self.xgb_model = XGBoostPredictor(features=features)
        self.elastic_model = ElasticNetPredictor(features=features)

    def tune_hyperparameters(self, df, n_iter=20):
        """
        Tunes hyperparameters for both models independently.
        """
        print("=" * 60)
        print("Tuning XGBoost hyperparameters...")
        print("=" * 60)
        self.xgb_model.tune_hyperparameters(df, n_iter=n_iter)

        print("\n" + "=" * 60)
        print("Tuning Elastic Net hyperparameters...")
        print("=" * 60)
        self.elastic_model.tune_hyperparameters(df, n_iter=n_iter)

    def train(self, df, tune=False):
        """
        Trains both models on the provided DataFrame.
        """
        if not self.features:
            raise ValueError("No features specified for training.")

        if tune:
            self.tune_hyperparameters(df)

        print("=" * 60)
        print("Training Ensemble Model (XGBoost + Elastic Net)")
        print("=" * 60)
        print(f"Model weights: XGBoost={self.weights['xgboost']:.2f}, ElasticNet={self.weights['elasticnet']:.2f}")

        # Train XGBoost
        print("\n--- Training XGBoost ---")
        self.xgb_model.train(df, tune=False)  # Already tuned if needed

        # Train Elastic Net
        print("\n--- Training Elastic Net ---")
        self.elastic_model.train(df, tune=False)  # Already tuned if needed

        # Evaluate ensemble on validation set
        y = df['Target']
        split_idx = int(len(df) * 0.8)
        y_val = y.iloc[split_idx:]

        # Get ensemble predictions on validation set
        val_df = df.iloc[split_idx:].copy()
        ensemble_probs = self.predict_proba(val_df)
        y_pred = np.argmax(ensemble_probs, axis=1)

        acc = accuracy_score(y_val, y_pred)
        print("\n" + "=" * 60)
        print(f"Ensemble Validation Accuracy: {acc:.4f}")
        print("=" * 60)
        print(classification_report(y_val, y_pred, target_names=['Home', 'Draw', 'Away']))

    def predict_proba(self, df):
        """
        Returns ensemble probabilities by weighted averaging of both models.
        """
        if not self.features:
            raise ValueError("No features specified.")

        # Get predictions from both models
        xgb_probs = self.xgb_model.predict_proba(df)
        elastic_probs = self.elastic_model.predict_proba(df)

        # Weighted average
        ensemble_probs = (
            self.weights['xgboost'] * xgb_probs +
            self.weights['elasticnet'] * elastic_probs
        )

        return ensemble_probs

    def get_feature_importance(self):
        """
        Returns feature importance from both models.
        """
        print("\n" + "=" * 60)
        print("Feature Importance - XGBoost Component")
        print("=" * 60)
        xgb_imp = self.xgb_model.get_feature_importance()

        print("\n" + "=" * 60)
        print("Feature Importance - Elastic Net Component")
        print("=" * 60)
        elastic_imp = self.elastic_model.get_feature_importance()

        # Create combined importance (weighted average)
        combined = pd.DataFrame({
            'Feature': self.features,
            'XGBoost_Importance': xgb_imp.set_index('Feature')['Importance'],
            'ElasticNet_Importance': elastic_imp.set_index('Feature')['Importance']
        })

        # Normalize importances to [0, 1] for fair comparison
        combined['XGBoost_Normalized'] = (
            combined['XGBoost_Importance'] / combined['XGBoost_Importance'].max()
        )
        combined['ElasticNet_Normalized'] = (
            combined['ElasticNet_Importance'] / combined['ElasticNet_Importance'].max()
        )

        # Weighted average of normalized importances
        combined['Ensemble_Importance'] = (
            self.weights['xgboost'] * combined['XGBoost_Normalized'] +
            self.weights['elasticnet'] * combined['ElasticNet_Normalized']
        )

        combined = combined.sort_values('Ensemble_Importance', ascending=False)

        print("\n" + "=" * 60)
        print("Combined Ensemble Feature Importance")
        print("=" * 60)
        print(combined[['Feature', 'Ensemble_Importance']])

        # Plot ensemble importance
        plt.figure(figsize=(10, 6))
        plt.barh(combined['Feature'], combined['Ensemble_Importance'])
        plt.xlabel('Ensemble Importance (Weighted Average)')
        plt.title('Ensemble Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('data/feature_importance_ensemble.png')
        print("Feature importance plot saved to data/feature_importance_ensemble.png")
        plt.close()

        return combined[['Feature', 'Ensemble_Importance']]

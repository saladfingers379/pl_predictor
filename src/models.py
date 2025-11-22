import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

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

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
            'random_state': 42  # Fix random seed for reproducible results
        }
        self.model = xgb.XGBClassifier(**self.params)
        
    def train(self, df):
        """
        Trains the model on the provided DataFrame.
        """
        if not self.features:
            raise ValueError("No features specified for training.")
            
        X = df[self.features]
        y = df['Target']
        
        # Split for validation during training (optional, but good practice)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
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
        plt.show()
        
        return feat_imp

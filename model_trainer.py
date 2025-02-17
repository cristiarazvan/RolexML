import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import re
from sklearn.ensemble import GradientBoostingRegressor

class RolexPriceTrainer:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_enc = {}
        self.scaler = StandardScaler()
        self.full_models = {}
        self.valid_values = {}
        self.feature_columns = None

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath, sep=';')
        return self

    def preprocess_data(self):
        # Handle complications
        self.df['Complication'] = self.df['Complication'].fillna('')
        dummies = self.df['Complication'].str.get_dummies(sep=',')
        self.df = self.df.join(dummies)
        self.df.drop('Complication', axis=1, inplace=True)
        
        self.valid_values['Complication'] = sorted(list(dummies.columns))
        
        # Handle RRP
        self.df = self.df[self.df['RRP'] != "POR"]
        self.df.loc[:, "RRP"] = self.df["RRP"].astype("float32")
        
        self._process_description()
        
        self._encode_categorical()
        
        return self

    def _process_description(self):
        self.df["Dial"] = self.df.Description.apply(lambda txt: self._get_part(r'^(.*?)\s*Dial', txt))
        self.df["Bracelet"] = self.df.Description.apply(lambda txt: self._get_part(r'\b(\w+)\b\s+Bracelet\b', txt))
        self.df.Bracelet[self.df["Bracelet"] == "Diamond"] = "DiamondBracelet"
        
        self.valid_values['Dial'] = sorted(self.df['Dial'].unique().tolist())
        self.valid_values['Bracelet'] = sorted(self.df['Bracelet'].unique().tolist())
        
        self.df.drop("Description", axis=1, inplace=True)
        
        dial_dummies = self.df['Dial'].str.get_dummies(sep=',')
        bracelet_dummies = self.df['Bracelet'].str.get_dummies(sep=',')
        self.df = pd.concat([self.df.drop(['Dial', 'Bracelet'], axis=1), 
                           dial_dummies, bracelet_dummies], axis=1)

    @staticmethod
    def _get_part(regex, txt):
        x = re.search(regex, txt)
        if x:
            if x.group(1) == "Mother of Peal Diamond":
                return "Mother of Pearl Diamond"
            if x.group(1) == "\xa0Standard":
                return "Standard"
            return x.group(1).strip()
        return "None"

    def _encode_categorical(self):
        for col in ['Reference', 'Collection']:
            self.label_enc[col] = LabelEncoder()
            self.df[col] = self.label_enc[col].fit_transform(self.df[col])
            self.valid_values[col] = sorted(self.label_enc[col].classes_.tolist())
            
        self.valid_values['Size'] = {
            'min': float(self.df['Size'].min()),
            'max': float(self.df['Size'].max()),
            'step': 1.0
        }
        
        self.df['Size'] = self.scaler.fit_transform(self.df['Size'].to_numpy().reshape(-1, 1))

    def split_data(self, test_size=0.2, random_state=42):
        self.y = self.df['RRP']
        self.X = self.df.drop(["RRP", "None"], axis=1)
        self.feature_columns = self.X.columns
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        return self

    def train_model(self, model_type='xgboost'):
        models = {
            'xgboost': xgb.XGBRegressor(n_estimators=3000, learning_rate=0.1, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=3000, random_state=42),
            'linear': LinearRegression(),
            'gradient_boost': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, random_state=42)
        }
        
        model = models.get(model_type)
        if model is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        return {
            'predictions': predictions,
            'mae': mean_absolute_error(self.y_test, predictions),
            'mse': mean_squared_error(self.y_test, predictions),
            'r2': r2_score(self.y_test, predictions)
        }

    def train_full_models(self):
        # Train models on the full dataset
        try:
            print("Starting full model training...")
            models = {
                'xgboost': xgb.XGBRegressor(n_estimators=3000, learning_rate=0.1, random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=3000, random_state=42),
                'linear': LinearRegression(),
                'gradient_boost': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, random_state=42)
            }
            
            X_full = self.df.drop(["RRP", "None"], axis=1)
            y_full = self.df['RRP']
            
            print(f"Training data shape: {X_full.shape}")
            print(f"Target data shape: {y_full.shape}")
            
            for name, model in models.items():
                print(f"Training {name} model...")
                try:
                    model.fit(X_full, y_full)
                    self.full_models[name] = model
                    # Verify model works with a test prediction
                    test_pred = model.predict(X_full[:1])
                    print(f"{name} model trained successfully. Test prediction: {test_pred[0]}")
                except Exception as e:
                    print(f"Error training {name} model: {str(e)}")
                    raise
            
            print("All models trained successfully")
            print(f"Number of trained models: {len(self.full_models)}")
            
        except Exception as e:
            print(f"Error in train_full_models: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def predict_price(self, input_data):
        # Predict price using all trained full models
        try:
            new_data = pd.DataFrame([input_data])
            
            print("Input data received:", input_data)
            
            for col in ['Reference', 'Collection']:
                try:
                    new_data[col] = self.label_enc[col].transform(new_data[col])
                    print(f"Encoded {col}:", new_data[col].values)
                except Exception as e:
                    print(f"Error encoding {col}:", str(e))
                    raise
            
            try:
                new_data['Size'] = self.scaler.transform([[float(new_data['Size'].iloc[0])]])
                print("Transformed Size:", new_data['Size'].values)
            except Exception as e:
                print("Error transforming Size:", str(e))
                raise
            
            print("Current columns:", new_data.columns.tolist())
            
            for dial in self.valid_values['Dial']:
                new_data[dial] = 1 if dial == input_data['Dial'] else 0
                
            for bracelet in self.valid_values['Bracelet']:
                new_data[bracelet] = 1 if bracelet == input_data['Bracelet'] else 0
            
            print("After adding Dial and Bracelet columns:", new_data.columns.tolist())
            
            for complication in self.valid_values['Complication']:
                new_data[complication] = 1 if complication in input_data.get('Complication', []) else 0
            
            print("After adding Complications:", new_data.columns.tolist())
            
            print("Required feature columns:", self.feature_columns.tolist())
            for col in self.feature_columns:
                if col not in new_data.columns:
                    print(f"Adding missing column: {col}")
                    new_data[col] = 0
            
            new_data = new_data[self.feature_columns]
            print("Final columns:", new_data.columns.tolist())
            
            predictions = {}
            valid_predictions = []
            
            print("Number of trained models:", len(self.full_models))
            print("Available models:", list(self.full_models.keys()))
            
            for name, model in self.full_models.items():
                try:
                    pred = float(model.predict(new_data)[0])
                    pred = max(0, pred)  # Ensure no negative prices
                    print(f"Prediction from {name}:", pred)
                    predictions[name] = pred
                    valid_predictions.append(pred)
                except Exception as e:
                    print(f"Error in {name} prediction: {str(e)}")
                    predictions[name] = None
            
            if valid_predictions:
                predictions['mean'] = np.mean(valid_predictions)
                print("Mean prediction:", predictions['mean'])
            else:
                predictions['mean'] = None
                print("No valid predictions available")
            
            return predictions
            
        except Exception as e:
            print("Error in predict_price:", str(e))
            import traceback
            traceback.print_exc()
            return {'mean': None}

    def get_valid_values(self):
        """Get all valid values for input fields"""
        return self.valid_values

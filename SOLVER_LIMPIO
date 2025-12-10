import pandas as pd
import numpy as np
import pickle
import difflib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class APS_Solver:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None

    # -----------------------------------------------------------
    # LIMPIEZA GENERAL Y CREACIÓN DE FEATURES
    # -----------------------------------------------------------
    def _clean_str(self, s):
        return str(s).strip().replace(" ", "_").replace("-", "_").lower()

    def _closest(self, value, valid_values):
        value = str(value).lower()
        match = difflib.get_close_matches(value, valid_values, n=1, cutoff=0.7)
        return match[0] if match else value

    def _preprocess(self, df, fit=True):
        df = df.copy().drop_duplicates()
    
        # Guardamos Revenue y eliminamos para preprocesar features
        revenue_col = None
        if "Revenue" in df.columns:
            revenue_col = df["Revenue"].astype(int)
            df = df.drop(columns=["Revenue"])
    
        # Columnas numéricas y categóricas
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        cat_cols = [c for c in df.select_dtypes(include=["object", "bool"]).columns]
    
        # Llenar NaNs en numéricas
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    
        # Limpiar y aproximar categorías de manera general
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
            valid_vals = df[col].dropna().unique()
            df[col] = df[col].apply(lambda x: difflib.get_close_matches(x, valid_vals, n=1, cutoff=0.7)[0] if difflib.get_close_matches(x, valid_vals, n=1, cutoff=0.7) else x)
    
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
        # Escalado numérico
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            self.feature_columns = df.columns
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
            # Reindex para mantener columnas de entrenamiento
            df = df.reindex(columns=self.feature_columns, fill_value=0)
    
        # Volver a añadir Revenue
        if revenue_col is not None:
            df["Revenue"] = revenue_col
    
        return df

    # -----------------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------------
    def train_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        print("==== Métricas de validación ====")
        print("Tasa de error:", 1 - accuracy_score(y_val, y_pred))
        print("Precisión:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1-score:", f1_score(y_val, y_pred))

    # -----------------------------------------------------------
    # TESTEO
    # -----------------------------------------------------------
    def test_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=False)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        y_pred = self.model.predict(X)
        print("==== Métricas de test ====")
        print("Tasa de error:", 1 - accuracy_score(y, y_pred))
        print("Precisión:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))
        print("F1-score:", f1_score(y, y_pred))

      
    
model = APS_Solver() 
model.train_model("online_shoppers_train.csv")
model.test_model("online_shoppers_test.csv") 

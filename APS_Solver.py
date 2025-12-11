import pandas as pd
import numpy as np
import pickle
import difflib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class APS_Solver:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.preprocessor_cluster = None

    # LIMPIEZA GENERAL Y CREACIÓN DE FEATURES
    
    def _clean_str(self, s):
        return str(s).strip().replace(" ", "_").replace("-", "_").lower()

    def _closest(self, value, valid_values):
        value = str(value).lower()
        match = difflib.get_close_matches(value, valid_values, n=1, cutoff=0.7)
        return match[0] if match else value

    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(
                (
                    self.model,
                    self.scaler,
                    self.label_encoders,
                    self.preprocessor_cluster
                ), f)
  

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            (
                self.model,
                self.scaler,
                self.label_encoders,
                self.preprocessor_cluster
            ) = pickle.load(f)

    def _preprocess(self, df, fit=True):
        df = df.copy().drop_duplicates()
    
        revenue_col = None
        if "Revenue" in df.columns:
            revenue_col = df["Revenue"].astype(int)
            df = df.drop(columns=["Revenue"])
    
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        cat_cols = [c for c in df.select_dtypes(include=["object", "bool"]).columns]
    
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        
        for col in cat_cols:
            df[col] = df[col].apply(self._clean_str)

        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
            valid_vals = df[col].dropna().unique()
            df[col] = df[col].apply(lambda x: self._closest(x, valid_vals))
    
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            self.feature_columns = df.columns
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
            df = df.reindex(columns=self.feature_columns, fill_value=0)
    
        if revenue_col is not None:
            df["Revenue"] = revenue_col
    
        if "ProductRelated_Duration" in df.columns and "Informational_Duration" in df.columns and "Administrative_Duration" in df.columns:
            df["Duration_Total"] = df["ProductRelated_Duration"] + df["Informational_Duration"] + df["Administrative_Duration"]
            df = df.drop(columns=["ProductRelated_Duration", "Informational_Duration", "Administrative_Duration"])


        for col in ["BounceRates", "OperatingSystems", "Browser", "Weekend"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        return df

    # ENTRENAMIENTO
    
    def train_model(self, file_path):
        self.cluster_data(file_path, save_csv=True, gafas = True)
        df = pd.read_csv("database_clusters.csv")
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=550, max_depth=12, random_state=42, class_weight="balanced")

        self.model.fit(X_train, y_train)

    # TESTEO
    
    def test_model(self, file_path):
        self.cluster_data(file_path, save_csv=True, gafas= False)
        df = pd.read_csv("database_clusters.csv")
        df = self._preprocess(df, fit=False)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        y_pred = self.model.predict(X)
        print("Tasa de error:", 1 - accuracy_score(y, y_pred))
        print("Precisión:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))
        print("F1-score:", f1_score(y, y_pred))


    # CLUSTERING SEPARADO POR REVENUE E INGENIERIA DE CARACTERISTICAS

    def cluster_data(self, file_path, k=9, save_csv=True, gafas = True):
        df = pd.read_csv(file_path)

        num_cols = ['Administrative','Total_Duration','Informational',
                    'ProductRelated', 'ExitRates','PageValues','SpecialDay']
        cat_cols = ['Month', 'VisitorType', 'Region', 'TrafficType']

        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        preprocessor = ColumnTransformer([
            ('num', Pipeline([('scaler', StandardScaler())]), [c for c in num_cols if c in df.columns]),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), 
             [c for c in cat_cols if c in df.columns])
        ])

        X = preprocessor.fit_transform(df)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        df['cluster'] = kmeans.fit_predict(X)

        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_dense)
        
        df["PC1"] = X_pca[:, 0]
        df["PC2"] = X_pca[:, 1]
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df["PC1"], df["PC2"], c=df["cluster"], cmap="tab10", alpha=0.7)
        
        if gafas:
            plt.title("Visualización de Clusters en el TRAIN con PCA")
        else:
            plt.title("Visualización de Clusters en el TEST con PCA")
            
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.grid(True)
        plt.show()
        
        if save_csv:
            df.to_csv("database_clusters.csv", index=False)

        self.preprocessor_cluster = preprocessor
        return df
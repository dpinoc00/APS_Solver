import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
import difflib

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

    # -----------------------------------------------------------
    # LIMPIEZA GENERAL Y CREACIÓN DE FEATURES
    # -----------------------------------------------------------
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
  
        print(f"Modelo guardado en: {file_path}")

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            (
                self.model,
                self.scaler,
                self.label_encoders,
                self.preprocessor_cluster
            ) = pickle.load(f)
        print(f"Modelo cargado desde: {file_path}")

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
    
        if "ProductRelated_Duration" in df.columns and "Informational_Duration" in df.columns and "Administrative_Duration" in df.columns:
            df["Duration_Total"] = df["ProductRelated_Duration"] + df["Informational_Duration"] + df["Administrative_Duration"]
            df = df.drop(columns=["ProductRelated_Duration", "Informational_Duration", "Administrative_Duration"])


        for col in ["BounceRates", "OperatingSystems", "Browser", "Weekend"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        return df

    # -----------------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------------
    def train_model(self, file_path):
        self.cluster_data(file_path, save_csv=True)
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        print("==== Métricas de validación ====")
        print("Tasa de error:", 1 - accuracy_score(y_val, y_pred))
        print("Precisión:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1-score:", f1_score(y_val, y_pred))
        print('\n')

    # -----------------------------------------------------------
    # TESTEO
    # -----------------------------------------------------------
    def test_model(self, file_path):
        self.cluster_data(file_path, save_csv=True)
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

    # ------------------------------------------------------
    # CLUSTERING SEPARADO POR REVENUE + FEATURE ENGINEERING
    # ------------------------------------------------------

    def cluster_data(self, file_path, k=8, save_csv=True):
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

        if save_csv:
            df.to_csv("database_clusters.csv", index=False)
            print("\nArchivo generado: database_clusters.csv")

        self.preprocessor_cluster = preprocessor
        return df

    def cluster_group(self, df_group, k, label="Global"):
        X = self.preprocessor_cluster.transform(df_group)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(X)

        df_group['cluster'] = cluster_labels
        df_group['cluster_global'] = label
        df_group['cluster_id'] = df_group['cluster_global'] + "_" + df_group['cluster'].astype(str)

        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_dense)
        df_group["PC1"] = X_pca[:, 0]
        df_group["PC2"] = X_pca[:, 1]

        # Visual opcional
        plt.figure(figsize=(8, 6))
        for c in range(k):
            pts = X_pca[cluster_labels == c]
            plt.scatter(pts[:, 0], pts[:, 1], alpha=0.6, label=f'Cluster {c}')
        plt.title(f'PCA – {label}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()
    
        return df_group



model = APS_Solver() 
model.train_model("online_shoppers_train.csv")
model.test_model("online_shoppers_test.csv") 

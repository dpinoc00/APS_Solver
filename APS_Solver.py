import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import difflib

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class APS_Solver:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.preprocessor_cluster = None

    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(
                (
                    self.model,
                    self.scaler,
                    self.label_encoders,
                    self.preprocessor_cluster
                ),
                f
            )
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

    
    
    def _encode_categorical(self, df, fit=True):
        ''''Codificación de variables categóricas'''
        df = df.copy()
        categorical_columns = df.select_dtypes(include=["object", "bool"]).columns
        
        
        for col in categorical_columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))

        return df

    # ------------------------------------------------------
    # LIMPIEZA Y PREPROCESADO
    # ------------------------------------------------------
    def _preprocess(self, df, fit=True):
        # eliminar duplicados
        df = df.copy().drop_duplicates()
    
        # eliminar col poco correlacionadas con Revenue
        drop_cols = ["OperatingSystems", "Browser"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
        num_cols = [
            "Administrative","Administrative_Duration",
            "Informational","Informational_Duration",
            "ProductRelated","ProductRelated_Duration",
            "BounceRates","ExitRates","PageValues"
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
    
        if "SpecialDay" in df.columns:
            df["SpecialDay"] = df["SpecialDay"].fillna(0)
    
        if "VisitorType" in df.columns:
            df["VisitorType"] = (
                df["VisitorType"]
                .astype(str)
                .str.replace(" ", "_")
                .str.replace("-", "_")
                .str.strip()
            )
            visitor_map = {
                "Returning_Visitor": "Returning_Visitor",
                "New_Visitor": "New_Visitor",
                "Other": "Other",
                "returning_visitor": "Returning_Visitor",
                "new_visitor": "New_Visitor"
            }
            df["VisitorType"] = df["VisitorType"].replace(visitor_map).fillna("Other")
    
        if "Month" in df.columns:
            df["Month"] = df["Month"].astype(str).str.strip()
            df["Month"] = df["Month"].apply(self._correct_month)
    
        for col in ["Weekend", "Revenue"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)
    
        for col in ["BounceRates", "ExitRates"]:
            if col in df.columns:
                df[col] = df[col].clip(0,1)
    
        for col in ["Administrative_Duration","Informational_Duration","ProductRelated_Duration"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
    
        # One-Hot Encoding (Month, VisitorType, Region, TrafficType)
        cat_cols = []
        for c in ["Month", "VisitorType", "Region", "TrafficType"]:
            if c in df.columns:
                cat_cols.append(c)
    
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
        # Escalado de variables numéricas
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Revenue"]
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            # Guardamos columnas finales para test
            self.feature_columns = df.columns.drop("Revenue")
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
            # Reindexamos para que coincidan con columnas de entrenamiento
            df = df.reindex(columns=list(self.feature_columns) + ["Revenue"], fill_value=0)
    
        return df


    # ------------------------------------------------------
    # CORRECCIÓN DE MESES
    # ------------------------------------------------------
    def _correct_month(self, val):
        valid_months_full = {
            "january": "Jan","february": "Feb","march": "Mar","april": "Apr",
            "may": "May","june": "Jun","july": "Jul","august": "Aug",
            "september": "Sep","october": "Oct","november": "Nov","december": "Dec"
        }
        val_clean = val.strip().lower()
        match = difflib.get_close_matches(val_clean, list(valid_months_full.keys()), n=1, cutoff=0.0)
        return valid_months_full[match[0]] if match else val_clean[:3].title()


    def train_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        print("==== Métricas de validación ====")
        print("Tasa de error:", 1 - accuracy_score(y_val, y_pred))
        print("Precisión:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1-score:", f1_score(y_val, y_pred))


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

   
    # ------------------------------------------------------
    # CLUSTERING SEPARADO POR REVENUE + FEATURE ENGINEERING
    # ------------------------------------------------------
    def cluster_data(self, file_path, k_true=4, k_false=4, save_excel=True):
        df = pd.read_csv(file_path)
    
        num_cols = [
            'Administrative','Administrative_Duration',
            'Informational','Informational_Duration',
            'ProductRelated','ProductRelated_Duration',
            'BounceRates','ExitRates','PageValues','SpecialDay'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
    
        cat_cols = ['Month', 'VisitorType', 'Weekend',
                    'OperatingSystems','Browser','Region','TrafficType']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
    
        numeric_transformer = Pipeline([('scaler', StandardScaler())])
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    
        self.preprocessor_cluster = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, [c for c in num_cols if c in df.columns]),
                ('cat', categorical_transformer, [c for c in cat_cols if c in df.columns])
            ]
        )

        def cluster_group(df_group, k, label):
            X = self.preprocessor_cluster.fit_transform(df_group)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            sub_labels = kmeans.fit_predict(X)
    
            df_group['cluster_sub'] = sub_labels
            df_group['cluster_global'] = label
            df_group['cluster_id'] = df_group['cluster_global'] + "_" + df_group['cluster_sub'].astype(str)
    
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_dense)
            df_group["PC1"] = X_pca[:, 0]
            df_group["PC2"] = X_pca[:, 1]
    
            plt.figure(figsize=(8, 6))
            for c in range(k):
                pts = X_pca[sub_labels == c]
                plt.scatter(pts[:, 0], pts[:, 1], alpha=0.6, label=f'Subcluster {c}')
            plt.title(f'PCA – {label}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.show()
    
            return df_group
    
        df_true = cluster_group(df[df["Revenue"] == True].copy(), k_true, "Revenue_True")
        df_false = cluster_group(df[df["Revenue"] == False].copy(), k_false, "Revenue_False")
    
        df_out = pd.concat([df_true, df_false]).sort_index()
    
        if save_excel:
            df_out.to_excel("database_clusters.xlsx", index=False)
            print("\nArchivo generado: database_clusters.xlsx")
    
        return df_out

      
    
model = APS_Solver() 
model.train_model("online_shoppers_train.csv")
model.test_model("online_shoppers_test.csv") 

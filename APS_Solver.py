import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
        self.preprocessor_cluster = None   # Nuevo: para clustering

    # ------------------------------------------------------
    # ENCODER CATEGÓRICAS (TRAIN/TEST)
    # ------------------------------------------------------
    def _encode_categorical(self, df, fit=True):
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
    # PREPROCESADO (ESCALADO + LIMPIEZA)
    # ------------------------------------------------------
    def _preprocess(self, df, fit=True):
        df = df.copy()
        df = df.drop_duplicates()

        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        cat_cols = df.select_dtypes(include=['object', 'bool']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        df = self._encode_categorical(df, fit)

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Revenue"]
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    # ------------------------------------------------------
    # ENTRENAR MODELO
    # ------------------------------------------------------
    def train_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)

        print("Tasa de error:", 1 - accuracy_score(y_val, y_pred))
        print("Precisión:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1-score:", f1_score(y_val, y_pred))

    # ------------------------------------------------------
    # TEST FINAL
    # ------------------------------------------------------
    def test_model(self, file_path):
        df = pd.read_csv(file_path)
        df = self._preprocess(df, fit=False)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        y_pred = self.model.predict(X)

        print("Tasa de error:", 1 - accuracy_score(y, y_pred))
        print("Precisión:", precision_score(y, y_pred))
        print("Recall:", recall_score(y, y_pred))
        print("F1-score:", f1_score(y, y_pred))

    # ------------------------------------------------------
    # NUEVO: CLUSTERING SEPARADO POR REVENUE
    # ------------------------------------------------------
    def cluster_data(self, file_path, k_true=4, k_false=4, save_excel=True):
        
        df = pd.read_excel(file_path)

        # Columnas numéricas usadas para clustering:
        num_cols = [
            'Administrative','Administrative_Duration',
            'Informational','Informational_Duration',
            'ProductRelated','ProductRelated_Duration',
            'BounceRates','ExitRates',
            'PageValues','SpecialDay',
            'OperatingSystems','Browser','Region',
            'TrafficType'
        ]

        # Columnas categóricas:
        cat_cols = ['Month', 'VisitorType', 'Weekend']

        # Pipeline de preprocesado para clustering
        numeric_transformer = Pipeline([ ('scaler', StandardScaler()) ])
        categorical_transformer = Pipeline([ ('onehot', OneHotEncoder(handle_unknown='ignore')) ])

        self.preprocessor_cluster = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )

        # Subfunción para aplicar clustering a cada grupo
        def cluster_group(df_group, k, label):
            X = self.preprocessor_cluster.fit_transform(df_group)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            sub_labels = kmeans.fit_predict(X)

            df_group['cluster_sub'] = sub_labels
            df_group['cluster_global'] = label

            X_dense = X.toarray() if hasattr(X, "toarray") else X
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_dense)
            df_group["PC1"] = X_pca[:,0]
            df_group["PC2"] = X_pca[:,1]

            plt.figure(figsize=(8,6))
            for c in range(k):
                pts = X_pca[sub_labels == c]
                plt.scatter(pts[:,0], pts[:,1], alpha=0.6, label=f'Subcluster {c}')
            plt.title(f'PCA – {label}')
            plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
            plt.show()

            return df_group

        df_true = cluster_group(df[df["Revenue"] == True].copy(),  k_true,  "Revenue_True")
        df_false = cluster_group(df[df["Revenue"] == False].copy(), k_false, "Revenue_False")

        df_out = pd.concat([df_true, df_false]).sort_index()

        if save_excel:
            df_out.to_excel("database_clusters.xlsx", index=False)
            print("\nArchivo generado: database_clusters.xlsx")

        return df_out


# ============================================================
#  USO
# ============================================================

df = pd.read_csv("online_shoppers_intention.csv")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Revenue"])
train_df.to_csv("online_shoppers_train.csv", index=False)
test_df.to_csv("online_shoppers_test.csv", index=False)

model = APS_Solver()
print('Para el train:')
model.train_model("online_shoppers_train.csv")
print('\nPara el test:')
model.test_model("online_shoppers_test.csv")

# Llamada al clustering (usa archivo Excel original)
model.cluster_data("database.xlsx", k_true=4, k_false=4)

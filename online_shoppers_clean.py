import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class APS_Solver:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    # ----------------- MÉTODO PRINCIPAL -----------------
    def clean_dataset(self, save_path="online_shoppers_clean.csv"):
        self._load_data()
        self._drop_columns(["OperatingSystems", "Browser", "Region", "TrafficType"])
        self._impute_nulls()
        self._clean_categorical()
        self._correct_types()
        self._correct_ranges()
        self._one_hot_encode(["VisitorType"])
        # Nota: Ya no guardamos aquí, lo haremos después de clustering
        print("Limpieza completada. El DataFrame limpio está listo para clustering.")

    # ----------------- MÉTODOS PRIVADOS -----------------
    def _load_data(self):
        self.df = pd.read_csv(self.filepath)

    def _drop_columns(self, cols):
        self.df = self.df.drop(columns=[c for c in cols if c in self.df.columns])
        self.df = self.df.drop_duplicates()

    def _impute_nulls(self):
        num_cols = [
            "Administrative", "Administrative_Duration",
            "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues"
        ]
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        if "SpecialDay" in self.df.columns:
            self.df["SpecialDay"] = self.df["SpecialDay"].fillna(0)

    def _clean_categorical(self):
        if "Month" in self.df.columns:
            self.df["Month"] = self.df["Month"].str.strip().str.capitalize().str[:3]
            valid_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            self.df.loc[~self.df["Month"].isin(valid_months), "Month"] = "Unknown"

        if "VisitorType" in self.df.columns:
            self.df["VisitorType"] = (
                self.df["VisitorType"]
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
            self.df["VisitorType"] = self.df["VisitorType"].replace(visitor_map).fillna("Other")

    def _correct_types(self):
        for col in ["Weekend", "Revenue"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool)

    def _correct_ranges(self):
        for col in ["BounceRates", "ExitRates"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(0, 1)

        for col in ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(lower=0)

    def _one_hot_encode(self, cols):
        cols = [c for c in cols if c in self.df.columns]
        if cols:
            self.df = pd.get_dummies(self.df, columns=cols, drop_first=True)

    # ----------------- KMEANS -----------------
    def kmeans_one(self):

        df = self.df.copy()

        num_cols = [
            'Administrative','Administrative_Duration',
            'Informational','Informational_Duration',
            'ProductRelated','ProductRelated_Duration',
            'BounceRates','ExitRates',
            'PageValues','SpecialDay'
        ]

        cat_cols = [c for c in ["Month", "Weekend"] if c in df.columns]

        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )

        def cluster_group(df_group, k, label):
            print(f"\n=== Clustering grupo: {label} (n={len(df_group)}) ===")

            X = preprocessor.fit_transform(df_group)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            sub_labels = kmeans.fit_predict(X)

            df_group.loc[:, 'cluster_sub'] = sub_labels
            df_group.loc[:, 'cluster_global'] = label

            # PCA para visualización
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_dense)

            df_group.loc[:, "PC1"] = X_pca[:, 0]
            df_group.loc[:, "PC2"] = X_pca[:, 1]

            # Plot
            plt.figure(figsize=(8, 6))
            for c in range(k):
                pts = X_pca[sub_labels == c]
                plt.scatter(pts[:, 0], pts[:, 1], alpha=0.6, label=f'Subcluster {c}')

            plt.title(f'PCA – {label}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"PCA_{label}.png")  # Guarda la imagen
            plt.show()  # Muestra la figura en Spyder
            plt.close()  # Cierra la figura para no saturar memoria

            summary = df_group.groupby('cluster_sub').agg({
                'Administrative': 'mean',
                'Informational': 'mean',
                'ProductRelated': 'mean',
                'BounceRates': 'mean',
                'ExitRates': 'mean',
                'PageValues': 'mean',
                'SpecialDay': 'mean',
                'cluster_sub': 'count'
            }).rename(columns={'cluster_sub': 'count'}).round(3)

            print(f"\nResumen subclusters para {label}:")
            print(summary)

            return df_group

        # Ejecutar para Revenue True y False
        df_true = cluster_group(df[df["Revenue"] == True].copy(), 3, "Revenue_True")
        df_false = cluster_group(df[df["Revenue"] == False].copy(), 3, "Revenue_False")

        df_out = pd.concat([df_true, df_false]).sort_index()

        # Agregar las columnas de cluster al DataFrame limpio original
        self.df['cluster_sub'] = df_out['cluster_sub']
        self.df['cluster_global'] = df_out['cluster_global']

        # Guardar el documento limpio con las nuevas columnas
        self.df.to_csv("online_shoppers_clean.csv", index=False)
        print("ARCHIVO LIMPIO ACTUALIZADO CON CLUSTERS: online_shoppers_clean.csv")

        # También guardar el archivo separado si lo deseas (opcional)
        df_out.to_csv("database_clusters.csv", index=False)
        print("ARCHIVO FINAL GENERADO: database_clusters.csv")


# ----------------- USO -----------------
solver = APS_Solver("online_shoppers_forStudents.csv")
solver.clean_dataset()
# El DataFrame limpio ahora está en solver.df
# Para hacer clustering:
solver.kmeans_one()

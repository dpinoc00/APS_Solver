import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class APS_Solver:

    def __init__(self):
        self.pipeline = None


    # ------------------------------------------
    # LIMPIEZA BÁSICA UTILIZADA EN TRAIN Y TEST
    # ------------------------------------------
    def _clean_data(self, df):

        # Eliminar duplicados
        df = df.drop_duplicates()

        # Separar numéricas y categóricas
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns

        # Rellenar nulos → MEDIA / MODA
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df


    # ------------------------------------------
    # LOAD MODEL
    # ------------------------------------------
    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)


    # ------------------------------------------
    # SAVE MODEL
    # ------------------------------------------
    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump(self.pipeline, f)


    # ------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------
    def train_model(self, file_path):

        df = pd.read_csv(file_path)

        # Limpieza homogénea
        df = self._clean_data(df)

        # Separar X e y
        X = df.drop("Revenue", axis=1)
        y = df["Revenue"]

        # Tipos de columnas
        numerical = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()

        # Preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical),
                ('cat', OneHotEncoder(handle_unknown="ignore"), categorical)
            ]
        )

        # Modelo
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )

        # Pipeline completo
        self.pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('classifier', model)
        ])

        # Train/test interno
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Entrenamiento
        self.pipeline.fit(X_train, y_train)

        # Validación interna
        y_pred = self.pipeline.predict(X_val)

        print("\n--- VALIDACIÓN INTERNA ---")
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print("Precision:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1:", f1_score(y_val, y_pred))


    # ------------------------------------------
    # TEST MODEL
    # ------------------------------------------
    def test_model(self, file_path):

        df_test = pd.read_csv(file_path)

        # MISMA limpieza exacta que en entrenamiento
        df_test = self._clean_data(df_test)

        X_test = df_test.drop("Revenue", axis=1)
        y_test = df_test["Revenue"]

        # Predicción
        y_pred = self.pipeline.predict(X_test)

        # Métricas requeridas
        error_rate = 1 - accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n--- RESULTADOS TEST ---")
        print("Tasa de error:", error_rate)
        print("Precisión:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class APS_Solver:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}

    # ------------------------------------------------------
    # CODIFICACIÓN DE VARIABLES CATEGÓRICAS (TRAIN/TEST)
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
    # LIMPIEZA Y PREPROCESADO
    # ------------------------------------------------------
    def _preprocess(self, df, fit=True):

        df = df.copy()

        # Eliminar duplicados
        df = df.drop_duplicates()

        # Rellenar nulos numéricos con la media
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        # Rellenar nulos categóricos con la moda
        cat_cols = df.select_dtypes(include=['object', 'bool']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Codificar categóricas
        df = self._encode_categorical(df, fit)

        # Escalar numéricas
        # Rellenar nulos numéricos
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Revenue"]
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        
        # Escalar numéricas
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Revenue"]
        if fit:
            self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
        
        return df 

    # ------------------------------------------------------
    # CARGAR MODELO
    # ------------------------------------------------------
    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            self.model, self.scaler, self.label_encoders = pickle.load(f)

    # ------------------------------------------------------
    # GUARDAR MODELO
    # ------------------------------------------------------
    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump((self.model, self.scaler, self.label_encoders), f)

    # ------------------------------------------------------
    # ENTRENAMIENTO DEL MODELO
    # ------------------------------------------------------
    def train_model(self, file_path):

        df = pd.read_csv(file_path)

        # Preprocesado completo
        df = self._preprocess(df, fit=True)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        # Partición train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Validación interna
        y_pred = self.model.predict(X_val)

        error_rate = 1 - accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        print("Tasa de error:", error_rate)
        print("Precisión:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

    # ------------------------------------------------------
    # TEST FINAL
    # ------------------------------------------------------
    def test_model(self, file_path):

        df = pd.read_csv(file_path)

        df = self._preprocess(df, fit=False)

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        y_pred = self.model.predict(X)

        error_rate = 1 - accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print("Tasa de error:", error_rate)
        print("Precisión:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)


df = pd.read_csv("online_shoppers_intention.csv")

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Revenue"])

train_df.to_csv("online_shoppers_train.csv", index=False)
test_df.to_csv("online_shoppers_test.csv", index=False)


model = APS_Solver()
print('Para el train:')
model.train_model("online_shoppers_train.csv")
print('\nPara el test:')
model.test_model("online_shoppers_test.csv")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class APS_Solver:

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.pipeline = None

    def _nulos_duplicados(self):        
        df = pd.read_csv("online_shoppers_intention_clean.csv")
        
        print("Valores nulos por columna:")
        print(df.isnull().sum())

        print("\nFilas duplicadas:", df.duplicated().sum())
        
        df = df.drop_duplicates()
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        
        # --- 3. Rellenar valores nulos categóricos con la moda ---
        cat_cols = df.select_dtypes(include=['object', 'bool']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        df.to_csv("online_shoppers_intention_clean.csv", index=False)
        
        print("Tratamiento completado. Archivo guardado como 'online_shoppers_intention_clean.csv'.")
        
        # Buscar caracteres inusuales en columnas tipo texto
        for col in cat_cols:
            print(f"\nRevisando columna: {col}")
            mask = df[col].astype(str).str.contains(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ .,_\-\/]", na=False)
            if mask.sum() > 0:
                print(df.loc[mask, col].unique())
            else:
                print("Sin caracteres extraños")
   

    # Cargar modelo
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model, self.scaler, self.label_encoders = pickle.load(f)


    # Guardar modelo
    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump((self.model, self.scaler, self.label_encoders), f)

    # Procesamiento general
    def _prepare_data(self, df):

        # Limpieza básica
        df = df.dropna()

        # Separar X e y si existe la columna Revenue
        if "Revenue" in df.columns:
            X = df.drop("Revenue", axis=1)
            y = df["Revenue"]
        else:
            X = df
            y = None

        # Identificar variables
        numerical = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = X.select_dtypes(include=["object"]).columns.tolist()

        # Preprocesador
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical),
                ('cat', OneHotEncoder(handle_unknown="ignore"), categorical)
            ]
        )

        return X, y
    

    # ENTRENAMIENTO
    def train_model(self, file_path):

        df = pd.read_csv(file_path)
        X, y = self._prepare_data(df)

        # División train/test para validar
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Modelo principal
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        )

        # Pipeline completo
        self.pipeline = Pipeline(steps=[
            ('preprocess', self.preprocessor),
            ('classifier', model)
        ])

        # Entrenar
        self.pipeline.fit(X_train, y_train)

        # Validación interna
        y_pred = self.pipeline.predict(X_val)
        print("\n--- VALIDACIÓN INTERNA ---")
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print("Precision:", precision_score(y_val, y_pred))
        print("Recall:", recall_score(y_val, y_pred))
        print("F1:", f1_score(y_val, y_pred))

        self.model = model


   #TEST FINAL
    def test_model(self, file_path):

        df_test = pd.read_csv(file_path)

        # Preparar datos (pero SIN reentrenar preprocesador)
        # Usar columnas necesarias
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

if __name__ == "__main__":
    # Ejemplo de uso (descomentar y adaptar paths si corres localmente)
    # solver = APS_Solver()
    # solver.train_model("online_shoppers_train.csv", do_gridsearch=False)
    # solver.test_model("online_shoppers_test.csv")
    pass

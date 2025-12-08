import pandas as pd
import numpy as np

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
        self._one_hot_encode(["VisitorType"])  # Solo VisitorType
        self.df.to_csv(save_path, index=False)
        print(f"Limpieza completada. Archivo generado: {save_path}")

    # ----------------- MÉTODOS PRIVADOS -----------------
    def _load_data(self):
        self.df = pd.read_csv(self.filepath)

    def _drop_columns(self, cols):
        self.df = self.df.drop(columns=[c for c in cols if c in self.df.columns])

    def _impute_nulls(self):
        # Columnas numéricas imputadas con mediana
        num_cols = [
            "Administrative", "Administrative_Duration",
            "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues"
        ]
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        # SpecialDay imputado con 0
        if "SpecialDay" in self.df.columns:
            self.df["SpecialDay"] = self.df["SpecialDay"].fillna(0)

    def _clean_categorical(self):
        # ------------------ Month ------------------
        if "Month" in self.df.columns:
            self.df["Month"] = self.df["Month"].str.strip().str.capitalize()
            # Solo conservamos las primeras tres letras y corregimos errores
            valid_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            self.df["Month"] = self.df["Month"].str[:3]  # Tomamos primeras 3 letras
            self.df.loc[~self.df["Month"].isin(valid_months), "Month"] = "Unknown"

        # ------------------ VisitorType ------------------
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
        bool_cols = ["Weekend", "Revenue"]
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool)

    def _correct_ranges(self):
        # BounceRates y ExitRates entre 0 y 1
        for col in ["BounceRates", "ExitRates"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(0, 1)
        # Duraciones no negativas
        dur_cols = [
            "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"
        ]
        for col in dur_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(lower=0)

    def _one_hot_encode(self, cols):
        existing_cols = [c for c in cols if c in self.df.columns]
        if existing_cols:
            self.df = pd.get_dummies(self.df, columns=existing_cols, drop_first=True)

        
solver = APS_Solver("online_shoppers_forStudents.csv")
solver.clean_dataset()
# El DataFrame limpio ahora está en solver.df

import pandas as pd

# Cargar dataset
df = pd.read_csv("online_shoppers_intention_clean.csv")

# Mostrar cantidad total de nulos por columna
print("Valores nulos por columna:")
print(df.isnull().sum())

# Detectar filas duplicadas
print("\nFilas duplicadas:", df.duplicated().sum())

df = df.drop_duplicates()

# --- 2. Rellenar valores nulos numéricos con la media ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# --- 3. Rellenar valores nulos categóricos con la moda ---
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv("online_shoppers_intention_clean.csv", index=False)

print("Tratamiento completado. Archivo guardado como 'online_shoppers_intention_clean.csv'.")



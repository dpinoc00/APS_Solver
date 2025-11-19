# APS_Solver
## Motivación:
Satisfacción de Usuarios en Compras Online Motivación En este trabajo utilizaremos un conjunto de datos disponible en UCI Machine Learning Repository sobre la intención de compra de los usuarios en plataformas de comercio electrónico. Este dataset fue recopilado para el análisis de comportamiento de compra de los usuarios en línea y es una herramienta valiosa para desarrollar modelos predictivos en el contexto del marketing y la experiencia del cliente. 
## La historia detrás de este conjunto de datos es la siguiente: 
Imagina que trabajas como analista de datos en una tienda en línea internacional que ha notado un comportamiento de compra fluctuante en sus usuarios. Aunque muchos usuarios navegan por el sitio web y agregan productos al carrito, un porcentaje significativo de ellos no finaliza la compra, lo que genera una pérdida potencial de ventas. La empresa ha decidido recopilar datos detallados sobre el comportamiento de los usuarios en su plataforma con el fin de entender mejor las razones. 

El reto para la tienda en línea es claro: el mercado de comercio electrónico es extremadamente competitivo, y la fidelización de los clientes está cada vez más complicada. Los compradores de hoy buscan una experiencia de usuario rápida, conveniente y sin fricciones, pero las razones por las cuales abandonan sus compras no son siempre obvias. ¿Es la velocidad del sitio web? ¿El diseño de la página? ¿La disponibilidad de productos? ¿Las recomendaciones de productos o las ofertas de descuento? 

Para abordar este problema, la tienda ha recopilado una cantidad significativa de datos sobre el comportamiento de los usuarios en el sitio web, que incluyen información sobre las visitas, el t iempo de permanencia, la interacción con las páginas de productos y las características de los usuarios, como la duración de la sesión y la hora del día en que accedieron a la tienda. Sin embargo, el equipo de marketing y ventas no tiene claridad sobre qué factores están influyendo realmente en la intención de compra, y cómo pueden mejorar la experiencia del usuario para aumentar la tasa de conversión. Tu misión como analista de datos es ayudar a la tienda a entender estos patrones. Deberás  desarrollar un modelo predictivo que permita identificar si un usuario tiene alta o baja probabilidad de realizar una compra, basándose en su comportamiento y características demográficas. Con esta información, la tienda podrá ajustar su estrategia de marketing, optimizar la interfaz del sitio y mejorar la experiencia de usuario para incrementar las ventas y la fidelidad de los clientes. 

https://www.kaggle.com/datasets/saadaliyaseen/shopping-behaviour-dataset/code

datos = pd.read_csv("online_shoppers_intention.csv")

num_df = datos.select_dtypes(include="number")
correlation = num_df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation, vmin=-1, vmax=1)
plt.colorbar(label="Correlación")

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Matriz de Correlación (Variables Numéricas)")
plt.tight_layout()
plt.show()

tipo = num_df["Browser"].value_counts()

umbral = 0.02
tipo_rel = tipo / tipo.sum()

otros = tipo_rel[tipo_rel < umbral].sum()
tipo_clean = tipo_rel[tipo_rel >= umbral]

if otros > 0:
    tipo_clean["Otros"] = otros

plt.figure(figsize=(6,6))

colores_pastel = [
    "#d8b4fe",
    "#f9a8d4",
    "#fecdd3", 
    "#fca5a5",
    "#fdb4bf"   
]

plt.pie(
    tipo_clean,
    labels=tipo_clean.index,
    autopct='%1.2f%%',
    startangle=-450,
    pctdistance=0.8,
    labeldistance=1.1,
    colors=colores_pastel[:len(tipo_clean)]
)

plt.title("Browsers used")
plt.tight_layout()
plt.show()


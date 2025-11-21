# APS_Solver
## Motivación:
Satisfacción de Usuarios en Compras Online Motivación En este trabajo utilizaremos un conjunto de datos disponible en UCI Machine Learning Repository sobre la intención de compra de los usuarios en plataformas de comercio electrónico. Este dataset fue recopilado para el análisis de comportamiento de compra de los usuarios en línea y es una herramienta valiosa para desarrollar modelos predictivos en el contexto del marketing y la experiencia del cliente. 
## La historia detrás de este conjunto de datos es la siguiente: 
Imagina que trabajas como analista de datos en una tienda en línea internacional que ha notado un comportamiento de compra fluctuante en sus usuarios. Aunque muchos usuarios navegan por el sitio web y agregan productos al carrito, un porcentaje significativo de ellos no finaliza la compra, lo que genera una pérdida potencial de ventas. La empresa ha decidido recopilar datos detallados sobre el comportamiento de los usuarios en su plataforma con el fin de entender mejor las razones. 

El reto para la tienda en línea es claro: el mercado de comercio electrónico es extremadamente competitivo, y la fidelización de los clientes está cada vez más complicada. Los compradores de hoy buscan una experiencia de usuario rápida, conveniente y sin fricciones, pero las razones por las cuales abandonan sus compras no son siempre obvias. ¿Es la velocidad del sitio web? ¿El diseño de la página? ¿La disponibilidad de productos? ¿Las recomendaciones de productos o las ofertas de descuento? 

Para abordar este problema, la tienda ha recopilado una cantidad significativa de datos sobre el comportamiento de los usuarios en el sitio web, que incluyen información sobre las visitas, el t iempo de permanencia, la interacción con las páginas de productos y las características de los usuarios, como la duración de la sesión y la hora del día en que accedieron a la tienda. Sin embargo, el equipo de marketing y ventas no tiene claridad sobre qué factores están influyendo realmente en la intención de compra, y cómo pueden mejorar la experiencia del usuario para aumentar la tasa de conversión. Tu misión como analista de datos es ayudar a la tienda a entender estos patrones. Deberás  desarrollar un modelo predictivo que permita identificar si un usuario tiene alta o baja probabilidad de realizar una compra, basándose en su comportamiento y características demográficas. Con esta información, la tienda podrá ajustar su estrategia de marketing, optimizar la interfaz del sitio y mejorar la experiencia de usuario para incrementar las ventas y la fidelidad de los clientes. 


Cómo introducirlo

“Este gráfico representa cómo se relaciona el número de páginas informativas visitadas por un usuario (‘Informational’) con el tiempo total que pasa en esa sección (‘Informational Duration’).”

Qué muestra el eje X y el eje Y

Eje X – Informational Duration:
Indica cuántos segundos permaneció el usuario navegando en la sección informativa.

Eje Y – Informational:
Muestra cuántas páginas informativas visitó el usuario durante esa sesión.

Cómo interpretar la curva

Puedes explicar algo como:

“La línea muestra cómo cambia la cantidad de páginas consultadas a medida que aumenta la duración en la sección informativa.”

“Si la curva tiene tendencia ascendente, significa que cuanto más tiempo pasa el usuario en el área informativa, más páginas visita.”

“Si hay tramos planos o irregulares, podría indicar que algunos usuarios pasan mucho tiempo sin navegar por muchas páginas o viceversa.”




grupo = datos.groupby("VisitorType")[["Administrative",
                                      "Informational",
                                      "ProductRelated"]].mean()

grupo = grupo.loc[["New_Visitor", "Returning_Visitor"]]
plt.figure(figsize=(10,6))

x = range(len(grupo.columns))

new_vals = grupo.loc["New_Visitor"]
ret_vals = grupo.loc["Returning_Visitor"]

width = 0.35

plt.bar([i - width/2 for i in x], new_vals, width=width,
        label="New Visitor", color="#a78bfa")
plt.bar([i + width/2 for i in x], ret_vals, width=width,
        label="Returning Visitor", color="#f9a8d4")  

plt.xticks(x, ["Administrative", "Informational", "ProductRelated"])
plt.ylabel("Páginas visitadas (promedio)")
plt.title("Páginas visitadas por tipo de visitante")
plt.legend()

plt.tight_layout()
plt.show()



1) Promedio mensual de las interacciones de los usuarios en las categorías ProductRelated, Informational y Administrative. Compara el comportamiento de los usuarios en cada tipo de actividad por mes. El gráfico muestra que los meses donde los usuarios más entran a la pagina web son enero y noviembre y que la sección de productos de la página web es la que más paginas visitadas tiene.

3) Heatmap: matriz de correlación. Muestra la fuerza que tienen las relaciones entre las diferentes variables para saber el peso con el que influye una en otra. El gráfico muestra que BounceRates está muy relacionada con ExitRates con un 0.91, ProductRelated y ProductRelated_Duration también tienen gran peso el uno en el otro. Los que menos correlación tienen son el número de páginas visitadas en la sección administrativa con la tasa de salida lo que indica que en la sección administrativa los usuarios no tienden a salir de la página.


4)Informational

Muestra la relación entre el tiempo en la sección informativa y la cantidad de páginas informativas visitadas, con los datos ordenados por duración. Grafica la evolución de la navegación informativa a medida que pasa el tiempo.
Cuando la línea es ascendente, indica que a mayor tiempo, más páginas consultadas; mientras que tramos planos o irregulares corresponden a usuarios que dedican tiempo sin recorrer muchas páginas.

5)Administrative

Permite analizar ela cantidad de páginas administrativas visitadas en función del tiempo dedicado, ordena los registros por duración. 
Cada curva ascendente señala que más tiempo implica mayor número de páginas visitadas; irregularidades o segmentos planos señalan diferencias en cómo los usuarios web emplean el tiempo en esta sección.



11)Gráfico de barras: Revenue a lo largo de los meses
El gráfico muestra cómo varía la proporción de usuarios que completan o no la compra (Revenue) a lo largo de los diferentes meses del año.
Se pueden observar fluctuaciones mensuales en la tasa de conversión, con algunos meses mostrando un mayor número de usuarios que finalizan la compra y otros donde la conversión es menor.
Estas variaciones podrían estar relacionadas con factores como promociones, eventos especiales, o cambios en el comportamiento de los usuarios. Por ejemplo, noviembre es el mes en el que más usuarios finalizan la compra lo que estaría relacionado con la promoción anual de “Black Friday”.



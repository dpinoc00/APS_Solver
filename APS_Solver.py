import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("online_shoppers_intention.csv", usecols=[2,3], names=["ExitRates","Month"], header=0)
print(data.head())

plt.xlabel("Iteraciones")
plt.ylabel("Error MSE")
plt.title("Evolución del error")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Sensor_Color_Distribution.csv')

probabilidades = {
    'R': data['R'].values,
    'O': data['O'].values,
    'Y': data['Y'].values,
    'G': data['G'].values,
    'B': data['B'].values
}

matriz_probabilidades = np.full((10, 10), 1/100)

def actualizar_matriz(evidencia):
    global matriz_probabilidades
    for i in range(10):
        for j in range(10):
            color = evidencia[i, j]
            distancia = abs(i - 5) + abs(j - 5)
            matriz_probabilidades[i, j] *= probabilidades[color][distancia]
    matriz_probabilidades /= np.sum(matriz_probabilidades)

def visualizar_matriz():
    plt.imshow(matriz_probabilidades, cmap='hot', interpolation='nearest')
    plt.show()

def encontrar_celda_mas_probable():
    return np.unravel_index(np.argmax(matriz_probabilidades), matriz_probabilidades.shape)

evidencia = np.random.choice(['R', 'O', 'Y', 'G', 'B'], size=(10, 10))
actualizar_matriz(evidencia)
print('La celda más probable es:', encontrar_celda_mas_probable())
visualizar_matriz()
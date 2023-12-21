# %% [markdown]
# ### Importacion de librerias

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Lectura de dataset

# %%
dataset_clientes = pd.read_csv('dataset_prueba.csv')
print('Cantidad de clientes:',len(dataset_clientes['mobile_number'].unique()))
print('Cantidad de registros (cliente * 3 meses):',len(dataset_clientes))
print('Cantidad de clientes que abandonaron al menos 1 vez:', len(dataset_clientes[dataset_clientes['churn']==1].groupby('mobile_number')['arpu'].count()))
print('Cantidad de abandonos en los ultimos 3 meses:', len(dataset_clientes[dataset_clientes['churn']==1]),'\n')
#7000842753, 7001865778, 7001625959, 7000087541, 7000498689, 7001905007
print('El objetivo es: El objetivo de esta prueba es analizar el churn de los clientes')
print('con respecto a otras variables e intentar predecirlo en el último mes disponible')

# %%
print(dataset_clientes.columns)

# %% [markdown]
# ### Creacion de clasificacion_clientes_revenue

# %%
q1 = np.percentile(dataset_clientes['arpu'], 25)
q2 = np.percentile(dataset_clientes['arpu'], 50)
q3 = np.percentile(dataset_clientes['arpu'], 75)

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

plt.boxplot(dataset_clientes['arpu'], showfliers=False)

plt.text(1, q1, f'Q1: {q1:.2f}', va='center', ha='left', bbox=dict(facecolor='white', alpha=0.9))
plt.text(1, q2, f'Q2: {q2:.2f}', va='center', ha='left', bbox=dict(facecolor='white', alpha=0.9))
plt.text(1, q3, f'Q3: {q3:.2f}', va='center', ha='left', bbox=dict(facecolor='white', alpha=0.9))

plt.text(1, lower_bound, f'Lower Bound: {lower_bound:.2f}', va='center', ha='left', bbox=dict(facecolor='white', alpha=0.9))
plt.text(1, upper_bound, f'Upper Bound: {upper_bound:.2f}', va='center', ha='left', bbox=dict(facecolor='white', alpha=0.9))

plt.ylim(-450, 1000)
plt.title('Average revenue per user')
plt.show()

# %%
deciles = list(dataset_clientes['arpu'].quantile([0.8, 0.9, 1.0]))
print('Deciles 0.8, 0.9, 1.0:', deciles)

def asignar_clasificacion(arpu):
    if arpu >= deciles[1]:
        return 'platino'
    elif arpu >= deciles[0]:
        return 'gold'
    else:
        return 'normal'

dataset_clientes['clasificacion_clientes_revenue'] = dataset_clientes['arpu'].apply(asignar_clasificacion)
print('\nClasificacion de clientes segun su revenue:')
print(dataset_clientes.groupby('clasificacion_clientes_revenue')['arpu'].count())

# %% [markdown]
# ### Creacion de flag_recarga

# %%
def flag_recarga(total_rech_num):
    if total_rech_num != 0:
        return 1
    else:
        return 0

dataset_clientes['flag_recarga'] = dataset_clientes['total_rech_num'].apply(flag_recarga)
print(dataset_clientes.groupby(['flag_recarga'])['arpu'].count())

print('Proporción entre 0 y 1 por mes:\n')
print(dataset_clientes.groupby(['last_date_of_month','flag_recarga'])['arpu'].count())
print('\nProporciones\nJunio:', 1607/98392)
print('Julio:', 1166/98232)
print('Agosto:', 2522/96377)

# TODO Nunca escriban valores a mano si ya los calculaste de antes. Usa esos cálculos y variables para reciclar esa info

# %% [markdown]
# ### Columnas con mas del 70% en null

# %%
porcentaje_nan = pd.DataFrame((dataset_clientes.isna().sum()/len(dataset_clientes))*100, columns=['Porcentaje']).reset_index()
col_delete = []

print('Columnas a eliminar:\n')
for i in porcentaje_nan.index:
    if porcentaje_nan.iloc[i, 1] > 70:
        col_delete.append(porcentaje_nan.iloc[i, 0])
        print(porcentaje_nan.iloc[i, 0])

dataset_clientes_2 = dataset_clientes.drop(col_delete, axis=1)
print('\nLa cantidad de columnas eliminadas es', len(col_delete))
print('Finalmente, quedarían', len(dataset_clientes_2.columns), 'columnas con menos del 70% de NaN')

# %%
print('Otros porcentaje de NaN:')
print(porcentaje_nan[(porcentaje_nan['Porcentaje']<70)&((porcentaje_nan['Porcentaje']!=0))]['Porcentaje'].unique())
print('No eliminaria otras columnas, pues los otros porcentajes de NaN son bajos, por lo que, podrían')
print('imputarse esos NaN con la media u otro valor')

# %% [markdown]
# ### Relacion churn vs total_rech_num

# %%
plt.hist(dataset_clientes_2[dataset_clientes_2['churn']==0]['total_rech_num'] ,bins=100, color='red', label='No churn', alpha=0.5, density=True)
plt.hist(dataset_clientes_2[dataset_clientes_2['churn']==1]['total_rech_num'] ,bins=50, color='blue', label='Churn', alpha=0.5, density=True)

plt.title('Distribución de la cantidad  de  recargas totales por mes')
plt.xlabel('Valor de total_rech_num')
plt.ylabel('Densidad de Probabilidad')

plt.legend()
plt.ylim(0, 0.4)
plt.xlim(0, 50)
plt.show()

print('Analisis')
print('Del grafico se puede observar que las personas que hacen churn tienen el promedio de recargas mas bajo y')
print('en su mayoria (>80%) hacen menos de 3 recargas. Por otro lado, los que no hacen churn tienen mayor cantidad')
print('de recargas en un mes. Conclusion: si en un mes un cliente tiene poca cantidad de recargas, podría haber una')
print('una tendencia a que haga churn')

# %%
dataset_clientes_2.to_csv('dataset_clientes.csv', index=False)



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
dataset_clientes = pd.read_csv('dataset_clientes.csv')

# %% [markdown]
# ### Ingenieria de datos

# %%
#Pasamos a variables dummies las variables categoricas
dataset_clientes = pd.concat([dataset_clientes, pd.get_dummies(dataset_clientes[['last_date_of_month','clasificacion_clientes_revenue']])], axis=1)
dataset_clientes['last_date_of_month'] = pd.to_datetime(dataset_clientes['last_date_of_month'])
dataset_clientes['date_of_last_rech'] = pd.to_datetime(dataset_clientes['date_of_last_rech'])

#Creamos la variable last_rech_diff basada en la cantidad de dias faltantes para cierre de mes
dataset_clientes['last_rech_diff'] = dataset_clientes['last_date_of_month'] - dataset_clientes['date_of_last_rech']
dataset_clientes['last_rech_diff'] = dataset_clientes['last_rech_diff'].dt.days

#Eliminar las variables categoricas y de fecha
dataset_clientes.drop(['last_date_of_month','date_of_last_rech','clasificacion_clientes_revenue', 'mobile_number'], axis=1, inplace=True)

#Completar los NaN con la media de cada variable
dataset_clientes = dataset_clientes.fillna(dataset_clientes.mean())

# TODO Estás introduciendo data leakage al hacer este promedio sobre todo tu dataset. El dataset de test no se toca. 

# %%
from sklearn.preprocessing import MinMaxScaler

# Crea un objeto MinMaxScaler
scaler = MinMaxScaler()

var_categ = ['churn','flag_recarga','last_date_of_month_6/30/2014','last_date_of_month_7/31/2014','last_date_of_month_8/31/2014',\
             'clasificacion_clientes_revenue_gold','clasificacion_clientes_revenue_normal','clasificacion_clientes_revenue_platino']

# TODO De nuevo, el último día del mes por periodo lo puedes calcular, no lo escribas a mano. Imagina que tienes 1000 meses distintos y tienes que escribir esto con aproximadamente la misma cantidad de caracteres. Y si aumenta a 2000, nada en tu código cambiaría

# Aplica el escalado a los datos
dataset_clientes_scaled = scaler.fit_transform(dataset_clientes[dataset_clientes.columns.drop(var_categ)])

# TODO más data leakage. Estás incluenciando tu dataset de entrenamiento con datos de tu test

dataset_clientes_scaled = pd.DataFrame(dataset_clientes_scaled, columns=dataset_clientes.columns.drop(var_categ))

dataset_clientes = pd.concat([dataset_clientes_scaled, dataset_clientes[var_categ]], axis=1)

# %% [markdown]
# ### Visualizacion de datos

# %%
import seaborn as sns

# Calculamos la correlacion
corr = dataset_clientes[dataset_clientes.columns.drop(var_categ)].corr()

# Crea un mapa de calor
sns.heatmap(corr, cmap="YlGnBu")

# Muestra el mapa de calor
plt.show()

# %%
var_categ = ['flag_recarga','last_date_of_month_6/30/2014','last_date_of_month_7/31/2014','last_date_of_month_8/31/2014',\
             'clasificacion_clientes_revenue_gold','clasificacion_clientes_revenue_normal','clasificacion_clientes_revenue_platino']

# Cramos un Pairplot
sns.pairplot(dataset_clientes[dataset_clientes.columns.drop(var_categ)], hue='churn')

# TODO El código se queda pegado acá. Probablemente tienes un pair plot con MUCHAS columnas. Si sabes que va a demorar, pon un estimado de cuánto uno debería esperar, para saber si es mejor dejarlo ejecutando y hacer algo más mientras. O simplemente reduce la cantidad de variables y quedate con las más relavfantes.

# %% [markdown]
# ### Division de datos

# %%
X = dataset_clientes.drop('churn', axis=1)
y = dataset_clientes['churn']

# %%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# %% [markdown]
# ### Modelamiento de datos

# %% [markdown]
# #### Arbol de decision

# %%
from sklearn.tree import DecisionTreeClassifier
seed = 1994

# Instanciamos un Objeto de la Clase de Árboles de Decisión para Clasificación
dt = DecisionTreeClassifier(criterion='gini', random_state=seed, max_depth=5)
dt.fit(X_train, y_train)

# %%
y_pred_dt = dt.predict(X_val)
y_prob_pred_dt = dt.predict_proba(X_val)[:,1]

# %%
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve

print('Accuracy del Árbol de Decisión en el Dataset de Validación: {:.2f}'.format(accuracy_score(y_val, y_pred_dt)))
print('AUC del Árbol de Decisión en el Dataset de Validación: {:.4f}'.format(roc_auc_score(y_val, y_prob_pred_dt)))
fpr_dt, tpr_dt, _ = roc_curve(y_val, y_prob_pred_dt)

# Calcula la curva de precisión-recuperación
precision_dt, recall_dt, _ = precision_recall_curve(y_val, y_prob_pred_dt)


# %% [markdown]
# #### Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = 4, max_depth = 12, class_weight = 'balanced',
                            min_samples_leaf = 20)
rf.fit(X_train, y_train)

# %%
y_pred_rf = rf.predict(X_val)
y_prob_pred_rf = rf.predict_proba(X_val)[:,1]

# %%
print('Accuracy del Random Forest en el Dataset de Validación: {:.2f}'.format(accuracy_score(y_val, y_pred_rf)))
print('AUC del Random Forest en el Dataset de Validación: {:.4f}'.format(roc_auc_score(y_val, y_prob_pred_rf)))
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_prob_pred_rf)

# Calcula la curva de precisión-recuperación
precision_rf, recall_rf, _ = precision_recall_curve(y_val, y_prob_pred_rf)

# %% [markdown]
# #### XGBoost

# %%
import xgboost as xgb

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val = xgb.DMatrix(X_val, label=y_val)

# %%
param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['eval_metric'] = 'auc'
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
#num_boost_round = 400
num_boost_round = 65


# %%
xgb_model = xgb.train(params=param, dtrain=xgb_train, num_boost_round=num_boost_round, evals=watchlist, verbose_eval=False)

# %%
y_prob_pred_xgb = xgb_model.predict(xgb_val)
print('AUC del XGBoost en el Dataset de Validación: {:.4f}'.format(roc_auc_score(y_val, y_prob_pred_xgb)))
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_prob_pred_xgb)

# Calcula la curva de precisión-recuperación
precision_xgb, recall_xgb, _ = precision_recall_curve(y_val, y_prob_pred_xgb)

# %% [markdown]
# #### Light GBM

# %%
import lightgbm as lgb

rounds = 500
deep = 8
eta = 0.05
seed_val = 0

# %%
params = {}
params["objective"] = "binary"
params['metric'] = 'auc'
params["max_depth"] = deep
params["min_data_in_leaf"] = 20
params["learning_rate"] = eta
params["bagging_fraction"] = 0.7
params["feature_fraction"] = 0.7
params["bagging_freq"] = 5
params["bagging_seed"] = seed_val
params["verbosity"] = 0
num_rounds = rounds
evals_result = {}

lgbm_train = lgb.Dataset(X_train, label=y_train)
lgbm_val = lgb.Dataset(X_val, label=y_val)

# %%
lgb_model = lgb.train(
            params=params,
            train_set=lgbm_train,
            num_boost_round=num_rounds,
            valid_sets=[lgbm_train, lgbm_val],
            #early_stopping_rounds=100,
            #verbose_eval=20,
            #evals_result=evals_result
            )

y_prob_pred_lgb = lgb_model.predict(X_val)

# %%
print('AUC del Light GBM en el Dataset de Validación: {:.4f}'.format(roc_auc_score(y_val, y_prob_pred_lgb)))
fpr_lgb, tpr_lgb, _ = roc_curve(y_val, y_prob_pred_lgb)

# Calcula la curva de precisión-recuperación
precision_lgb, recall_lgb, _ = precision_recall_curve(y_val, y_prob_pred_lgb)

# %% [markdown]
# ### Analisis de resultados

# %% [markdown]
# #### Curva ROC

# %%
from scipy.integrate import trapz

auc_dt = trapz(tpr_dt, fpr_dt)
auc_rf = trapz(tpr_rf, fpr_rf)
auc_xgb = trapz(tpr_xgb, fpr_xgb)
auc_lgb = trapz(tpr_lgb, fpr_lgb)

# %%
#Curvas ROC de los 3 algoritmos probados (DT, RF, XGB, LGBM)
plt.figure(1, figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_dt, tpr_dt, label='Decision Tree', color='blue')
plt.plot(fpr_rf, tpr_rf, label='Random Forest', color='orange')
plt.plot(fpr_xgb, tpr_xgb, label='XGB', color='green')
plt.plot(fpr_lgb, tpr_lgb, label='LGBM', color='red')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')

plt.text(0.83, 0.2, '\n\n\n\n\n                                 ', bbox=dict(facecolor='white', alpha=0.2))
plt.text(0.83, 0.35, f'AUC (DT) = {auc_dt:.4f}', fontsize=9.5, color='blue')
plt.text(0.83, 0.3, f'AUC (RF) = {auc_rf:.4f}', fontsize=9.5, color='orange')
plt.text(0.83, 0.25, f'AUC (XGB) = {auc_xgb:.4f}', fontsize=9.5, color='green')
plt.text(0.83, 0.2, f'AUC (LGBM) = {auc_lgb:.4f}', fontsize=9.5, color='red')

# TODO Mismo problema, recicla los valores, no los escribas

plt.show()

# %% [markdown]
# #### Curca de Precision-Recall

# %%
from sklearn.metrics import auc

auc_dt = auc(recall_dt, precision_dt)
auc_rf = auc(recall_rf, precision_rf)
auc_xgb = auc(recall_xgb, precision_xgb)
auc_lgb = auc(recall_lgb, precision_lgb)

# %%
#Curvas Precision-Recall de los 3 algoritmos probados (DT, RF, XGB, LGBM)
plt.figure(1, figsize=(10, 7))
plt.plot([1, 0], [0, 1], 'k--')
# Gráfico de precisión-recuperación
plt.plot(recall_dt, precision_dt, label='Decision Tree', color='blue')
plt.plot(recall_rf, precision_rf, label='Random Forest', color='orange')
plt.plot(recall_xgb, precision_xgb, label='XGB', color='green')
plt.plot(recall_lgb, precision_lgb, label='LGBM', color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')

plt.text(0.83, 0.65, '\n\n\n\n\n                                 ', bbox=dict(facecolor='white', alpha=0.2))
plt.text(0.83, 0.8, f'AUC (DT) = {auc_dt:.4f}', fontsize=9.5, color='blue')
plt.text(0.83, 0.75, f'AUC (RF) = {auc_rf:.4f}', fontsize=9.5, color='orange')
plt.text(0.83, 0.7, f'AUC (XGB) = {auc_xgb:.4f}', fontsize=9.5, color='green')
plt.text(0.83, 0.65, f'AUC (LGBM) = {auc_lgb:.4f}', fontsize=9.5, color='red')

# TODO Mismo problema, recicla los valores, no los escribas

plt.show()

# %% [markdown]
# #### Feature Importance

# %%
import shap

explainer = shap.Explainer(dt)
dt_shap_values = explainer.shap_values(X_val)
plt.title('Decision Tree Feature Importance')
shap.summary_plot(dt_shap_values, X_val)

explainer = shap.Explainer(rf)
rf_shap_values = explainer.shap_values(X_val)
plt.title('Random Forest Feature Importance')
shap.summary_plot(rf_shap_values, X_val)

explainer = shap.Explainer(xgb_model)
xgb_shap_values = explainer.shap_values(X_val)
xgb_shap_values = [xgb_shap_values, xgb_shap_values]
plt.title('XGBoost Feature Importance')
shap.summary_plot(xgb_shap_values, X_val)

explainer = shap.Explainer(lgb_model)
lgb_shap_values = explainer.shap_values(X_val)
plt.title('LightGBM Feature Importance')
shap.summary_plot(lgb_shap_values, X_val)

# %%
#Explica por qué elegiste tal modelo.  
#a. ¿Qué métricas usaste para evaluar el desempeño del modelo? ¿Por qué? 
#b. ¿Cómo podrías mejorar la performance del modelo? 
#c. ¿Por qué elegiste este modelo? 

# %%
print('Eligiría el modelo LGBM porque tiene alta eficacia para manejar grandes volumenes de datos y de alta dimensionalidad.')
print('Lo anteriormente mencionado se respalda por las graficas de curvas ROC y Precision-Recall. En ambas graficas se observa')
print('que tiene mejor performance.')
print('Para el caso especifico de nuestro modelo de churn, a mi me interesaría minimizar los falsos positivos (clientes identificados')
print('como abandonados pero que no lo hacen), por lo que le dare prioridad a mi grafico de Precision-Recall. Donde se ve que')
print('LGBM tiene mejor rendimiento.')
print('Finalmente, podría mejorar el rendimiento del modelo utilizando mejores tecnicas de pre procesamiento y analizando mas a')
print('detalle la correlacion entre las variables. Para esto es importante notar cuales son las variables mas importantes del modelo.')

# TODO No es necesario usar tantos prints separados. Puedes usar bloques de texto multilinea o simplemente imprimir todo seguido y después verlo en un editor de texto o terminar con las funcionalidades básicas (wrap entre otros)

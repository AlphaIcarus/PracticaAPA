#!/usr/bin/env python
# coding: utf-8

# ## Seismic bumps
# ### Introducción
# 
# El dataset elegido se corresponde con un conjunto de datos relacionados con el sector minero, concretamente fueron obtenidos en una mina de carbón de Polonia. El dataser describe, por cada muestra, una situación en la que se ha dado un evento sísmico, describiendo factores que determinan la fuerza del mismo. 
# 
# En estos casos, la estadística es inefectiva para la predicción de eventos, por lo que se requiere del uso de técnicas más avanzadas.
# 
# A partir de estos datos, se busca determinar, a partir de técnicas de aprendizaje, predecir futuras situaciones, para discernir si son situaciones de peligro o no peligro.
# 
# Veamos cómo se muestran los datos, y qué relaciones existen entre las variables:

# In[1]:


from sklearn.model_selection import train_test_split
import missingno as msno
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
import statsmodels.api as sm
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.regressor import AlphaSelection
from sklearn.model_selection import train_test_split,  KFold, cross_val_score, GridSearchCV
from scipy import stats
# sns.set()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

from sklearn import set_config
import warnings

set_config(display='text')
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 3)

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier,ExtraTreesClassifier

from skopt import BayesSearchCV

from IPython.display import display, HTML
show_html = lambda html: display(HTML(html))


# In[2]:


# Carga de datos
seismic = pd.read_csv("./seismic-bumps.csv", header=0, delimiter = ',')
# Eliminar variables que no son de utilidad para el problema
seismic = seismic.drop(columns=['id'])


# In[3]:


# Descripción de los datos y análisis:
# Visualizacion basica
seismic.head()


# In[4]:


# Mirar medias, desviacion estandar, etc.
seismic.describe().T


# In[5]:


# Cómputo de la frecuencia de valores por variable, en forma de histograma
seismic.loc[:,:].hist(figsize=(20,20));


# In[6]:


# Visualización de las relaciones de las variables con la variable objetivo
g = sns.PairGrid(seismic[:500], diag_sharey=False)    # Reducimos el número de muestras para facilitar el cómputo
g.map_upper(sns.scatterplot);
g.map_lower(sns.kdeplot);
g.map_diag(sns.kdeplot);


# In[7]:


# Matriz de correlación entre la variable objetivo y el resto de variables
corr = seismic.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, cmap='seismic',  center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5});


# ### Descripción de los datos
# 
# El objetivo es predecir si la siguiente estimación tendrá valor class == 0 o class == 1 (no peligro / peligro).
# 
# Observamos que el dataset consta de 19 atributos y 2584 observaciones. Todos los atributos son numéricos menos seismic, seismoacustic y shift, que son categóricos. La variable objetivo es class, que es un atributo booleano que determina si existe peligro (==1) o no (==0).
# 
# Observamos que, a priori, ninguno de ellos consta de una distribución normal. Las variables categóricas (seismic, seismoacustic y shift) se corresponden con la peligrosidad de la actividad y el tipo de actividad (shift/coal-getting). Las variables numéricas se refieren a las mediciones registradas de energía, números de baches sísmicos registrados en rangos de energía, estadísticas de energía promedio y máximos de energía y, finalmente, resultado de peligrosidad / no peligrosidad.
# 
# Observamos también que los valores de los atributos no se reparten de forma homogenea, sino que se concentran en valores determinados, por eso encontramos picos tan altos en los histogramas. 
# 
# En cuanto a la correlación de las variables, encontramos que la variable objetivo "class" no tiene una correlación buena con ninguna de las demás variables. Los valores de correlación más cercanos se encuentran entre el 0.2 y 0.3. Por otra parte, sí que observamos buenas correlaciones entre otras variables, siendo la correlación más fuerte entre maxenergy y energy, con un 1 de correlación; rbumps3 con rbumps, con un valor aproximado entre 0.8 y 0.9; y rbumps2 y rbumps, con un valor aproximado de 0.8.

# ### Resolución mediante regresión/clasificación: Estudio preliminar 
# 
# A continuación, queremos ver si el problema se podría resolver mediante una regresión lineal o una clasificación.
# 
# Para ello, separamos el conjunto de training y de test en proporción 70%/30%, conjuntamente con un preproceso de los atributos categóricos a variables "dummy", que nos ayudarán a facilitar la regresión lineal.

# In[8]:


#X = seismic[['seismic','seismoacoustic','shift','genergy','gpuls','gdenergy','gdpuls','ghazard','nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89','energy','maxenergy']]
X = seismic.drop(columns=['class'])
Y = seismic[['class']]

X = pd.get_dummies(data=X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Instanciamos el modelo LinearRegression 
lr = LinearRegression();

# Ajustamos con los datos de entrenamiento con el método fit
lr.fit(X_train,y_train);

# Predecimos con el método predict 
y_pred = lr.predict(X_train);


# In[9]:


model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()
result.summary()


# ### Conclusiones del estudio preliminar
# 
# Observamos que la predicción con LinearRegression no es buena para este dataset concreto (R² muy bajo) y por ello necesitaremos de otros métodos de predicción.

# ### Preproceso de los datos
# 
# Antes de entrenar los modelos, queremos ver que los datos son completos, su dimensionalidad es adecuada, los datos están bien codificados, etc.
# 
# Empezamos de cero, descargando el dataset de nuevo y separando los datos:

# In[10]:


# Carga de datos
seismic = pd.read_csv("./seismic-bumps.csv", header=0, delimiter = ',')

X = seismic.drop(columns=['class'])
Y = seismic[['class']]


# ##### Detección de valores nulos

# In[11]:


X.isna().sum()


# In[12]:


Y.isna().sum()


# Vemos que, efectivamente, el dataset no contiene valores nulos.

# ##### Valores anómalos (outliers)
# 
# Usamos el histograma de valores para ver si existen valores anómalos.

# In[13]:


X.loc[:,:].hist(figsize=(20,20));


# Observamos en el histograma que los valores anómalos, como en la gráfica de maxenergy o energy, observamos valores que se alejan de la media, pero eso es debido a existen eventos de una carga energética mayor que ocurren con una frecuencia mucho menor que otras actividades de menor carga energética, por lo que eliminar estos valores sería impropio, ya que lo que queremos predecir son los eventos peligrosos, normalmente de mayor carga energética.

# ##### Valores incoherentes/incorrectos
# 
# Usando el histograma anterior, también se pueden ver los valores incoherentes o incorrectos. A primera vista, no parece que existan valores incorrectos.

# ##### Codificación de variables no continuas o no ordenadas

# In[14]:


# Transformamos las variables categóricas en variables dummies usando get_dummies()
X = pd.get_dummies(data=X, drop_first=True)


# ##### Posible eliminación de variables irrelevantes o redundantes
# 
# Vemos que id no proporciona ningún tipo de información relevante, por lo que lo vamos a eliminar.

# In[15]:


# Eliminar variables que no son de utilidad para el problema
seismic = seismic.drop(columns=['id'])


# ##### Creación de nuevas variables que puedan ser útiles
# 
# A priori, no sentimos la necesidad de crear nuevas variables para este problema. Creemos que el dataset ya es lo suficientemente descriptivo para proceder al entrenamiento y las predicciones.

# ##### Normalización de la variables
# 
# Normalizaremos las variables. Para ello, utilizamos estandarización:
# 
# AL FINAL NO LO HACEMOS, ESTO DA PROBLEMAS PARA LA REGRESIÓN LOGÍSTICA

# In[16]:


X_std = X.copy()
Y_std = Y.copy()


# In[17]:


X_std.describe().T


# In[18]:


Y_std.describe().T


# ##### Transformación de las variables
# 
# A priori, no encontramos la necesidad de transformar ninguna variable. Puede ser que durante la experiencia del entrenamiento de modelos, veamos problemas y tengamos que volver al preproceso de datos para eliminar/añadir/transformar variables.

# ### Selección de modelos lineales/cuadráticos y estimación de rendimiento
# 
# Dado que el dataset especifica que el tipo de problema es de clasificación, tomaremos los siguientes 3 métodos y probaremos cuál de los 3 funciona mejor: regresión logística, Naive Bayes y k-vecinos más cercanos.

# ##### Primeramente, separamos los conjuntos de training y de test para el futuro entrenamiento de los modelos:

# In[19]:


# Separación en conjuntos de test y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, test_size=0.3, random_state=0)


# #### Regresión logística

# In[20]:


# Regresión logística
lr = LogisticRegression(max_iter=10000)
print(np.mean(cross_val_score(lr,X_train,y_train,cv=10, error_score='raise')))


# Vemos que los resultados de la cross-validated score son prometedores. Vamos a entrenar un modelo usando este método y veamos su acierto real.

# In[21]:


lr_model = lr.fit(X_train, y_train)
lr_model.score(X_test, y_test)


# In[22]:


print(classification_report(lr_model.predict(X_test), y_test))


# In[23]:


plt.figure(figsize=(8,8));
ConfusionMatrixDisplay.from_estimator(lr_model, X_test, y_test, ax=plt.subplot());


# #### Naive Bayes (Gaussian)

# In[24]:


gnb = GaussianNB()
print(np.mean(cross_val_score(gnb,X_train,y_train,cv=10)))


# Observamos que es ligeramente inferior a regresión logística, pero veamos los resultados tras la predicción de valores.

# In[25]:


gnb_model = gnb.fit(X_train, y_train)
gnb_model.score(X_test, y_test)


# In[26]:


print(classification_report(gnb_model.predict(X_test), y_test))


# In[27]:


plt.figure(figsize=(8,8));
ConfusionMatrixDisplay.from_estimator(gnb_model, X_test, y_test, ax=plt.subplot());


# ##### K-nearest neighbours
# 
# Para usar KNN debemos tener los datos estandarizados, por lo que usamos MinMax scaler.

# In[28]:


# Normalización de los datos
scaler = MinMaxScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# In[29]:


knn =  KNeighborsClassifier()
print(np.mean(cross_val_score(knn,X_train_s,y_train,cv=10)))


# Vemos que la cross-validated score ofrece resultados teóricos prácticamente idénticos a regresión logística. Veamos los resultados prácticos:

# In[30]:


param = {'n_neighbors':[1, 3, 5, 7, 11, 15, 30, 50], 
          'weights':['distance', 'uniform'], 
          'leaf_size':[1, 5, 10, 20, 30],
          'metric': ['l2', 'l1', 'cosine']}

results = (0, None, 0) #Punctuation, Model, cv

for cv in range(2,10):
    knn_gs =  GridSearchCV(knn,param,cv=cv, n_jobs=-1)
    knn_gs.fit(X_train_s, y_train["class"]);
    score = pd.DataFrame(knn_gs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').loc[0, 'mean_test_score']

    if results[0] < score:
        results = (score, knn_gs, cv)
    
knn_gs = results[1]    
print("Para KNN, el mejor valor para cv dentro del experimento es:", results[2], "con puntuación de:", results[0])


# In[31]:


cv=5
niter = 15


# In[32]:


show_html(pd.DataFrame(knn_gs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# In[33]:


cls = [str(v) for v in sorted(Y['class'].unique())]

print(classification_report(knn_gs.predict(X_test_s), y_test,target_names=cls))


# #### ROC curve
# 
# Después de entrenar los tres modelos, podemos observar la curva ROC de los tres modelos en conjunto, y valorar cuál de los tres ofrece mejores resultados:

# In[34]:


RocCurveDisplay.from_estimator(lr_model, X_test,y_test, pos_label=1, ax=plt.subplot());
RocCurveDisplay.from_estimator(gnb_model, X_test,y_test, pos_label=1, ax=plt.subplot());
RocCurveDisplay.from_estimator(knn_gs, X_test_s,y_test, pos_label=1, ax=plt.subplot());
plt.show()


# Tenemos que apreciar que KNN utiliza datos normalizados, de manera que el gráfico no muestra una comparativa muy precisa respecto al resto de gráficas. A partir de esta, obtenemos los siguientes resultados:
# 
# Observamos que KNN ofrece la mayor relación true positive/false positive en la mayor parte del rango comprendido por la gráfica. Si queremos coger un valor que ofrezca una buena relación, podemos coger el mejor modelo para x=0.4, por ejemplo. Observamos que el mejor resultado lo ofrece KNN.
# 
# Regresión linear ofrece ligeramente mejores resultados prácticos que KNN, pero obtiene mayor tasa de falsos positivos en relación a los auténticos positivos, por lo que nos quedaremos con KNN.
# 
# Naive Bayes no lo consideramos porque ofrece menor score que Linear regression y KNN, así como mayor tasa de falsos positivos.

# ### Selección de modelos no lineales y estimación de rendimiento
# 
# En el estudio de los modelos no lineales, hemos seleccionado los siguientes: MLP, Random Forest y Gradient Boosting. Veamos su entrenamiento y selección de los mejores hiperparámetros:

# ##### MLP (Multi-Layer Perceptron)
# 
# Para utilizar MLP debemos normalizar los datos. Como hemos visto en laboratorio, la estandarización proporciona una mayor convergencia, por lo que el proceso es más rápido.

# In[35]:


# Estandarización de los datos de entrenamiento y testing
sdscaler = StandardScaler()

X_train_sd = sdscaler.fit_transform(X_train)
X_test_sd = sdscaler.transform(X_test)


# In[36]:


# Declaración del clasificador MLP con early_stopping para menor sobre ajuste.
mlp = MLPClassifier(max_iter=10000, early_stopping=True, n_iter_no_change=15, random_state=0)
print(np.mean(cross_val_score(mlp,X_train_sd,y_train,cv=10)))


# Vemos que la cross-validation score es similar a KNN. Veamos el acierto práctico cuando se entrena el modelo y se predicen los valores sobre X_test.

# In[37]:


param = {'hidden_layer_sizes':[10, 50, 100, 200, 500, 1000], 
         'activation':['relu', 'logistic', 'identity'], 
         'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1]  }

results = (0,None,0) # Score, Model, cv

for cv in range(2,10):
    mlp =  MLPClassifier(max_iter=10000, early_stopping=True, n_iter_no_change=20,learning_rate='adaptive',random_state=0)
    mlp_gs =  GridSearchCV(mlp,param,cv=cv, n_jobs=-1, refit=True)
    mlp_gs.fit(X_train_sd, y_train["class"]);
    
    score = pd.DataFrame(mlp_gs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='mean_test_score').loc[0,"mean_test_score"]
    if results[0] < score:
        results = (score,mlp_gs,cv)
        
mlp_gs = results[1]
print("Para MLP, el mejor valor para cv dentro del experimento es:", results[2], "con puntuación de:", results[0])


# In[38]:


show_html(pd.DataFrame(mlp_gs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# Como en los modelos previos, el resultado práctico es peor que el teórico. Observamos que los mejores hiperparámetros para este entrenamiento son: Activación ReLU, 100 de tamaño de capas ocultas y tasa inicial de aprendizaje a 0.001. 

# In[39]:


param = {'hidden_layer_sizes':[10, 50, 100, 200, 300], 
'activation':['relu', 'identity', 'logistic'], 
'alpha':[0.0001, 0.001, 0.01],
'momentum': [0.95, 0.90, 0.85, 0.8], 
'learning_rate_init': [0.001, 0.01, 0.1],
'n_iter_no_change':[10, 20, 40, 50], 
'learning_rate': ['constant', 'invscaling', 'adaptive']}

mlp =  MLPClassifier(max_iter=10000,early_stopping=True,random_state=0)
mlp_bs =  BayesSearchCV(mlp,param,
                        n_iter=niter, 
                        cv=cv, n_jobs=-1, 
                        refit=True,random_state=0)
mlp_bs.fit(X_train_sd, y_train["class"]);


# In[40]:


show_html(pd.DataFrame(mlp_bs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# Probamos con Bayes Search pero no obtenemos resultados mejores. Observamos que los mejores hiperparámetros nos proprocionan una puntuación con una cota superior de 0.93, que en ningún caso se supera, ya sea usando Bayes Search o no. Dado que no podemos mejorarlo, nos quedaremos con el resultado del modelo más simple (el que no utiliza Bayes) para compararlo con el resto de modelos obtenidos.
# 
# Al final del entrenamiento de los 3 modelos elegidos en la sección de modelos no lineales, veremos la curva ROC y compararemos su comportamiento.

# In[41]:


print(classification_report(mlp_bs.predict(X_test), y_test,target_names=cls))


# In[42]:


plt.figure(figsize=(8,8));
ConfusionMatrixDisplay.from_estimator(mlp_bs, X_test,y_test, ax=plt.subplot())


# ##### Random Forest

# In[43]:


rf =  RandomForestClassifier(random_state=0)
print(np.mean(cross_val_score(rf,X_train_sd,y_train,cv=10)))


# In[44]:


iter=40
param = {'n_estimators': [5,10,25,40, 50, 75,100, 200], 
         'criterion':['gini', 'entropy'], 
         'max_depth':[None, 1, 2, 3,  5,  8, 9,10,15],
         'min_samples_leaf':[1,2,3,5,10]}

rf_bs =  BayesSearchCV(rf,param,n_iter=iter, cv=cv, n_jobs=-1, refit=True, random_state=0)
rf_bs.fit(X_train, y_train["class"]);


# In[45]:


show_html(pd.DataFrame(rf_bs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# In[46]:


print(classification_report(rf_bs.predict(X_test), y_test,target_names=cls))


# In[47]:


plt.figure(figsize=(8,8));
ConfusionMatrixDisplay.from_estimator(rf_bs, X_test,y_test, ax=plt.subplot())


# ##### Gradient Boosting

# In[48]:


param = {'n_estimators': [5,10,25,40, 50, 75,100, 200], 
         'loss':['log_loss', 'exponential'], 
         'criterion':['friedman_mse', 'squared_error'], 
         'max_depth':[None, 1, 2, 3,  5,  8, 9,10,15],
         'min_samples_leaf':[1,2,3,5,10], 
         'learning_rate':[0.1,0.5, 1,3, 5, 10, 15]}

gb =  GradientBoostingClassifier(random_state=0,n_iter_no_change=5)
gb_bs =  BayesSearchCV(gb,param,n_iter=iter, cv=cv, n_jobs=-1, refit=True, random_state=0)
gb_bs.fit(X_train, y_train["class"]);


# In[49]:


show_html(pd.DataFrame(gb_bs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# In[50]:


print(classification_report(gb_bs.predict(X_test), y_test,target_names=cls))


# In[51]:


plt.figure(figsize=(8,8));
ConfusionMatrixDisplay.from_estimator(gb_bs, X_test,y_test, ax=plt.subplot())


# In[52]:


plt.figure(figsize=(8,8));
RocCurveDisplay.from_estimator(gb_bs, X_test,y_test, pos_label=1, ax=plt.subplot(), name='Gradient Boosting');
RocCurveDisplay.from_estimator(rf_bs, X_test,y_test, pos_label=1, ax=plt.subplot(), name='Random Forest');
RocCurveDisplay.from_estimator(mlp_gs, X_test,y_test, pos_label=1, ax=plt.subplot(), name="Multi-Layer Perceptron");


# Como MLP necesita que la entrada esté normalizada (estandarización) la curva ROC no aparece de manera adecuada en comparación al resto de valores, para Gradient Boosting y Random Forest. 
# 
# Si analizamos las gráficas comparables, vemos que la línea de Gradient Boosting casi siempre toma valores más altos en el eje vertical (True positives) que la gráfica de Random Forest, por lo que podemos pensar que Gradient Boosting ofrece mejores resultados (mayor tasa de auténticos positivos en relación a los falsos positivos).
# 
# Como los resultados de MLP en la práctica son prácticamente iguales a GB y RF, cogeremos como mejor modelo Gradient Boosting, en el apartado de modelos no lineales.

# ### Elección final del modelo, justificación y características
# 
# Del apartado de modelos lineales hemos escogido K Nearest Neighbours, y de los modelos no lineales hemos escogido Gradient Boosting. Veamos las curvas ROC y los datos comparativos para decidirnos sobre el modelo final:

# In[53]:


plt.figure(figsize=(8,8));
RocCurveDisplay.from_estimator(gb_bs, X_test,y_test, pos_label=1, ax=plt.subplot(), name='Gradient Boosting');
RocCurveDisplay.from_estimator(knn_gs, X_test_s,y_test, pos_label=1, ax=plt.subplot());


# In[54]:


show_html(pd.DataFrame(knn_gs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# In[55]:


show_html(pd.DataFrame(gb_bs.cv_results_).loc[:,['params', 'mean_test_score','rank_test_score']].sort_values(by='rank_test_score').head().to_html())


# In[56]:


print(classification_report(knn_gs.predict(X_test_s), y_test,target_names=cls))


# In[57]:


print(classification_report(gb_bs.predict(X_test), y_test,target_names=cls))


# A través de la curva ROC podemos observar que Gradient Boosting ofrece mejores resultados en todos los valores del eje horizontal, por lo que de momento consideramos Gradient Boosting como la mejor opción.
# 
# Si analizamos los valores de mean test score para ambos modelos, vemos que los valores son muy similares, superando KNN a Gradient Boosting por 0.003. La diferencia es mínima, por lo que realmente no podemos añadir diferencia a través de estos datos.
# 
# Finalmente, observamos que las puntuaciones tras las predicciones sobre el conjunto de test son muy similares también. Para KNN obtenemos unos valores de (0.98, 0.94, 0.96), mientras que para GB tenemos (1.00, 0.95, 0.97). Los valores para GB son ligeramente superiores a KNN, por lo que tomaremos como mejor resultado en este aspecto el modelo de GB.
# 
# Tras todos los análisis, vemos que GB es mejor que KNN para este dataset en concreto, con una mayor puntuación y menor tasa de falsos positivos por número de auténticos positivos.

# ### Referencias
# 
# Utilizamos citación en formato APA para esta práctica:
# 
# - UCI Machine Learning Repository: seismic-bumps Data Set. (s. f.). Recuperado 24 de octubre de 2022, de http://archive.ics.uci.edu/ml/datasets/seismic-bumps
# - haloboy777. (n.d.). Haloboy777/arfftocsv: Arff to CSV converter (python). GitHub. Recuperado 27 de diciembre de 2022, de https://github.com/haloboy777/arfftocsv 

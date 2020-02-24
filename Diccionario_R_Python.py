# source("libreria.R")
dir()
#library(libreria)
import os
import csv
import pandas as pd
import numpy as np

# setwd("C:/Users/laguila/Desktop")
os.chdir("C:/Users/ldelaguila/Google Drive/ENRI")
# getwd()
print(os.getcwd())
#%% LECTURA Y ESCRITURA
# read.csv(file = "archivo.csv", header = F, sep = ",", row.names = T, col.names = T, skip = 10, nrows = 1000)
# data.table::fread("archivo.csv", stringsAsFactors = F)
np.loadtxt('digits.csv', delimiter="," , skiprows=1, usecols=[0,2], dtype=str)
recfromcsv(file, delimiter = ",", names = True, dtype = None)
genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
pd.read_csv('titanic.csv', header = None, nrows = 5, comment='#', na_values=['Nothing'], delimiter=" ")
xl = pd.ExcelFile('battledeath.xlsx')
df1 = xl.parse("2004") #nombre del sheet
df2 = xl.parse(0, skiprows=[0], names=['Country', 'AAM due to War (2002)'], parse_cols=[0])#primera hoja, renombrando columnas

#Leer a trozos
df_chunk = pd.read_csv(r'../input/data.csv', chunksize=1000)
chunk_list = []  # append each chunk df here 
for chunk in df_chunk:  #esto ya va leyendo trozo a trozo
    chunk_filter = chunk_preprocessing(chunk)
    chunk_list.append(chunk_filter)
df_concat = pd.concat(chunk_list)


# write.csv(dataframe, "archivo.csv")
fichero.to_csv('nombre.csv')

#%% VECTORES Y MATRICES
# a=c(1, 2, 3)
a=np.array([1,2,3])
# a[1]
a[1]
# a[1:2]
a[1:3]
# runif(n = 4, min = 0, max = 10)
import random
random.sample(range(10), 4)
# sample(10)
from random import sample 
sample(population=range(10), k=10)
#seq(0,10,length.out = 4)
np.linspace(0,10,4)
# a[1]=abs(-5)
a[0]=abs(-5)

# matriz=matrix(data = 1:100, nrow = 10, byrow=T)
matriz = np.arange(100).reshape(10,10,)
# matriz[1:5,1]=4
matriz[0:4,0]=4

# #Trasponer matrix
# t(matrix)
np.transpose(matriz)
matriz.T
#matriz*matriz
matriz*matriz
#matrix %*% matriz
matriz @ matriz
#%% LISTAS
# lista=list(a, 2*a, 4)
lista=[a, 2*a, 4]
# lista[[1]]
lista[1]

#%% DATAFRAMES

# aux=data.frame(unos=rep(1,10), otros=1:10)
df = pd.DataFrame({"unos": np.repeat(1,10), 'col2': range(1,11,1)})
# aux$unos
df["unos"]
df.unos

# data(iris)
from sklearn.datasets import load_iris
iris = load_iris()
import sklearn
iris=sklearn.datasets.load_iris()
datos=pd.DataFrame(iris.data)
datos['species']=iris.target
datos.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
datos.dropna(how="all", inplace=True) # remove any empty lines
datos["Species"]=datos["Species"].replace(0, iris.target_names[0])
datos["Species"]=datos["Species"].replace(1, iris.target_names[1])
datos["Species"]=datos["Species"].replace(2, iris.target_names[2])


# aux=data.frame(iris)
iris=datos.copy()
# str(iris)

# table(iris$Sepal.Length, iris$Sepal.Width)
from collections import Counter
Counter(datos.Species)
# head(iris)
iris.head()
# tail(iris)
iris.tail()
# summary(iris)
iris.describe()
# dim(iris)
iris.shape()
# sum((iris$Sepal.Length))
iris["SepalLength"].sum()
# min(iris$Sepal.Length)
iris["SepalLength"].min()
# max(iris$Sepal.Length)
iris["SepalLength"].max()
# mean(iris$Sepal.Length)
iris["SepalLength"].mean()
# median(iris$Sepal.Length)
iris["SepalLength"].median()
# sd(iris$Sepal.Length)
iris["SepalLength"].std()
# quantile(x = iris$Sepal.Length, 0.35)
np.quantile(iris["SepalLength"],0.35)
# cor(iris[,1:4])
iris[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].corr(method="pearson")


# which(data$Sepal.Length<6)
np.where(iris["SepalLength"]<6)
# data[1,4]=0.3
iris.iloc[0, 3]=0.3
iris.iloc[:, :]
iris.iloc[:,list(range(0,3))]
iris.loc[0, ['PetalWidth']]=0.3
# data[which(data$Sepal.Length<5.2),1]=NA
iris.loc[iris['SepalLength']<5.2,['SepalLength']]=np.NaN
# data[is.na(data)]=5.2
iris=iris.fillna(5.2)
# data=na.omit(data)
iris=iris.dropna(how='any').reset_index()
iris.dropna(how='all')
# data$Sepal.Length[str_detect(data$Species, pattern = "set")]="esta"
iris.loc[iris['Species'].str.contains('set'), 'Species']='esta'
# str_replace(data$Species, "set", "sat")
iris["Species"].str.replace("set", "sat")

# variable="Sepal.Length"
# match(variable, colnames(iris))
# colnames(iris)[colnames(iris) %in% variable]
variable='SepalLength'
variable in iris.columns #True
iris.iloc[:,iris.columns==variable]

# library(dplyr)
import dplython as dp
iris = dp.DplyFrame(iris)
from dplython import (DplyFrame, X, diamonds, select, sift,
  sample_n, sample_frac, head, arrange, mutate, group_by,
  summarize, DelayFunction)
# data(iris)
# data=iris %>% 
# select(Petal.Length, Petal.Width, Sepal.Length, Sepal.Width, Species)
iris >> dp.select(X.Species) >> dp.head()

iris[['Species', 'PetalLength']]
iris.drop('SepalLength', axis=1) #quitar esa columna
iris.drop(5, axis=0) #quitar la sexta fila
# data=iris %>% 
# filter(Petal.Length>1 & Petal.Length<100)
iris >> dp.sift(X.PetalLength>5)

iris[(iris['PetalLength']>5) & (iris['PetalLength']<6)]
# data=iris %>% 
# dplyr::group_by(Species) %>%
# summarise(media=mean(Petal.Length)) 
iris >> dp.group_by(X.Species) >> dp.summarize(media=X.PetalLength.mean())

iris.groupby(['Species'])['PetalLength'].agg(['mean', 'sum', 'count'])
iris.groupby(['Species'])['PetalLength'].agg({'var1':'mean', 'var2':'sum', 'var3':'count'})
iris.groupby(['Species'])['PetalLength'].agg({'var1':['mean', 'sum']})
aggregations = {
    'dsuma':'sum',
    
}
import math
iris.groupby(['Species'])['PetalLength'].agg({'dsuma':'sum', 'otro': lambda x: math.sqrt(x.mean()) - 1})
# data=iris %>% 
# mutate(total=Sepal.Length+Petal.Length, otro=ifelse(Petal.Length>2, "grande", "pequeÃ±o"))
iris >> dp.mutate(redondeado=X.PetalLength.round(), redondeado2=X.SepalLength.round())

iris.assign(redondeado = lambda x: x.PetalLength.round(), redondeado2 = lambda x: x.SepalLength.round())

#ifelse(y==0, 0, 1)
np.where((y == 0), 0, 1)
# data=iris %>% 
# distinct(Species, Sepal.Length, .keep_all = T)
iris >> dp.distinct(X.SepalLength)

iris.drop_duplicates()
iris.drop_duplicates(subset='PetalLength')
# #ordenando
# data=iris %>% 
# arrange(Sepal.Length, Sepal.Width)
iris >> dp.arrange(X.PetalLength)

iris.sort_values("PetalLength", ascending=False)
# data$ceros=0
iris['ceros']=0
# data$ceros=NULL
del(iris['ceros'])

# data=data.frame(x1=rep(1,10), x2=rep(2, 10))
data = pd.DataFrame({'x1':np.ones(10), 'x2':np.repeat(2, 10)})
# data2=data.frame(x3=rep(3,10), x4=rep(4, 10))
data2 = pd.DataFrame({'x3':np.repeat(3, 10), 'x4':np.repeat(4, 10)})
# data3=data.frame(x1=rep(1,10), x5=rep(5, 10))
data3 = pd.DataFrame({'x1':np.ones(10), 'x5':np.repeat(5, 10)})
# total=left_join(data, data3, by="x1")
# total2=merge(data, data2, by="x1")
data.merge(data3, on='x1', how='left')
pd.merge(data, data3, on='x1', how='left')
pd.concat([data, data3], keys=['x', 'y']) #el x e y son para diferenciarlos
pd.concat({'x':data, 'y':data3})
# total3=cbind(data, data2)
total3=pd.concat([data, data3], axis=1)
# total4=rbind(data, data2)
total4=pd.concat([data, data3], axis=0)
total4=data.append(data3)
# library(tidyverse)
# data(iris)
# data=iris %>% distinct(Sepal.Length, Species, .keep_all = T)
iris.duplicated()
iris.drop_duplicates()
iris = iris.drop_duplicates(iris.columns[~iris.columns.isin(['SepalLength', 'Species'])], keep='first')

# data=spread(data, key=Species, value=Sepal.Length)
spread=pd.pivot_table(iris, values='SepalLength', index=['SepalWidth', 'PetalLength', 'PetalWidth'], columns='Species').reset_index()
# data=gather(data, key=Species, value=Sepal.Length, 4:6)
iris2=spread.melt(id_vars=['SepalWidth', 'PetalLength', 'PetalWidth'], var_name='Species', value_name='SepalLength').dropna(how='any').reset_index()


# #renombrar filas y columnas
# colnames(iris)=c("n1", "n2", "n3", "n4")
# rownames(iris)=1:nrow(iris)
iris.columns=['SapalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
iris.rename(columns={'Species': 'Especies'},
            index={0:'cero',1:'uno'})


# #Formatear columnas
# iris$Sepal.Length=as.integer(iris$Sepal.Length)
iris.SepalLength.map(lambda x: int(x))
# iris$Sepal.Length=as.numeric(iris$Sepal.Length)
iris.SepalLength.map(lambda x: float(x))
# iris$Species=as.character(iris$Species)
iris.Species.map(lambda x: str(x))
# iris$Species=as.factor(iris$Species)

import math
# iris$Petal.Length=round(iris$Petal.Length,2)
round(iris['PetalLength'], 1)
iris['PetalLength'].round(1)
# iris$Petal.Length=ceiling(iris$Petal.Length,2)
iris['PetalLength'].map(lambda x: math.ceil(x))
# iris$Petal.Length=floor(iris$Petal.Length,2)
iris['PetalLength'].map(lambda x: math.floor(x))

# rm(data)
del(data)

# str_detect(as.character(iris$Species), string = "setosa")
mask = iris['Species'].map(lambda x: bool(re.match('set', x)))
iris['Species'][mask] #columa Spcies
# substr(as.character(iris$Species), 1, 4)
iris['Species'].map(lambda x: x[0:4])
# str_replace(as.character(iris$Species), pattern = "a", replacement = "e")
iris['Species'].map(lambda x: x.replace('set', 'sat'))
# str_remove(as.character(iris$Species), pattern = "a")
iris['Species'].map(lambda x: x.replace('set', ''))
# colSums(iris[,1:4])
iris.iloc[:,0:4].sum(axis=1)
# rowSums(iris[,1:4])
iris.iloc[:,0:4].sum(axis=1)


# data(iris)
# datos=iris
# datos$Sepal.Length[datos$Sepal.Length<5]=NA
# fill(datos, "Sepal.Length", .direction = "down")
iris=iris.fillna(5.2)

# #Crear directorios
# dir.create("carpeta_guardado")
os.mkdir('dir1')
os.makedirs('dir1/dir2/dir3', exist_ok=True) #hacer varios a la vez, y no de error en caso de que exista
# dir.exists("carpeta_guardado")
os.path.isdir('Favorites')



Cambio 1
Cambio 2









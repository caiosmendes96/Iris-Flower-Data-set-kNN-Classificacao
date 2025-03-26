# Iris-Flower-Data-set-kNN-Classificacao
Utilização do banco de dados “Iris Flower Data Set” para abordar o problema de classificação utilizando o algoritmo k-NN em Python.


Importação e transformação do banco de dados Íris para a classificação utilizando o algoritmo KNN euclidiano.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from statistics import mean
from sklearn.metrics import accuracy_score 
from sklearn.metrics import explained_variance_score as evs

dados = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

atributos = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Variety']

iris =  pd.read_csv(dados, names = atributos)

transformacao = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


iris['Variety_num'] = iris['Variety'].map(transformacao)


iris = iris.drop(columns = 'Variety')

x = iris.drop(columns = 'Variety_num')
y = iris['Variety_num']

iris.sample(5)

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal_Length</th>
      <th>Sepal_Width</th>
      <th>Petal_Length</th>
      <th>Petal_Width</th>
      <th>Variety_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>6.3</td>
      <td>3.3</td>
      <td>4.7</td>
      <td>1.6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>135</th>
      <td>7.7</td>
      <td>3.0</td>
      <td>6.1</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.6</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.5</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>6.1</td>
      <td>2.9</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<b>Atributo alvo: Variety </b>

Para o treinamento do KNN, utilizamos 70% objetos para o treinamento e 30% para teste. </br></br>
O parâmetro "random_state" em train_test_split  é responsável por randomizar a seleção dos objetos escolhidos para o treinamento e teste. </br></br>
Ao alterar o valor do "random_state", os objetos de treinamento e teste serão diferentes. Com isso, os valores de acurácia e o melhor K poderão mudar. </br></br>
Para esse experimento, utilizamos o "random_state" igual a 11. </br>

Vamos procurar o melhor valor de K (hiperparâmetro) com os objetos de treinamento ao encontrar a maior acurácia variando o valor de vizinhos entre 1 e 20.  </br>


```python
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size = 0.30, random_state = 11)

print("Quantidade de objetos para o treino: ", len(train_X))
print("Quantidade de objetos para o teste: ", len(test_X))
print("-------------------------------------")

listaAcuracia = []
for n in range(1,20):
    
    knn = neighbors.KNeighborsClassifier(n_neighbors = n)
    
    knn.fit(train_X, train_y)
    
    predicaoTeste = knn.predict(test_X)
    listaAcuracia.append(accuracy_score(test_y, predicaoTeste))
    
    

melhorK = listaAcuracia.index(max(listaAcuracia)) + 1

print("Maior Acurácia: ", max(listaAcuracia)*100)
print("Valor de k:", melhorK)


```

    Quantidade de objetos para o treino:  105
    Quantidade de objetos para o teste:  45
    -------------------------------------
    Maior Acurácia:  97.77777777777777
    Valor de k: 5
    

<b>Apresentação de um exemplo da classificação: </b>


```python
melhorKnn = neighbors.KNeighborsClassifier(n_neighbors = melhorK)
melhorKnn.fit(train_X, train_y)


#Variáveis para uma predição utilizando regressão

#Dados do objeto nº 98
sepalLength = 5.1
sepalWidth = 2.5
petalLength = 3.0
petalWidth = 1.1

predicao = melhorKnn.predict([[sepalLength,sepalWidth,petalLength,petalWidth],])

variety_names = {
    0:'Iris-setosa',
    1:'Iris-versicolor', 
    2:'Iris-virginica'
}
print("Classificação com o melhor valor de K: ", melhorK)
print("Classificação (tabela): ", variety_names[y[98]] )
print("Classificação (resultado KNN): ", variety_names[predicao[0]] )

```

    Classificação com o melhor valor de K:  5
    Classificação (tabela):  Iris-versicolor
    Classificação (resultado KNN):  Iris-versicolor
    

    C:\Users\caios\anaconda3\Lib\site-packages\sklearn\base.py:464: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
      warnings.warn(
    


```python

```




```python

```

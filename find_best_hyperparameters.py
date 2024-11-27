from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import pickle


from sklearn.feature_extraction.text import CountVectorizer


import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

# Abrindo as duas bases e separando em baseA e baseB
baseA = pd.read_csv("webkb-parsed.csv")
baseB = pd.read_csv("SyskillWebert.csv")
print("dados carregados com sucesso")

# divisão dos dados de treino e testre
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(baseA['text'], baseA['class'], test_size=0.20, random_state=42)
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(baseB['text'], baseB['class'], test_size=0.20, random_state=42)
print("dados de treino e teste separados com sucesso")

with open('X_train_A.pkl', 'wb') as f:
    pickle.dump(X_train_A, f)
with open('X_train_B.pkl', 'wb') as f:
    pickle.dump(X_train_B, f)
    
with open('X_test_A.pkl', 'wb') as f:
    pickle.dump(X_test_A, f)
with open('X_test_B.pkl', 'wb') as f:
    pickle.dump(X_test_B, f)

with open('y_train_A.pkl', 'wb') as f:
    pickle.dump(y_train_A, f)
with open('y_train_B.pkl', 'wb') as f:
    pickle.dump(y_train_B, f)
    
with open('y_test_A.pkl', 'wb') as f:
    pickle.dump(y_test_A, f)
with open('y_test_B.pkl', 'wb') as f:
    pickle.dump(y_test_B, f)
    
print("dados de treino e teste salvos em disco com sucesso")

# Vetorizando bases de treino
vectorizer = CountVectorizer()
XA = vectorizer.fit_transform(X_train_A)
YA = y_train_A
XB = vectorizer.fit_transform(X_train_B)
YB = y_train_B


def calculateHyperParams(estimator, param_grid, X, Y, fileName):
    # cv 4 para dividir em 4 grupos para o cross validation
    # n_jobs -1 para usar o máximo possível de jobs em paralelo
    gridSearch = GridSearchCV(estimator=estimator(), param_grid=param_grid, cv=3, n_jobs=-1)

    # aplicando o gridSearch aos dados de treino
    gridSearch.fit(X, Y)

    print("melhores parâmetros")
    print(gridSearch.best_params_)

    # exportando resultados para dataframe e depois csv
    df = pd.DataFrame(gridSearch.cv_results_)
    df.to_csv(fileName)


    
param_grid_naive = {
        'alpha': [x*0.1 for x in range(1, 20)],
        'fit_prior': [True, False],
        'force_alpha': [True, False]
    }

print("naive bayes baseA")
calculateHyperParams(MultinomialNB, param_grid_naive, XA, YA, "result_naive_A.csv")
print("naive bayes baseB")
calculateHyperParams(MultinomialNB, param_grid_naive, XB, YB, "result_naive_B.csv")

param_grid_KNN = {
    'n_neighbors': range(1,20)
}
print("KNN baseA")
calculateHyperParams(KNeighborsClassifier, param_grid_KNN, XA, YA, "result_KNN_A.csv")
print("KNN baseB")
calculateHyperParams(KNeighborsClassifier, param_grid_KNN, XB, YB, "result_KNN_B.csv")


param_grid_logistic_regression = [
    {'penalty':['l1','l2'],
    'C' : np.logspace(-4,4,5),
    'solver': ['lbfgs','newton-cg'],
    'max_iter': [200]
    }
]

print("logistic regression baseA")
calculateHyperParams(LogisticRegression, param_grid_logistic_regression, XA, YA, "result_Logistic_A.csv")
print("logistic regression baseB")
calculateHyperParams(LogisticRegression, param_grid_logistic_regression, XB, YB, "result_Logistic_B.csv")


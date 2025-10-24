import pandas as pd
import numpy as np 

from sklearn.datasets import load_iris 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("student-scores.csv")

iris = load_iris()

X = pd.DataFrame(iris.data,columns = iris.feature_names)
y = pd.Series(iris.target, name='species')

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=200)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Kovalski relátorio:")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

sns.heatmap(confusion_matrix(y_test,y_pred),annot = True, cmap = 'Blues', xticklabels =iris.target_names, yticklabels= iris.target_names)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão que me confunde- regressão logística')
plt.show()

coef_df = pd.DataFrame(
    model.coef_,
    columns = iris.feature_names,
    index = iris.target_names
)

#print("Coeficientes de regressão logística (por classe):")
#display(coef_df)

print("Interceptos do modelo:")
print(model.intercept_)

amostra = X_test.iloc[0]
amostra_esc = scaler.transform([amostra])

probas = model.predict_proba(amostra_esc)[0]

print("Amostra:")
print(amostra)

print("\nProbabilidade prevista para cada classe:")
for classe, prob in zip (iris.target_names,probas):
    print(prob)

print("\nClasse prevista:", iris.target_names[np.argmax(probas)])
print("Classe real:", iris.target_names[y_test.iloc[0]])

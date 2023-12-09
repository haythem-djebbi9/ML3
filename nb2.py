from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Charger le jeu de données Iris
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Diviser le dataset en données d’apprentissage et données de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Créer le modèle Naive Bayes Gaussian et entraîner les données
model = GaussianNB()
model.fit(X_train, Y_train)

# Prédire les étiquettes sur les données de test
Y_pred = model.predict(X_test)

# Calculer les métriques de performance
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Afficher les résultats
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

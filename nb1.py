import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 1. Charger et Explorer les données
loan_data = pd.read_csv("loan.csv")

# Utiliser seaborn countplot pour explorer la colonne 'purpose'
sns.countplot(x='purpose', data=loan_data)

# 2. Convertir la colonne 'purpose' en variables indicatrices
loan_data = pd.get_dummies(loan_data, columns=['purpose'])

# 3. Définir les variables caractéristiques (feature) X et la variable cible (target) Y
X = loan_data.drop('target_column_name', axis=1)  # Remplacer 'target_column_name' par le nom réel de la colonne cible
Y = loan_data['target_column_name']

# 4. Diviser le dataset en données d’apprentissage et données de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 5. Créer le modèle Naive Bayes Gaussian et entraîner les données
model = GaussianNB()
model.fit(X_train, Y_train)

# 6. Évaluer le modèle
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

# Interpréter les résultats
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

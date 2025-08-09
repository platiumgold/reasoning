import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('winequality-red.csv', sep=';')
df = df.drop(['pH'], axis=1)

X = df.drop('quality', axis=1)
y = df['quality']

separator = int(len(X)*0.3)
X_train, X_test = X[separator:], X[:separator]
y_train, y_test = y[separator:], y[:separator]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, verbose=0, cv=5, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred))

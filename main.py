import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Завантаження даних
penguins = sns.load_dataset('penguins')

# Видаляємо пропущені значення
penguins_clean = penguins.dropna()

# One-Hot Encoding для 'island' та 'sex'
penguins_encoded = pd.get_dummies(penguins_clean, columns=['island', 'sex'], drop_first=True)

# Ознаки (X) та цільова змінна (y)
X = penguins_encoded.drop('species', axis=1)
y = penguins_encoded['species']

# Розподіл: 80% тренування, 20% тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Оцінка точності за допомогою крос-валідації
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean CV accuracy: {cv_scores.mean():.4f}')

# Навчання
pipeline.fit(X_train, y_train)

# Оцінка на тестових даних
test_accuracy = pipeline.score(X_test, y_test)
print(f'Test set accuracy: {test_accuracy:.4f}')

# Параметри для RandomForest
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Підбір параметрів
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation accuracy: {grid_search.best_score_:.4f}')

# Прогнозування на тестовому наборі
y_pred = grid_search.predict(X_test)

# Звіт класифікації
print(classification_report(y_test, y_pred))

# Матриця помилок
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.best_estimator_.named_steps['classifier'].classes_)
disp.plot()
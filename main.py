import kagglehub
from sklearn.impute import SimpleImputer
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Download latest version
path = kagglehub.dataset_download("saddasdaasda/httpswww-kaggle-comctitanicdata")

# Ensure path points to a specific file
if os.path.isdir(path):
    # Assuming the dataset contains a single CSV file
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if files:
        path = os.path.join(path, files[0])
    else:
        raise FileNotFoundError("No CSV file found in the dataset directory.")

# load dataset into pandas dataframe
df = pd.read_csv(path)

imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# remove Columns [PassengerId, Name , Ticket]
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# make label Encoding for Embarked Column
df['Embarked'] = df['Embarked'].astype('category').cat.codes

# make label Encoding for Sex Column
df['Sex'] = df['Sex'].astype('category').cat.codes

# Handle Cabin column: fill missing values and encode as category
df['Cabin'] = df['Cabin'].fillna('Unknown').astype('category').cat.codes

# split data into train and test
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to compare
models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

# Define hyperparameter grids
param_grids = {
    'RandomForest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    },
    'LogisticRegression': {
        'model__C': [0.1, 1, 10],
        'model__solver': ['lbfgs', 'liblinear']
    }
}

# Perform hyperparameter tuning and compare models
for model_name, model in models.items():
    print(f"Tuning hyperparameters for {model_name}...")
    pipeline = Pipeline([
        ('model', model)
    ])
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} Accuracy after tuning: {accuracy}")

    # Test with sample input
    sample_input = X_train.sample(1)
    print(f"Sample Input: {sample_input}")
    print(f"{model_name} Prediction: {best_model.predict(sample_input)}")

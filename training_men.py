from pathlib import Path
import numpy as np
import optuna.distributions
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from optuna.integration import OptunaSearchCV

dataset_path = Path(".", "training_data", "men", "datasets.npz")
data = np.load(dataset_path)
X = data["X"]
y = data["y"]

rf_search_space = {
        'n_estimators': optuna.distributions.IntDistribution( 50, 300),
        'max_depth': optuna.distributions.IntDistribution( 2, 20),
        'min_samples_split': optuna.distributions.IntDistribution( 2, 10),
        'min_samples_leaf': optuna.distributions.IntDistribution( 1, 10),
        'max_features': optuna.distributions.CategoricalDistribution(['auto', 'sqrt', 'log2']),
        'bootstrap': optuna.distributions.CategoricalDistribution([True, False])
    }
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)
clf = RandomForestClassifier(random_state=42)

optuna_search = OptunaSearchCV(
    clf,
    rf_search_space,
    cv=cv,
    n_trials=100,
    scoring='matthews_corrcoef',
    random_state=42,
    n_jobs=-1,
    verbose=100
)
# Run search
optuna_search.fit(X, y)

# Best params and score
print("Best hyperparameters:", optuna_search.best_params_)
print("Best cross-val accuracy:", optuna_search.best_score_)
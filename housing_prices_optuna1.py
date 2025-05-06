"""LightGBM model for housing prices prediction with Optuna hyperparameter optimization."""
import warnings
from pathlib import Path
import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna

from housing_prices_features import feature_engineering
from util import name_feature_importances

warnings.filterwarnings('ignore')

# Load the data
subdirectory = Path('housing-prices')
df_train = pd.read_csv(subdirectory / 'train.csv')
df_submit = pd.read_csv(subdirectory / 'test.csv')

# Outlier removal
df_train = df_train[df_train['GrLivArea'] < 4500]

# Feature engineering
df_train_fe = feature_engineering(df_train)
df_submit_fe = feature_engineering(df_submit)

# Target transformation
y = np.log1p(df_train_fe['SalePrice'])
X = df_train_fe.drop(['SalePrice', 'Id'], axis=1)
X_submit = df_submit_fe.drop(['Id'], axis=1)
ids_submit = df_submit_fe['Id']

# Categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Convert to categorical dtype for LightGBM
for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_submit[col] = X_submit[col].astype('category')

# Stratified K-Fold based on binned target
y_bins = pd.qcut(y, q=5, duplicates='drop').astype(str)

# Optuna objective
def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0, log=True),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'n_estimators': 1000,
    }

    dtrain = lgb.Dataset(X, label=y, categorical_feature=categorical_cols, free_raw_data=False)

    cv_result = lgb.cv(
        params,
        dtrain,
        folds=StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y_bins),
        num_boost_round=1000,
        seed=42,
        callbacks=[lgb.early_stopping(20)],
    )

    first_key = list(cv_result.keys())[0]
    return min(cv_result[first_key])

# Optimize with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest parameters:")
pprint.pprint(study.best_params)

# Final training
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_estimators': 1000,
})

model = lgb.LGBMRegressor(**best_params)
model.fit(X, y, categorical_feature=categorical_cols)

# Feature importances
feature_importances = name_feature_importances(model.feature_importances_, X.columns)
print("\nTop feature importances:")
pprint.pprint(feature_importances[:7])

# Train-test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert split parts again to 'category'
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

model.fit(X_train, y_train, categorical_feature=categorical_cols)

# Predictions
y_pred = np.expm1(model.predict(X_test))
y_submit = np.expm1(model.predict(X_submit))

# Scores
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
mse = mean_squared_error(np.expm1(y_test), y_pred)
r2 = r2_score(np.expm1(y_test), y_pred)

print(f"\nRMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save submission
submission = pd.DataFrame({'Id': ids_submit, 'SalePrice': y_submit})
submission.to_csv(subdirectory / 'housing-submission.csv', index=False)

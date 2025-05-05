"""
Kaggle Housing Prices with Native LightGBM CV + Optuna
"""
import warnings
from pathlib import Path
import pprint
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna

from housing_prices_features import feature_engineering
from util import name_feature_importances

# pylint: disable=line-too-long

warnings.filterwarnings('ignore')

# Load the data
subdirectory = Path('housing-prices')
df_train = pd.read_csv(subdirectory / 'train.csv')
df_submit = pd.read_csv(subdirectory / 'test.csv')

# Feature engineering
df_train_fe = feature_engineering(df_train)
df_submit_fe = feature_engineering(df_submit)

# Target and features
X = df_train_fe.drop(['SalePrice', 'Id'], axis=1)
y = df_train_fe['SalePrice']
X_submit = df_submit_fe.drop(['Id'], axis=1)
ids_submit = df_submit_fe['Id']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns)
])
preprocessor.fit(X)
X_transformed = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())
X_test_transformed = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())
X_submit_transformed = pd.DataFrame(preprocessor.transform(X_submit), columns=preprocessor.get_feature_names_out())

# Optuna + LightGBM native CV
def objective(trial):
    """Objective function for Optuna optimization."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'n_estimators': 1000,
        'num_leaves': trial.suggest_int('num_leaves', 30, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.4),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.008, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.8, 1.0, log=True),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 3),
    }

    dataset = lgb.Dataset(X_transformed, label=y)

    cv_result = lgb.cv(
        params,
        dataset,
        nfold=5,
        stratified=False,
        num_boost_round=1000,
        seed=42,
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    if 'rmse-mean' in cv_result:
        return min(cv_result['rmse-mean'])
    elif 'l2-mean' in cv_result:
        return min(cv_result['l2-mean'])
    else:
        # Print available keys to help debug
        print(f"Available keys in cv_result: {list(cv_result.keys())}")
        # Fall back to first available metric if neither expected key is found
        first_key = list(cv_result.keys())[0]
        return min(cv_result[first_key])


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest parameters:")
pprint.pprint(study.best_params)

# Train final model
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_estimators': 1000
})

model = lgb.LGBMRegressor(**best_params)
model.fit(X_transformed, y)

# Feature importances
feature_importances = name_feature_importances(model.feature_importances_, X_transformed.columns)
print("\nTop feature importances:")
pprint.pprint(feature_importances[:7])

# Predictions
y_pred = model.predict(X_test_transformed)
y_submit = model.predict(X_submit_transformed)

# Scores
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save submission
submission = pd.DataFrame({'Id': ids_submit, 'SalePrice': y_submit})
submission.to_csv(subdirectory / 'housing-submission.csv', index=False)

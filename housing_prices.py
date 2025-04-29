"""
Kagger Housing Prices
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pprint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from housing_prices_features import feature_engineering
from util import ProgressBarCallbackN, name_feature_importances

# Load the data
subdirectory = Path('housing-prices')
df_train = pd.read_csv(subdirectory / 'train.csv')
df_submit = pd.read_csv(subdirectory / 'test.csv')

# Feature engineering
df_train_fe = feature_engineering(df_train)
df_submit_fe = feature_engineering(df_submit)

# Check the data types of the features
print("\nColumn types:\n", df_train_fe.dtypes.value_counts(),"\n")
print("Float: ",df_train_fe.select_dtypes(include=['float', 'float64', 'float32']).columns.tolist())
print("Object: ",df_train_fe.select_dtypes(include=['object']).columns.tolist(),"\n")

# Train-test split
X = df_train_fe.drop(['SalePrice','Id'], axis=1)
y = df_train_fe['SalePrice']

X_submit = df_submit_fe.drop(['Id'], axis=1)
ids_submit = df_submit_fe['Id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),numerical_columns),
        ('cat', OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
            ),
        categorical_columns
        ),
    ],
    remainder='passthrough'
)

print("Preprocessing the data...\n")
preprocessor.fit(X)
X_transformed = preprocessor.transform(X)
X_test_transformed = preprocessor.transform(X_test)
X_submit_transformed = preprocessor.transform(X_submit)

lgbm_model = lgb.LGBMRegressor()

# Hyperparameter tuning
param_space = {
    'learning_rate': Real(0.08, 0.12, prior='log-uniform'),
    'n_estimators': Integer(250, 400),
    'max_depth': Categorical([3, 4, 5, 6]),
    'num_leaves': Integer(30, 60),
    'min_child_samples': Integer(2, 8),
    'subsample': Real(0.7, 0.9),
    'colsample_bytree': Real(0.2, 0.4),
    'reg_alpha': Real(0.001, 0.008, prior='log-uniform'),
    'reg_lambda': Real(0.8, 1.0, prior='log-uniform'),
    'bagging_freq': Integer(1, 3),
    'min_split_gain': Real(0.002, 0.01),
    'force_row_wise': Categorical([True]),
    'verbose': Categorical([-1]),
}

N_ITER = 2

bayes_search = BayesSearchCV(
    lgbm_model,
    param_space,
    n_iter=N_ITER,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    random_state=42,
)

fit_params = {}

# Fit the pipeline
progress_callback = ProgressBarCallbackN(N_ITER, "Bayesian Optimization")
#with suppress_output(suppress_stderr=True):
bayes_search.fit(X_transformed, y, callback=progress_callback, **fit_params)

# Show the best parameters
pprint.pprint(bayes_search.best_params_)

# Show feature importances
fe_imp = bayes_search.best_estimator_.feature_importances_
name_feature_importances(fe_imp, X_transformed.columns)

# Score for submission
y_pred = bayes_search.predict(X_test_transformed)
y_submit = bayes_search.predict(X_submit_transformed)

# Show best score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the submission
submission = pd.DataFrame({'Id': ids_submit, 'SalePrice': y_submit})
submission.to_csv(subdirectory / 'housing-submission.csv', index=False)

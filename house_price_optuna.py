"""House Price Prediction with Winkler Score Optimization using Optuna and LightGBM."""
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler

from colorama import Fore, Style
from colorama import init as colorama_init

from house_price_features import ClusterLocations, feature_clean_up
from house_price_winkler import WinklerOptimizedRegressor, winkler_score
from util import Timer

# pylint: disable=invalid-name, line-too-long
optuna.logging.set_verbosity(optuna.logging.WARNING)
colorama_init()

timer = Timer(name="Optimization iterations", start=False, verbose=True)

def optuna_logging_callback(study, trial):
    """Callback function to log the progress of the optimization."""
    print(
        f"Trial {Fore.YELLOW}{trial.number}{Style.RESET_ALL}: value={Fore.YELLOW}{trial.value:,.2f}{Style.RESET_ALL}"
        f" best_trial={Fore.YELLOW}{study.best_trial.number}{Style.RESET_ALL}"
        f" best_value={Fore.YELLOW}{study.best_value:,.2f}{Style.RESET_ALL}"
        )
    formatted_params = {k: f"{v:,.2f}" if isinstance(v, float) else v for k, v in trial.params.items()}
    print(f"params={formatted_params}")

def optuna_objective(trial, df, alpha):
    """Objective function for Optuna to minimize negative Winkler score via cross-validation."""

    # Hyperparameters to optimize
    cluster_params = {
        'n': trial.suggest_int('n', 20, 800, log=True),  # Adjusted for larger dataset
        'time_scale': trial.suggest_float('time_scale', 0.001*1/36.5, 2*1/36.5)
    }

    # Hyperparameters to optimize
    lgbm_params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 10000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 12, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 64, log=True),  # depends on max_depth
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 1000, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),  # bagging frequency
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),   # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True), # L2
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),  # minimum gain to split
        'max_bin': trial.suggest_int('max_bin', 128, 512),  # histogram binning
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1
    }

    # Split the data into training and testing sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Cluster locations
    clusters = ClusterLocations(**cluster_params)
    clusters.fit(df_train)

    # Extract features and target variable
    target = 'sale_price'
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # Assign clusters to the training and testing data
    X_train = clusters.transform(X_train)
    X_test = clusters.transform(X_test)

    # Clean up features
    X_train = feature_clean_up(X_train)
    X_test = feature_clean_up(X_test)

    # Create and fit the model with fixed alpha and the optimized parameters
    model = WinklerOptimizedRegressor(
        alpha=alpha,
        **lgbm_params
    )

    model.fit(X_train, y_train)

    # Calculate the score on validation set
    lower_bounds, upper_bounds = model.predict_interval(X_test)
    winkler = winkler_score(y_test, lower_bounds, upper_bounds, alpha)

    timer.show()
    return winkler



def optimize_winkler_predictor(df, n_trials, alpha):
    """Run the Optuna optimization process to find optimal parameters."""
    print("Starting hyperparameter optimization with n_trials:", n_trials)

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        study_name='winkler_regressor_optimization'
    )

    # Define objective function to be passed to optimize
    def objective_wrapper(trial):
        return optuna_objective(trial, df, alpha)

    # Run the optimization
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        callbacks=[optuna_logging_callback]
    )

    print(f"{Fore.GREEN}\nSummary of the hyperparameter optimization:{Style.RESET_ALL}")
    print(f'Number of finished trials: {len(study.trials)}')
    print(f'Best Winkler score: {study.best_value:.2f}')
    print(f'Best hyperparameters: {study.best_params}')

    return study.best_params, study


def run_winkler_predictor_optimization1(df, n_trials, alpha):
    """Main function to run the optimization and evaluation process."""

    timer.start()

    # Run the optimization
    best_params, study = optimize_winkler_predictor(df, n_trials=n_trials, alpha=alpha)

    timer.show()
    # Define param keys for each group
    cluster_param_keys = ['n', 'time_scale']
    lgbm_param_keys = list(set(best_params.keys()) - set(cluster_param_keys))

    # Split into sub-dictionaries
    cluster_params = {k: best_params[k] for k in cluster_param_keys}
    lgbm_params = {k: best_params[k] for k in lgbm_param_keys}
    lgbm_params['random_state'] = 42

    # Train the best model
    target = 'sale_price'
    X = df.drop(columns=[target])
    y = df[target]

    # Cluster locations again with the best parameters
    print(f"{Fore.GREEN}\nTraining best model - clustering...{Style.RESET_ALL}")
    clusters = ClusterLocations(**cluster_params)
    clusters.fit(df)

    # Assign clusters to the training and testing data
    X = clusters.transform(X)
    X = feature_clean_up(X)
    timer.show()

    # Return the best model and parameters for further use
    print(f"{Fore.GREEN}\nTraining best model - fitting predictors...{Style.RESET_ALL}")
    best_model = WinklerOptimizedRegressor(
        alpha=alpha,
        **lgbm_params
    )
    best_model.fit(X, y)
    timer.show()

    if hasattr(best_model, 'feature_names_') and best_model.feature_names_ is not None:
        print(f"{Fore.YELLOW}\nFeature Importances:{Style.RESET_ALL}")
        importances = best_model.feature_importances()
        print(importances[:30])

    return best_model, clusters, lgbm_params, cluster_params, study

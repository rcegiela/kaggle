"""House Price Prediction with Winkler Score Optimization using Optuna and LightGBM."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

# pylint: disable=invalid-name, line-too-long

def winkler_score(y_true, lower_bounds, upper_bounds, alpha):
    """Calculate the Winkler score for prediction intervals."""
    # Calculate the width of the prediction intervals
    widths = upper_bounds - lower_bounds

    # Calculate indicator for whether the true value falls within the prediction interval
    within_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)

    # Calculate the penalties for values outside the interval
    penalty_below = 2/alpha * (lower_bounds - y_true) * (~within_interval & (y_true < lower_bounds))
    penalty_above = 2/alpha * (y_true - upper_bounds) * (~within_interval & (y_true > upper_bounds))

    # Calculate the final Winkler score (lower is better)
    winkler = np.mean(widths + penalty_below + penalty_above)

    return winkler


class WinklerOptimizedRegressor(BaseEstimator, RegressorMixin):
    """A regressor that produces prediction intervals optimized for the Winkler score."""

    def __init__(self, alpha=0.1, n_estimators=500, learning_rate=0.05, max_depth=None,
                 min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
                 reg_alpha=0.0, reg_lambda=0.0, random_state=42, verbose=-1, n_jobs=-1):
        """Initialize the regressor with properly defined parameters."""
        self.alpha = alpha  # Fixed at 0.1 as specified
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.verbose = verbose
        self.feature_names_ = None
        self.n_jobs = n_jobs

        self.lower_model = None
        self.upper_model = None
        self.point_model = None

    def _get_lgbm_params(self):
        """Get parameters for LightGBM models."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs
        }

    def fit(self, X, y):
        """Fit the regressor to the training data."""
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()

        # Create the models with appropriate parameters
        params = self._get_lgbm_params()

        # Create the lower bound model
        self.lower_model = LGBMRegressor(
            objective='quantile',
            alpha=self.alpha/2,
            **params
        )

        # Create the upper bound model
        self.upper_model = LGBMRegressor(
            objective='quantile',
            alpha=1-self.alpha/2,
            **params
        )

        # Create the point prediction model
        self.point_model = LGBMRegressor(
            objective='regression',
            **params
        )

        # Fit all models
        if hasattr(X, 'columns'):
            # Pass feature names directly to LightGBM
            self.lower_model.fit(X, y, feature_name=self.feature_names_)
            self.upper_model.fit(X, y, feature_name=self.feature_names_)
            self.point_model.fit(X, y, feature_name=self.feature_names_)
        else:
            self.lower_model.fit(X, y)
            self.upper_model.fit(X, y)
            self.point_model.fit(X, y)

        return self

    def predict(self, X):
        """Predict the point estimates for the input samples."""
        return self.point_model.predict(X)

    def predict_interval(self, X):
        """Predict the lower and upper bounds for the input samples."""
        lower_bounds = self.lower_model.predict(X)
        upper_bounds = self.upper_model.predict(X)

        # Ensure lower bounds are actually lower than upper bounds
        inconsistent = lower_bounds > upper_bounds
        if np.any(inconsistent):
            # Swap values where inconsistent
            temp = lower_bounds[inconsistent]
            lower_bounds[inconsistent] = upper_bounds[inconsistent]
            upper_bounds[inconsistent] = temp

        return lower_bounds, upper_bounds

    def score(self, X, y): # pylint: disable=arguments-differ
        """Calculate the negative Winkler score for the input samples."""
        lower_bounds, upper_bounds = self.predict_interval(X)
        return -winkler_score(y, lower_bounds, upper_bounds, self.alpha)

    def feature_importances(self, sort_by='lower_model', ascending=False):
        """Get feature importances from all models."""
        if not hasattr(self.point_model, 'feature_importances_'):
            raise ValueError("Models have not been trained yet.")

        importances = {
            'point_model': self.point_model.feature_importances_,
            'lower_model': self.lower_model.feature_importances_,
            'upper_model': self.upper_model.feature_importances_
        }

        if self.feature_names_ is not None:
            results = {}
            for model_name, imp in importances.items():
                results[model_name] = pd.Series(imp, index=self.feature_names_)
            df = pd.DataFrame(results)
            return df.sort_values(by=sort_by, ascending=ascending)
        else:
            return importances


def optuna_objective(trial, X, y, cv_folds=5, alpha=0.1):
    """Objective function for Optuna to minimize negative Winkler score via cross-validation."""
    # Fixed alpha as specified
    alpha = 0.1

    # Hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }

    # Cross-validation setup
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []

    # Get indices for cross-validation
    indices = np.arange(len(X))

    print(f"Starting cross-validation with {cv_folds} folds:", end='')
    for train_idx, val_idx in kf.split(indices):
        print(len(scores) + 1, end=' ')
        if hasattr(X, 'iloc'):  # Check if X is a DataFrame
            # Use .iloc for pandas DataFrame/Series to select rows by position
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

            if hasattr(y, 'iloc'):  # Check if y is a Series
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:  # y is a numpy array
                y_train, y_val = y[train_idx], y[val_idx]
        else:  # Handle numpy arrays
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # Create and fit the model with fixed alpha and the optimized parameters
        model = WinklerOptimizedRegressor(
            alpha=alpha,
            **params
        )

        model.fit(X_train, y_train)

        # Calculate the score on validation set
        lower_bounds, upper_bounds = model.predict_interval(X_val)
        winkler = winkler_score(y_val, lower_bounds, upper_bounds, alpha)
        scores.append(winkler)

    # Return the mean score across all folds
    mean_score = np.mean(scores)
    print(f"Mean Winkler score for this trial: {mean_score:.2f}")

    return mean_score


def optimize_winkler_predictor(X, y, n_trials=50, cv_folds=5, alpha=0.1):
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
        return optuna_objective(trial, X, y, cv_folds, alpha)

    # Run the optimization
    study.optimize(
        objective_wrapper,
        n_trials=n_trials
    )

    print(f'Number of finished trials: {len(study.trials)}')
    print(f'Best Winkler score: {study.best_value:.2f}')
    print(f'Best hyperparameters: {study.best_params}')

    return study.best_params, study


def evaluate_best_model(X_train, y_train, X_test, y_test, best_params, alpha):
    """Evaluate the model with the best parameters on test data."""
    # Create model with best parameters
    model = WinklerOptimizedRegressor(
        alpha=alpha,
        **best_params
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate on test data
    lower_bounds, upper_bounds = model.predict_interval(X_test)
    test_winkler = winkler_score(y_test, lower_bounds, upper_bounds, alpha)
    point_preds = model.predict(X_test)

    print(f'Winkler score on test data: {test_winkler:.2f}')
    print(f'Mean interval width: {np.mean(upper_bounds - lower_bounds):.2f}')
    print(f'Coverage rate: {np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds)):.2f}')
    print(f'Target coverage rate: {1-alpha:.2f}')

    # Display feature importances if available
    if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
        print("\nFeature Importances:")
        importances = model.feature_importances()
        print(importances)

    return test_winkler, model, point_preds, lower_bounds, upper_bounds


def plot_prediction_results(y_test, point_preds, lower_bounds, upper_bounds, n_samples=100):
    """Plot the prediction intervals and point predictions against the true values."""
    # Convert to numpy arrays if they're pandas Series
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    # Select random samples if there are more points than n_samples
    if len(y_test) > n_samples:
        np.random.seed(42)  # For reproducibility
        idx = np.random.choice(range(len(y_test)), n_samples, replace=False)
        y_test = y_test[idx]
        point_preds = point_preds[idx]
        lower_bounds = lower_bounds[idx]
        upper_bounds = upper_bounds[idx]

    # Sort everything by true values for better visualization
    sort_idx = np.argsort(y_test)
    y_sorted = y_test[sort_idx]
    point_sorted = point_preds[sort_idx]
    lower_sorted = lower_bounds[sort_idx]
    upper_sorted = upper_bounds[sort_idx]

    # Set font size to 10pt
    plt.rcParams.update({'font.size': 10})

    plt.figure(figsize=(6, 3))

    # Plot the true values
    plt.scatter(range(len(y_sorted)), y_sorted, label='True values', color='blue', alpha=0.7, s=10)

    # Plot the point predictions
    plt.scatter(range(len(point_sorted)), point_sorted, label='Point predictions', color='red', alpha=0.7, s=10)

    # Plot the prediction intervals
    plt.fill_between(range(len(lower_sorted)), lower_sorted, upper_sorted,
                     alpha=0.2, label='90% Prediction interval', color='blue')

    plt.xlabel('Samples (sorted by true value)', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title('True values, Point Predictions and Prediction Intervals', fontsize=10)
    plt.tick_params(axis='both', labelsize=8)
    plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.show()


def run_winkler_predictor_optimization(df, n_trials=50, cv_folds=5, alpha=0.1):
    """Main function to run the optimization and evaluation process."""
    target = 'sale_price'
    X = df.drop(columns=[target])
    y = df[target]

    # Keep as DataFrames/Series rather than converting to numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run the optimization
    best_params, study = optimize_winkler_predictor(X, y, n_trials=n_trials, cv_folds=cv_folds, alpha=alpha)

    # Plot the optimization history and parameter importances
    ax1 = optuna.visualization.matplotlib.plot_optimization_history(study) # pylint: disable=unused-variable
    ax2 = optuna.visualization.matplotlib.plot_param_importances(study) # pylint: disable=unused-variable

    # Evaluate the best model
    print("\nEvaluating the best model on test data...")
    test_winkler, test_model, point_preds, lower_bounds, upper_bounds = evaluate_best_model( # pylint: disable=unused-variable
        X_train, y_train, X_test, y_test, best_params, alpha=alpha
    )

    # Plot the results with specific figure size
    plot_prediction_results(y_test, point_preds, lower_bounds, upper_bounds)

    print("\nOptimization and evaluation complete!")
    print("Final optimized parameters (with alpha fixed at 0.1):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Return the best model and parameters for further use
    best_model = WinklerOptimizedRegressor(
        alpha=alpha,
        **best_params
    )
    best_model.fit(X, y)

    return best_model, best_params

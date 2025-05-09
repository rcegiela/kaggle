"""House Price Prediction with Winkler Score Optimization."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin

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

    def __init__(self, alpha=0.1, n_estimators=500, learning_rate=0.05, max_depth=8,
                 num_leaves=None, min_child_samples=20, min_child_weight=1e-3,
                 subsample=1.0, subsample_freq=1,
                 colsample_bytree=1.0, colsample_bynode=0.8,
                 reg_alpha=0.0, reg_lambda=0.0, min_split_gain=0.0,
                 max_bin=255, boosting_type='gbdt',
                 random_state=42, verbose=-1, n_jobs=-1):
        """Initialize the regressor with defined parameters."""
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves if num_leaves is not None else 2 ** max_depth - 1
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.max_bin = max_bin
        self.boosting_type = boosting_type
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.feature_names_ = None
        self.lower_model = None
        self.upper_model = None
        self.point_model = None

    def _get_lgbm_params(self):
        """Get parameters for LightGBM models."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bynode': self.colsample_bynode,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_split_gain': self.min_split_gain,
            'max_bin': self.max_bin,
            'boosting_type': self.boosting_type,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs
        }

    def fit(self, X, y):
        """Fit the regressor to the training data."""
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns.tolist()

        params = self._get_lgbm_params()

        self.lower_model = LGBMRegressor(objective='quantile', alpha=self.alpha / 2, **params)
        self.upper_model = LGBMRegressor(objective='quantile', alpha=1 - self.alpha / 2, **params)
        self.point_model = LGBMRegressor(objective='regression', **params)

        if hasattr(X, 'columns'):
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

        inconsistent = lower_bounds > upper_bounds
        if np.any(inconsistent):
            temp = lower_bounds[inconsistent]
            lower_bounds[inconsistent] = upper_bounds[inconsistent]
            upper_bounds[inconsistent] = temp

        return lower_bounds, upper_bounds

    def score(self, X, y):
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

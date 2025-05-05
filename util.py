"""
Utility functions for optimization and feature importance analysis.
"""
import time
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm


class ProgressBarCallback:
    """
    A callback class to display a progress bar for optimization processes.
    Jupyter notebook version.
    """
    def __init__(self, n_iter, desc='Optimization Progress', create_bar=True):
        if create_bar:
            self.pbar = tqdm(total=n_iter, desc=desc)
        self.iter_count = 0
        self.start_time = time.time()

    def __call__(self, res):
        self.iter_count += 1
        elapsed = time.time() - self.start_time
        self.pbar.set_postfix({
            'best_score': f'{-res.fun:.2f}',
            'elapsed': f'{elapsed:.1f}s'
        })
        self.pbar.update(1)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()


class ProgressBarCallbackN(ProgressBarCallback):
    """
    A callback class to display a progress bar for optimization processes.
    """
    def __init__(self, n_iter, desc='Optimization Progress'):
        super().__init__(n_iter, desc, create_bar=False)
        self.pbar = tqdm_n(total=n_iter, desc=desc)


def name_feature_importances(feature_importances, feature_names):
    """
    This function takes a list of feature importances and their corresponding names
    """
    features_with_importance = list(zip(feature_names, feature_importances))
    print("No of features:", len(features_with_importance))
    return sorted(features_with_importance, key=lambda x: x[1], reverse=True)

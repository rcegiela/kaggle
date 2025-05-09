"""
Utility functions for optimization and feature importance analysis.
"""
import time
from datetime import datetime
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
from colorama import Fore, Style

# pylint: disable=invalid-name, line-too-long

class Timer:
    """A simple timer class to measure elapsed time."""
    def __init__(self, name=None, start=False, verbose=False):
        self.name = name
        self.verbose = verbose

        if start:
            self.start()
        else:
            self.start_time = None
            self.mid_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.mid_time = self.start_time
        if self.verbose:
            dt = datetime.fromtimestamp(self.start_time)
            print(f"Timer {self.name} initialized at {dt.strftime('%Y-%m-%d %H:%M:%S')}.")


    def elapsed(self):
        """Get the elapsed time since the timer was started."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        elapsed_time = time.time() - self.start_time
        elapsed_mid_time = time.time() - self.mid_time
        self.mid_time = time.time()
        return elapsed_time, elapsed_mid_time

    def stop(self):
        """Reset the timer."""
        self.start_time = None

    def show(self):
        """Print the elapsed time in a formatted string."""
        elapsed_time, elapsed_mid_time = self.elapsed()
        print(
            f"{self.name+': ' if self.name else ''}Elapsed time: "
            f"{Fore.RED}{elapsed_time:.2f}{Style.RESET_ALL} sec "
            f"({Fore.RED}{elapsed_mid_time:.2f}{Style.RESET_ALL} sec)"
            )


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

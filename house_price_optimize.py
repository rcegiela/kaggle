"""House Price Prediction - Optimization Script"""
from pathlib import Path
import pandas as pd

from colorama import Fore, Style
from colorama import init as colorama_init

from house_price_features import feature_engineering, feature_clean_up
from house_price_optuna import run_winkler_predictor_optimization1
from util import Timer

# pylint: disable=invalid-name, line-too-long
colorama_init()
timer=Timer(name="Overall", start=True, verbose=True)

# Load the data
print(Fore.GREEN+"Loading data..."+Style.RESET_ALL)
subdirectory = Path('house-price')
df_train = pd.read_csv(subdirectory / 'dataset.csv')
df_test = pd.read_csv(subdirectory / 'test.csv')
print("No of training records: ", len(df_train))
print("No of features: ", len(df_train.columns))
timer.show()
#df_train.head()

# Feature engineering
print(Fore.GREEN+"Feature engineering..."+Style.RESET_ALL)
df_train_fe=feature_engineering(df_train)
df_test_fe=feature_engineering(df_test)
print("No of features: ", len(df_train_fe.columns))
timer.show()

# Optimizze Winkler predictor
print(Fore.GREEN+"Optimizing Winkler predictor..."+Style.RESET_ALL)
best_model, clusters, _, _, _ = run_winkler_predictor_optimization1(df_train_fe, n_trials=500, alpha=0.1)
timer.show()

# Apply clustering
print(Fore.GREEN+"Clustering for submission..."+Style.RESET_ALL)
X_test = clusters.transform(df_test_fe)
X_test = feature_clean_up(X_test)
timer.show()

# Submit results
print(Fore.GREEN+"Predicting for submission..."+Style.RESET_ALL)
lower_bounds, upper_bounds = best_model.predict_interval(X_test)

result_df = pd.DataFrame({
        'id': df_test_fe['id'],
        'pi_lower': lower_bounds,
        'pi_upper': upper_bounds
    })

result_df.to_csv(subdirectory / 'submission.csv', index=False)
timer.show()

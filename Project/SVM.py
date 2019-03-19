import pandas as pd
import numpy as np
import math, datetime, time
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
warnings.filterwarnings("ignore")

style.use('ggplot')

# run download_dataset.py first!
df = pd.read_csv('neo_price.csv')

# compute new stats to use as features
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
date = df['Date']
# we are only going to use those columns
df = df[['Date','Close', 'HL_PCT', 'PCT_change', 'Volume']]
df.set_index('Date', inplace=True)
forecast_col = 'Close'
#forecast_col = 'PCT_change'
df.fillna(-99999, inplace=True)

forecast_out = 10
X = np.array(df)
X = preprocessing.scale(X)  # normalization
y = np.array(df[forecast_col])

# training
X_train = X[:-2 * forecast_out]
y_train = y[forecast_out:-forecast_out]
X_train, X_test, y_train, y_test = tts(
X_train, y_train, test_size=0.1)

clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)
conf = clf.score(X_test, y_test)
print('Confidence:',conf)

X_eval30 = X[-2 * forecast_out:-forecast_out]
y_eval_forecast = clf.predict(X_eval30)

df['Prediction'] = np.nan
df['Prediction'][-forecast_out:] = y_eval_forecast

# predict days into future
X_pred30 = X[-forecast_out:]
y_pred_forecast = clf.predict(X_pred30)

print(y_pred_forecast,conf,forecast_out)
plt.xticks(np.arange(0, 1611, step=1),date)
df['Close'].plot()
df['Prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


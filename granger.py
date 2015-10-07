import math
import datetime
import numpy
import scipy
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
from statsmodels.tsa import ar_model
from statsmodels.tsa import arima_model
from statsmodels.stats import diagnostic
from statsmodels.regression.linear_model import OLS
import statsmodels
import _math

"""
Eric Salina
"""

####################CONSTANTS###################
MAX_LAG = 5


####################LOAD DATA###################
clicksPerDay = pd.read_csv("clicksPerDay.csv", header=0, index_col=0, parse_dates=True)
encountersPerDay = pd.read_csv("encountersPerDay.csv", header=0, index_col=0, parse_dates=True)

#set index to the range between Jan 1 2011 - Dec 31 2012, since data may miss any day
#without any clicks or encounters (though not likely).
newIndex = pd.date_range(start=datetime.datetime(2011, 1, 1), end=datetime.datetime(2012, 12, 1)).rename("Date")
clicksPerDay = clicksPerDay.reindex(newIndex)
encountersPerDay = encountersPerDay.reindex(newIndex)

#convert to string, since stattools is unhappy with datetimes
newIndex = pd.Index(data=[dt.strftime("%m/%d/%y") for dt in newIndex])
clicksPerDay = clicksPerDay.set_index(newIndex)
encountersPerDay = encountersPerDay.set_index(newIndex)

#clicksPerDay.index.name = None
#encountersPerDay.index.name = None

clicksPerDay = pd.Series(data=clicksPerDay["count_clicks"], index=clicksPerDay.index)
encountersPerDay = pd.Series(data=encountersPerDay["count_encounter"], index=encountersPerDay.index)

clicksPerDay = clicksPerDay.fillna(method="ffill")
encountersPerDay = encountersPerDay.fillna(method="ffill")



#########GRANGER############

#to test for AR model with multiple lags, the actual AR object (or ARMA for
#that matter) does not work well. It only gives one parameter for the exog
#data. We can fix this by using the OLS function, with mulitple regressors.
#We can create lists of the exog and endog vars with lagged values of their
#original lists respectively. We can then pass these in as additional
#regressors to the OLS function through the `endog` argument, and achieve
#essentially the same thing that ARMA didn't give us. meh

endog = encountersPerDay.tolist()
exog = clicksPerDay.tolist()

endogLags = []
exogLags = []

for lag in range(0, MAX_LAG+1):
	end = -(MAX_LAG-lag) if -(MAX_LAG-lag) != 0 else len(endog)
	laggedEndog = endog[lag:end]
	endogLags.append(laggedEndog)
	laggedExog = exog[lag:end]
	exogLags.append(laggedExog)

endogLags.reverse()
exogLags.reverse()

endog = endogLags.pop(0)
exog = exogLags.pop(0)

results = OLS(endog, exog=exog).fit()
print results.fvalue
print results.f_pvalue

#calculating f value by hand based on:
#http://connor-johnson.com/2014/02/18/linear-regression-with-python/
N = results.nobs
P = results.df_model
dfn, dfd = P, N-P-1
F = results.mse_model / results.mse_resid
p = 1.0 - scipy.stats.f.cdf(F, dfn, dfd)
print F
print p
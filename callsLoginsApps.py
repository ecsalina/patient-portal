import math
import datetime
import numpy as np
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

# ####################CONSTANTS###################
MAX_LAG = 30


# ####################LOAD DATA###################
x = pd.read_csv("loginsPerDay.csv", header=None, index_col=0, parse_dates=True)
y = pd.read_csv("callsPerDay.csv", header=None, index_col=0, parse_dates=True)
z = pd.read_csv("appsPerDay.csv", header=None, index_col=0, parse_dates=True)

#set index to the range between Jan 1 2011 - Feb 1 2013, since data may miss some days
newIndex = pd.date_range(start=datetime.datetime(2011, 1, 1), end=datetime.datetime(2013, 2, 1)).rename("Date")
x = x.reindex(newIndex)
y = y.reindex(newIndex)
z = z.reindex(newIndex)

#convert to string, since stattools is unhappy with datetimes
newIndex = pd.Index(data=[dt.strftime("%m/%d/%y") for dt in newIndex])
x = x.set_index(newIndex)
y = y.set_index(newIndex)
z = z.set_index(newIndex)

x = pd.Series(data=x[1], index=x.index)
y = pd.Series(data=y[1], index=y.index)
z = pd.Series(data=z[1], index=z.index)

x = x.fillna(0)
y = y.fillna(0)
z = z.fillna(0)


newIndex = [x.index[i] for i in range (0, len(z.index), 7)]

listX = [x[i:i+7] for i in range(0, len(x), 7)]
xWeekly = []
for week in listX:
	val = week.sum()
	xWeekly.append(val)
x = pd.Series(xWeekly, index=newIndex)

listY = [y[i:i+7] for i in range(0, len(y), 7)]
yWeekly = []
for week in listY:
	val = week.sum()
	yWeekly.append(val)
y = pd.Series(yWeekly, index=newIndex)

listZ = [z[i:i+7] for i in range(0, len(z), 7)]
zWeekly = []
for week in listZ:
	val = week.sum()
	zWeekly.append(val)
z = pd.Series(zWeekly, index=newIndex)


print(x)
print(y)
print(z)


# ####################PRE-ANALYSIS####################
#1. Augmented Dickey-Fuller test for unit root to test if stationary (if not,
#take I(2) series, etc.).
results = stattools.adfuller(x)
print("ADF logins per day:")
print("stat: "+str(results[0]))
print("pval: "+str(results[1]))

results = stattools.adfuller(y)
print("ADF calls per day:")
print("stat: "+str(results[0]))
print("pval: "+str(results[1]))

results = stattools.adfuller(z)
print("ADF apps per day:")
print("stat: "+str(results[0]))
print("pval: "+str(results[1]))

#2. Chi-Square for seasonality
#group values by day
def testChiSquare(series):
	days = series.index.tolist()
	vals = series.tolist()
	data = zip(days, vals)
	bins = [0 for i in range(7)]
	for line in data:
		bins[datetime.datetime.strptime(line[0], "%m/%d/%y").weekday()] += line[1]

	exp = [sum(vals)/7 for i in range (7)]
	print exp

	chiSquare = 0
	for i in range(7):
		chiSquare += (bins[i] - exp[i])**2/exp[i]

	print chiSquare
	print scipy.stats.chi2.sf(chiSquare, 6)

print("logins seasonality")
testChiSquare(x)
print("calls seasonality")
testChiSquare(y)
print("apps seasonality")
testChiSquare(z)

############COINTEGRATION TEST############
#test to see if both series move in same general direction.
#fit OLS to both series, and run ADF on residuals. Stationary residuals
#indicate that the series is cointegrate, while nonstationary means they
#are not cointegrated.
#x=logins, y-calls
results = OLS(x.values, y.values).fit()
print(results.params)
print(results.pvalues)
resultsADF = stattools.adfuller(results.resid)
print(resultsADF)
#this adf test is opposite of standard. H0 -> non-stationary, while
#ha -> stationary
#want small p-value

###############GRANGER TEST#################
def grangerTest(exog, endog):
	MAX_LAG = 30
	ARaic = ar_model.AR(exog.tolist()).fit(maxlag=MAX_LAG, ic="aic")
	ARbic = ar_model.AR(exog.tolist()).fit(maxlag=MAX_LAG, ic="bic")
	#select the fewer number of parameters between both criteria.
	numExog = len(ARaic.params) if len(ARaic.params) < len(ARbic.params) else len(ARbic.params)

	print("Optimal number of lags for exog data is "+str(numExog))

	ARaic = ar_model.AR(endog.tolist()).fit(maxlag=MAX_LAG, ic="aic")
	ARbic = ar_model.AR(endog.tolist()).fit(maxlag=MAX_LAG, ic="bic")
	#select the fewer number of parameters between both criteria.
	numEndog = len(ARaic.params) if len(ARaic.params) < len(ARbic.params) else len(ARbic.params)

	print("Optimal number of lags for endog data is "+str(numEndog))

	#now that I know the optimal number of parameters, I can call the
	#granger causality function of statsmodels.
	data = pd.concat([endog, exog], axis=1)
	print("\nGranger causality results of indep onto dep")
	results = stattools.grangercausalitytests(data, maxlag=numEndog)

	data = pd.concat([exog, endog], axis=1)
	print("\nGranger causality results of dep onto indep")
	results = stattools.grangercausalitytests(data, maxlag=numExog)
	regr = results[2][1]
	print(regr[0].params)
	print(regr[1].params)
	print(regr[1].pvalues)

print("indep = calls, dep = logins")
grangerTest(y, x)




################# PLOT ##################
def plotxy(title, x, y):
	fig = plt.figure()
	plt.plot(x, y, color="orange")
	plt.grid()
	plt.title(title)
	fig.autofmt_xdate()
	plt.show()

#plotxy("Aggregate Weekly Logins 2011-Feb. 2013", range(len(x)), x.values)
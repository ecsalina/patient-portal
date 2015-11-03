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
cats = pd.read_csv("patientCategories.csv", header=None, index_col=0).iloc[:,0]
#order: ID, category
x = pd.read_csv("loginsPerDayPerUser.csv", header=None, index_col=0, parse_dates=True)
y = pd.read_csv("callsPerDayPerUser.csv", header=None, index_col=0, parse_dates=True)
z = pd.read_csv("appsPerDayPerUser.csv", header=None, index_col=0, parse_dates=True)
#order: date, user

#test how many IDs not included
#print z.loc[z['cat'] == 0]

#assigns the correct category to each row
x['cat'] = x[1].apply(lambda ID: cats.loc[ID] if ID in cats.index else 0)
y['cat'] = y[1].apply(lambda ID: cats.loc[ID] if ID in cats.index else 0)
z['cat'] = z[1].apply(lambda ID: cats.loc[ID] if ID in cats.index else 0)

#divides time series amongst categories, and counts number of people each day
x1 = x.loc[x['cat'] == 1].drop('cat', axis=1)
x1 = x1.groupby(x1.index).count()
x2 = x.loc[x['cat'] == 2].drop('cat', axis=1)
x2 = x2.groupby(x2.index).count()
x3 = x.loc[x['cat'] == 3].drop('cat', axis=1)
x3 = x3.groupby(x3.index).count()
x4 = x.loc[x['cat'] == 4].drop('cat', axis=1)
x4 = x4.groupby(x4.index).count()

y1 = y.loc[y['cat'] == 1].drop('cat', axis=1)
y1 = y1.groupby(y1.index).count()
y2 = y.loc[y['cat'] == 2].drop('cat', axis=1)
y2 = y2.groupby(y2.index).count()
y3 = y.loc[y['cat'] == 3].drop('cat', axis=1)
y3 = y3.groupby(y3.index).count()
y4 = y.loc[y['cat'] == 4].drop('cat', axis=1)
y4 = y4.groupby(y4.index).count()

z1 = z.loc[z['cat'] == 1].drop('cat', axis=1)
z1 = z1.groupby(z1.index).count()
z2 = z.loc[z['cat'] == 2].drop('cat', axis=1)
z2 = z2.groupby(z2.index).count()
z3 = z.loc[z['cat'] == 3].drop('cat', axis=1)
z3 = z3.groupby(z3.index).count()
z4 = z.loc[z['cat'] == 4].drop('cat', axis=1)
z4 = z4.groupby(z4.index).count()


#set index to the range between Jan 1 2011 - Feb 1 2013, since data may miss some days
newIndex = pd.date_range(start=datetime.datetime(2011, 1, 1), end=datetime.datetime(2013, 2, 1)).rename("Date")
x1 = x1.reindex(newIndex)
x2 = x2.reindex(newIndex)
x3 = x3.reindex(newIndex)
x4 = x4.reindex(newIndex)

y1 = y1.reindex(newIndex)
y2 = y2.reindex(newIndex)
y3 = y3.reindex(newIndex)
y4 = y4.reindex(newIndex)

z1 = z1.reindex(newIndex)
z2 = z2.reindex(newIndex)
z3 = z3.reindex(newIndex)
z4 = z4.reindex(newIndex)


#convert to string, since stattools is unhappy with datetimes
def finishConversion(q):
	newIndex2 = pd.Index(data=[dt.strftime("%m/%d/%y") for dt in newIndex])
	q = q.set_index(newIndex2)
	q = pd.Series(data=q[1], index=q.index)
	q = q.fillna(0)
	return q

x1 = finishConversion(x1)
x2 = finishConversion(x2)
x3 = finishConversion(x3)
x4 = finishConversion(x4)

y1 = finishConversion(y1)
y2 = finishConversion(y2)
y3 = finishConversion(y3)
y4 = finishConversion(y4)

z1 = finishConversion(z1)
z2 = finishConversion(z2)
z3 = finishConversion(z3)
z4 = finishConversion(z4)

def convertToWeek(q):
	newIndex = [q.index[i] for i in range (0, len(q.index), 7)]

	listX = [q[i:i+7] for i in range(0, len(q), 7)]
	xWeekly = []
	for week in listX:
		val = week.sum()
		xWeekly.append(val)
	q = pd.Series(xWeekly, index=newIndex)
	return q

x1 = convertToWeek(x1)
x2 = convertToWeek(x2)
x3 = convertToWeek(x3)
x4 = convertToWeek(x4)

y1 = convertToWeek(y1)
y2 = convertToWeek(y2)
y3 = convertToWeek(y3)
y4 = convertToWeek(y4)

z1 = convertToWeek(z1)
z2 = convertToWeek(z2)
z3 = convertToWeek(z3)
z4 = convertToWeek(z4)








x = x4
y = y4
z = z4

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

print("------------------\n\n\n")

print("indep = apps, dep = logins")
grangerTest(z, x)

print("------------------\n\n\n")

print("indep = apps, dep = calls")
grangerTest(z, y)

print("------------------\n\n\n")




################# PLOT ##################
def plotxy(title, x, y):
	fig = plt.figure()
	plt.plot(x, y, color="orange")
	plt.grid()
	plt.title(title)
	fig.autofmt_xdate()
	plt.show()

#plotxy("Aggregate Weekly Logins 2011-Feb. 2013", range(len(x)), x.values)



#plot all clusters for entire period
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
# ax1.plot(pd.to_datetime(z1.index), z1.values, color="orange")
# ax1.grid()
# ax1.set_title("Cluster 1")
# ax2.plot(pd.to_datetime(z2.index), z2.values, color="orange")
# ax2.grid()
# ax2.set_title("Cluster 2")
# ax3.plot(pd.to_datetime(z3.index), z3.values, color="orange")
# ax3.grid()
# ax3.set_title("Cluster 3")
# ax4.plot(pd.to_datetime(z4.index), z4.values, color="orange")
# ax4.grid()
# ax4.set_title("Cluster 4")
# fig.suptitle("Aggregate Weekly Appointments by Cluster, 2011-Feb. 2013")
# plt.show()



# ax4.plot(pd.to_datetime(pd.rolling_mean(x4,20).index), pd.rolling_mean(x4,20).values, color="blue")











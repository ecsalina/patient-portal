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
import _math

"""
Eric Salina
9/24/15
WPI

Below I carry out a Granger causality test to determine whether the clicks
data or encounter data from an online patient portal "granger cause" the other.

I start by loading the data and analyzing the PACF of each, as well as the
cross correlation of each. The PACF should give some intuitive sense about
roughly how many lags should be used in the AR models, while the cross-
correlation gives a sense of how good the past values of each time series are
at predicting the values of the other series. The Ljunge-Box test for each of
these gives quantifiable estimates of the significance of each lag.

Next I follow roughly the steps outlined by Dave Giles on his blog at:
http://davegiles.blogspot.com/2011/04/testing-for-granger-causality.html
for conducting the Granger causality test.

First, I must determine the appropriate lag for each AR model. Because this
would be difficult to compute alone, I use the SciPy AR model fit function,
which allows me to specify which criterion I would like to use and
returns an AR model with the appropriate number of lags.

I then use these numbers of lags to introduce the appriopriate number of lags
as exogenous variables in a new AR model for each time series.

With these new time series, I test whether the added parameters are
significantly different from zero by using an F-test. If any are, then
there is evidence for correlation, or "granger causality" between from on
time series unto the other.
These final steps I simplify by implementing the statsmodels function for
Granger Causality

"""


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

clicksPerDay.index.name = None
encountersPerDay.index.name = None

clicksPerDay = pd.Series(data=clicksPerDay["count_clicks"], index=clicksPerDay.index)
encountersPerDay = pd.Series(data=encountersPerDay["count_encounter"], index=encountersPerDay.index)



####################PRE-ANALYSIS####################
#1. Augmented Dickey-Fuller test for unit root to test if stationary (if not,
#take I(2) series, etc.).
#not implemented since is not entirely needed for Granger test.


#2. difference once to I(1) series which are roughly stationary
#(e ~iid N(0,sigma^2))
#not implemented since is not entirely necesary for Granger test.


#3. calculate partial autocorrelation function (PACF) w/ Ljung-Box test to see
#which lags help predict current values.
clicksPACF = stattools.pacf_ols(clicksPerDay, nlags=120)
encountersPACF = stattools.pacf_ols(encountersPerDay, nlags=120)

#Ljung-Box test
clicksPACF_LJ = _math.ljungBox(clicksPACF, len(clicksPerDay), 120)
encountersPACF_LJ = _math.ljungBox(encountersPACF, len(encountersPerDay), 120)
print("clicks PACF Ljung-Box")
print(tabulate(clicksPACF_LJ, headers=["lag", "R", "Q", "p-val"]))
print("encounters PACF Ljung-Box")
print(tabulate(encountersPACF_LJ, headers=["lag", "R", "Q", "p-val"]))

#plot
def plotxy(title, x, y, LSRL=None):
	fig = plt.figure()
	plt.bar(x, y, width=0.08)
	plt.grid()
	plt.title(title)
	fig.autofmt_xdate()
	plt.show()

plotxy("clicks", range(7), clicksPACF)
plotxy("encounters", range(7), encountersPACF)


#4. calculate cross-correlation functions
"""
I could use the following function, however as the lags progress, I find
that the R values diverge from my implementation. I know that my implementation
is not wrong, so I select my implementation for use.

clicksCCF = stattools.ccf(clicksPerDay, encountersPerMont)
Note: first var is dependent, second is independent, which is unclear from 
documentation
"""

clicksCCF = _math.cc_ols(clicksPerDay, encountersPerDay, 120)
encountersCCF = _math.cc_ols(encountersPerDay, clicksPerDay, 120)

#Ljung-Box test
clicksCCF_LJ = _math.ljungBox(clicksCCF, len(clicksPerDay), 120)
encountersCCF_LJ = _math.ljungBox(encountersCCF, len(encountersPerDay), 120)
print("clicks CCF Ljung-Box")
print(tabulate(clicksCCF_LJ, headers=["lag", "R", "Q", "p-val"]))
print("encounters CCF Ljung-Box")
print(tabulate(encountersCCF_LJ, headers=["lag", "R", "Q", "p-val"]))






####################ANALYSIS####################
#1. select appriopriate number of lags
ARaic = ar_model.AR(clicksPerDay.tolist()).fit(maxlag=120, ic="aic")
ARbic = ar_model.AR(clicksPerDay.tolist()).fit(maxlag=120, ic="bic")
#select the fewer number of parameters between both criteria.
numLagsclick = len(ARaic.params) if len(ARaic.params) < len(ARbic.params) else len(ARbic.params)

print("Optimal number of lags for click data is "+str(numLagsclick))

ARaic = ar_model.AR(encountersPerDay.tolist()).fit(maxlag=120, ic="aic")
ARbic = ar_model.AR(encountersPerDay.tolist()).fit(maxlag=120, ic="bic")
#select the fewer number of parameters between both criteria.
numLagsEnc = len(ARaic.params) if len(ARaic.params) < len(ARbic.params) else len(ARbic.params)

print("Optimal number of lags for encounter data is "+str(numLagsEnc))


#2. now that I know the optimal number of parameters, I can call the
#granger causality function of statsmodels.
data = pd.concat([encountersPerDay, clicksPerDay], axis=1)
print("\nGranger causality results of clicks onto encounters")
results = stattools.grangercausalitytests(data, maxlag=numLagsEnc)

data = pd.concat([clicksPerDay, encountersPerDay], axis=1)
print("\nGranger causality results of encounters onto clicks")
results = stattools.grangercausalitytests(data, maxlag=numLagsclick)

#ideally, I would implement this myself, however statsmodels is buggy and
#does not deal with exogenous variables well, meaning I would have to
#implement an AR fitting algorithm, which is non-ideal.
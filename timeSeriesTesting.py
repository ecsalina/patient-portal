import math
import numpy
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
from statsmodels.tsa import ar_model
from statsmodels.tsa import arima_model
from tabulate import tabulate
import _math

"""
Eric Salina
9/24/15
WPI

Below I carry out a Granger causality test to determine whether the sessions
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
which allows me to specify which criterion I would like to use (BIC here) and
returns an AR model with the appropriate number of lags.

I then use these numbers of lags to introduce the appriopriate number of lags
as exogenous variables in a new AR model for each time series.

With these new time series, I test whether the added parameters are
significantly different from zero by using an ANOVA test. If any are, then
there is evidence for correlation, or "granger causality" between from on
time series unto the other.

"""


####################LOAD DATA###################
sessionsPerMonth = pd.read_csv("sessionsPerMonth.csv", index_col=0).dropna(axis=1)
encountersPerMonth = pd.read_csv("encountersPerMonth.csv", index_col=0).dropna(axis=1)

sessIndex = sessionsPerMonth.index
encIndex = encountersPerMonth.index

#only take patients which are in both datasets
sessionsPerMonth = sessionsPerMonth.reindex(encIndex)
encountersPerMonth = encountersPerMonth.reindex(sessIndex)

#give same index:
sessionsPerMonth = sessionsPerMonth.reindex(sessIndex).dropna()
encountersPerMonth = encountersPerMonth.reindex(sessIndex).dropna()

#PERFORM ANALYSIS ON SUM OF ALL USER ACTIVITY PER MONTH
sessionsPerMonth = sessionsPerMonth.sum(axis=0)
encountersPerMonth = encountersPerMonth.sum(axis=0)






####################PRE-ANALYSIS####################
#1. Augmented Dickey-Fuller test for unit root to test if stationary (if not,
#take I(2) series, etc.).
#not implemented since is not entirely needed for Granger test.


#2. difference once to I(1) series which are roughly stationary
#(e ~iid N(0,sigma^2))
#not implemented since is not entirely necesary for Granger test.


#3. calculate partial autocorrelation function (PACF) w/ Ljung-Box test to see
#which lags help predict current values.
sessionsPACF = stattools.pacf_ols(sessionsPerMonth, nlags=6)
encountersPACF = stattools.pacf_ols(encountersPerMonth, nlags=6)

#Ljung-Box test
sessionsPACF_LJ = _math.ljungBox(sessionsPACF, len(sessionsPerMonth), 6)
encountersPACF_LJ = _math.ljungBox(encountersPACF, len(encountersPerMonth), 6)
print("sessions PACF Ljung-Box")
print(tabulate(sessionsPACF_LJ, headers=["lag", "R", "Q", "p-val"]))
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

plotxy("sessions", range(7), sessionsPACF)
plotxy("encounters", range(7), encountersPACF)


#4. calculate cross-correlation functions
"""
I could use the following function, however as the lags progress, I find
that the R values diverge from my implementation. I know that my implementation
is not wrong, so I select my implementation for use.

sessionsCCF = stattools.ccf(sessionsPerMonth, encountersPerMont)
Note: first var is dependent, second is independent, which is unclear from 
documentation
"""

sessionsCCF = _math.cc_ols(sessionsPerMonth, encountersPerMonth, 6)
encountersCCF = _math.cc_ols(encountersPerMonth, sessionsPerMonth, 6)

#Ljung-Box test
sessionsCCF_LJ = _math.ljungBox(sessionsCCF, len(sessionsPerMonth), 6)
encountersCCF_LJ = _math.ljungBox(encountersCCF, len(encountersPerMonth), 6)
print("sessions CCF Ljung-Box")
print(tabulate(sessionsCCF_LJ, headers=["lag", "R", "Q", "p-val"]))
print("encounters CCF Ljung-Box")
print(tabulate(encountersCCF_LJ, headers=["lag", "R", "Q", "p-val"]))






####################ANALYSIS####################
#1. select appriopriate number of lags
ARaic = ar_model.AR(sessionsPerMonth.tolist()).fit(maxlag=20, ic="aic")
ARbic = ar_model.AR(sessionsPerMonth.tolist()).fit(maxlag=20, ic="bic")
#select the fewer number of parameters between both criteria.
numLagsSess = len(ARaic.params) if len(ARaic.params) < len(ARbic.params) else len(ARbic.params)

print("Optimal number of lags for session data is "+str(numLagsSess))

ARaic = ar_model.AR(encountersPerMonth.tolist()).fit(maxlag=20, ic="aic")
ARbic = ar_model.AR(encountersPerMonth.tolist()).fit(maxlag=20, ic="bic")
#select the fewer number of parameters between both criteria.
numLagsEnc = len(ARaic.params) if len(ARaic.params) < len(ARbic.params) else len(ARbic.params)

print("Optimal number of lags for encounter data is "+str(numLagsEnc))


#2. now that I know the optimal number of parameters, I can call the
#granger causality function of statsmodels.
data = pd.concat([encountersPerMonth, sessionsPerMonth], axis=1)
print("\nGranger causality results of sessions onto encounters")
results = stattools.grangercausalitytests(data, maxlag=numLagsEnc)

data = pd.concat([sessionsPerMonth, encountersPerMonth], axis=1)
print("\nGranger causality results of encounters onto sessions")
results = stattools.grangercausalitytests(data, maxlag=numLagsSess)

#ideally, I would implement this myself, however statsmodels is buggy and
#does not deal with exogenous variables well, meaning I would have to
#implement an AR fitting algorithm, which is non-ideal.
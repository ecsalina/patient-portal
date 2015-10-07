import scipy
import numpy as np
import pandas
from statsmodels.tsa.stattools import acf, adfuller



def cc_ols(tsY, tsX, maxLag):
	"""
	Calculates and returns the correlation between the X time series
	and Y time series, up to and including max lag.

	Returns list of R's, in pairs of (lag, R).
	The lag is performed on the tsX series, to determine if lagged values
	of the tsX series have effect upon the tsY series.
	"""
	coeffs = []

	tsY = tsY.tolist()
	tsX = tsX.tolist()

	#lag=0 (i.e. R=1)
	Rcoeff = scipy.stats.stats.pearsonr(tsX, tsY)[0]
	coeffs.append(Rcoeff)

	#other lags
	for lag in range(1, maxLag+1):
		ts = tsY[lag:]
		lagts = tsX[:-lag]
		Rcoeff = scipy.stats.stats.pearsonr(lagts, ts)[0]
		coeffs.append(Rcoeff)
	return coeffs





def ljungBox(ACF, n, maxLag):
	"""
	Calculates the Ljung-Box test based on a given ACF list, the length of the
	timeseries, and the maximum lag desired to be computed.

	A high Q indicates that at the given lag, the independent variable is
	good at predicting the dependent variable. A low Q is the opposite
	(little help in prediction).
	"""
	coeffs = [[i, val] for i,val in enumerate(ACF)]

	for lag in range(0, maxLag+1):
		frac = 0.0
		for k in range(1, lag+1):
			frac += (coeffs[k][1]**2 / (n-k))
		Q = frac*n*(n+2)
		pval =  1.0 if Q == 0 else scipy.stats.chi2.sf(Q, lag)
		coeffs[lag].append(Q)
		coeffs[lag].append(pval)

	return coeffs

def ljungBox2(x, maxlag):
	lags = np.asarray(range(1, maxlag+1))
	x = x.tolist()
	n = len(x)
	acfx = acf(x, nlags=maxlag) # normalize by nobs not (nobs-nlags)
	acf2norm = acfx[1:maxlag+1]**2 / (n - np.arange(1,maxlag+1))

	qljungbox = n * (n+2) * np.cumsum(acf2norm)[lags-1]
	pval = scipy.stats.chi2.sf(qljungbox, lags)
	return qljungbox, pval
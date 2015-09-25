Eric Salina
9/24/15
WPI

I carry out a Granger causality test to determine whether the sessions
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
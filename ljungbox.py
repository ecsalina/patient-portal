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
import statsmodels
import _math

"""
Eric Salina
"""

####################CONSTANTS###################
MAX_LAG = 30


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



####################LJUNG-BOX####################
clicksPACF = stattools.pacf_ols(clicksPerDay, nlags=MAX_LAG)
encountersPACF = stattools.pacf_ols(encountersPerDay, nlags=MAX_LAG)

#my implementation
results = _math.ljungBox(encountersPACF, len(encountersPerDay), MAX_LAG)
lag, R, Q, p = zip(*results)
print np.asarray(Q[1:])
print np.asarray(p[1:])

#statsmodels implementation
results = diagnostic.acorr_ljungbox(encountersPerDay, lags=MAX_LAG)
print results

#my copy of the statsmodels implementation
results = _math.ljungBox2(encountersPerDay, maxlag=MAX_LAG)
print results


#they are not the same... I wonder why.
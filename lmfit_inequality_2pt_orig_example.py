"""
Fit Using Inequality Constraint
===============================

NB. bjs, this is the original example on which many of the others are based.

Sometimes specifying boundaries using ``min`` and ``max`` are not sufficient,
and more complicated (inequality) constraints are needed. In the example below
the center of the Lorentzian peak is constrained to be between 0-5 away from
the center of the Gaussian peak.

See also: https://lmfit.github.io/lmfit-py/constraints.html#using-inequality-constraints

This example from:
https://cars9.uchicago.edu/software/python/lmfit/examples/example_fit_with_inequality.html

"""
import matplotlib.pyplot as plt
import numpy as np
import copy

from lmfit import Minimizer, Parameters, report_fit
from lmfit.lineshapes import gaussian, lorentzian


def residual(pars, x, data):
    model = (gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g']) +
             lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l']))
    return model - data

def residual2(pars, x, data):
    model = (gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g']) +
             lorentzian(x, pars['amp_l'], pars['cen_g']+pars['peak_split'], pars['wid_l']))
    return model - data

###############################################################################
# Generate the simulated data using a Gaussian and Lorentzian line shape:
np.random.seed(0)
x = np.linspace(0, 20.0, 601)

data = (gaussian(x, 21, 6.1, 1.2) + lorentzian(x, 10, 9.6, 1.3) +
        np.random.normal(scale=0.5, size=x.size))

###############################################################################
# Create the fitting parameters and set an inequality constraint for ``cen_l``.
# First, we add a new fitting  parameter ``peak_split``, which can take values
# between 0 and 5. Afterwards, we constrain the value for ``cen_l`` using the
# expression to be ``'peak_split+cen_g'``:
pfit = Parameters()
pfit.add(name='amp_g', value=10)
pfit.add(name='amp_l', value=10)
pfit.add(name='cen_g', value=5)
pfit.add(name='peak_split', value=2.5, min=0, max=5, vary=True)
pfit.add(name='cen_l', expr='peak_split+cen_g')
pfit.add(name='wid_g', value=1)
pfit.add(name='wid_l', expr='wid_g')

pfit2 = copy.deepcopy(pfit)
pfit3 = copy.deepcopy(pfit)
pfit4 = copy.deepcopy(pfit)

min1 = Minimizer(residual, pfit, fcn_args=(x, data))
out1 = min1.leastsq()
best_fit1 = data + out1.residual

min2 = Minimizer(residual2, pfit2, fcn_args=(x, data))
out2 = min2.leastsq()
best_fit2 = data + out2.residual

min3 = Minimizer(residual, pfit3, fcn_args=(x, data))
out3 = min3.least_squares()
best_fit3 = data + out3.residual

min4 = Minimizer(residual2, pfit4, fcn_args=(x, data))
out4 = min4.least_squares()
best_fit4 = data + out4.residual

###############################################################################
# Performing a fit, here using the ``leastsq`` algorithm, gives the following
# fitting results:
report_fit(out1.params)
report_fit(out2.params)
report_fit(out3.params)
report_fit(out4.params)

###############################################################################
# and figure:
plt.plot(x, data, 'bo')
plt.plot(x, best_fit1, 'r--', label='best fit')
plt.legend(loc='best')
plt.show()

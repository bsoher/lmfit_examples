"""
LMFIT Using Inequality Constraint and Jacobian - Indiv Gaussian lshape
-------------------------------------------------------------------------------

Based on: https://cars9.uchicago.edu/software/python/lmfit/examples/example_fit_with_inequality.html

This example was to demonstrate that the least_squares.py default Jacobian 
estimate method, '2-point', does not give the same results as a closed form
calculated Jacobian due to the inequality constraint imposed, and how the
Minimizer.__residual() call sets/updates the free/constrained parameters before
sending each 2-point call off to the evaluating function.

This model uses: individual gaussian decay for each peak

"""
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Minimizer, Parameters, report_fit

def sum_gaussians(pars, t):
    ''' model is a sum of gaussians converted from the time to frequency domain'''
    if not any('delta' in item for item in pars.keys()):
        model  = pars['amp1'] * np.exp(-1j * pars['cen1']*2*np.pi*t) * np.exp(-(t/pars['wid1'])**2)
        model += pars['amp2'] * np.exp(-1j * pars['cen2']*2*np.pi*t) * np.exp(-(t/pars['wid2'])**2)
        model += pars['amp3'] * np.exp(-1j * pars['cen3']*2*np.pi*t) * np.exp(-(t/pars['wid3'])**2)
    else:
        cen2 = pars['cen1']+pars['delta_cen2']
        model  = pars['amp1'] * np.exp(-1j * pars['cen1'] * 2 * np.pi * t) * np.exp(-(t / pars['wid1']) ** 2)
        model += pars['amp2'] * np.exp(-1j *       cen2   * 2 * np.pi * t) * np.exp(-(t / pars['wid2']) ** 2)
        model += pars['amp3'] * np.exp(-1j * pars['cen3'] * 2 * np.pi * t) * np.exp(-(t / pars['wid3']) ** 2)
    model = np.fft.fft(model).real
    return model

def dfunc(pars, *args, **kwargs):
    pder = []
    # pre-calculate recurring functions
    gauss1 = np.exp(-1j* pars[3]         *2*np.pi*t) * np.exp(-(t/pars[6])**2)
    gauss2 = np.exp(-1j*(pars[3]+pars[4])*2*np.pi*t) * np.exp(-(t/pars[7])**2)
    gauss3 = np.exp(-1j* pars[5]         *2*np.pi*t) * np.exp(-(t/pars[8])**2)

    # calculate the 9 partial derivatives

    pder.append(gauss1)
    pder.append(gauss2)
    pder.append(gauss3)
    pder.append((-1j*2*np.pi*t*pars[0] * gauss1)+(-1j*2*np.pi*t*pars[1] * gauss2))
    pder.append(-1j*2*np.pi*t*pars[1] * gauss2)
    pder.append(-1j*2*np.pi*t*pars[2] * gauss3)
    pder.append((2.0 * (t**2) / (pars[6]**3)) * pars[0] * gauss1)
    pder.append((2.0 * (t**2) / (pars[7]**3)) * pars[1] * gauss2)
    pder.append((2.0 * (t**2) / (pars[8]**3)) * pars[2] * gauss3)

    # convert to frequency domain
    pder = [np.fft.fft(item).real for item in pder]
    pder = np.array(pder)
    return pder.T

def func(pars, t, data):
    model = sum_gaussians(pars, t)
    return model - data


if __name__ == '__main__':

    sw = 256.0
    npts = 512
    np.random.seed(0)
    t = np.arange(npts) / sw

    ###########################################################################
    # Create the fitting parameters and set an inequality constraint
    # - pars is the ground truth parameter values
    # - pfit is the model parameters initial values

    pars = Parameters()
    pars.add_many(('amp1',   10, True, None, None, None, None),
                  ('amp2',    2, True, None, None, None, None),
                  ('amp3',   10, True, None, None, None, None),
                  ('cen1',  128, True, None, None, None, None),
                  ('cen2',  143, True, None, None, None, None),
                  ('cen3',   64, True, None, None, None, None),
                  ('wid1', 0.04, True, None, None, None, None),
                  ('wid2', 0.04, True, None, None, None, None),
                  ('wid3', 0.04, True, None, None, None, None))

    pfit = Parameters()
    pfit.add_many(('amp1',    8, True, None, None, None, None),
                  ('amp2',    1, True, None, None, None, None),
                  ('amp3',   12, True, None, None, None, None),
                  ('cen1',  128, True, 0.0, 255.0, None, None),
                  ('delta_cen2',   13, True, 5.0,  20.0, None, None),
                  ('cen2', None, False, None, None, 'cen1+delta_cen2', None),
                  ('cen3',   64, True, 0.0, 255.0, None, None),
                  ('wid1', 0.03, True, 0.01, 1.00, None, None),
                  ('wid2', 0.03, True, 0.01, 1.00, None, None),
                  ('wid3', 0.03, True, 0.01, 1.00, None, None))


    ###########################################################################
    # Generate the simulated data using summed Gaussian lines:

    data0 = sum_gaussians(pars, t).real                         # noiseless data
    data  = data0 + np.random.normal(scale=5.0, size=t.size)    # data + noise

    min1 = Minimizer(func, pfit, fcn_args=(t, data))

    ###########################################################################
    # Test the closed form Jacobian vs 2-point estimation method

    fvar_names = [pfit[key].name for key in list(pfit.keys()) if pfit[key].expr is None]
    fvar_vals  = np.array([pfit[key].value for key in fvar_names])
    all_vals   = np.array([item[1] for item in list(pfit.valuesdict().items())])

    f0 = func(pfit, t, data)        # function evaluation for initial values

    pder = dfunc(fvar_vals, t)      # pder (closed form) for initial values
    pder = pder.T

    # Code below is modeled from minimizer.py least_squares.py and _numdiff.py
    #  ... it recreates the partial derivatives estimated from the initial
    #  values (same as passed into 'dfunc()' above) using the 2-point
    #  estimation method, which is default option in least_squares.py
    #
    # There is only one partial derivative that is significantly different
    # between the closed form (pder) and 2-point estimate method (diff). It is
    # the one calculated for 'cen1' which is part of the inequality set up for
    # the 'cen2' parameter. This is demonstrated in the plot below.

    res = min1.prepare_fit(pfit)
    diff = np.zeros([len(fvar_vals), npts], dtype=np.float64)
    h = 1.49e-08 * np.maximum(1.0, np.abs(fvar_vals))
    h_vecs = np.diag(h)
    for i in range(h.size):             # from around line 551 _numdiff.py
        x = fvar_vals + h_vecs[i]
        dx = x[i] - fvar_vals[i]
        df1 = min1._Minimizer__residual(x, apply_bounds_transformation=False)
        df = df1.real
        diff[i] = (df - f0) / dx

    fig, axs = plt.subplots(3,2)
    axs[0, 0].set_title("'cen1' pder function'")
    axs[0, 0].plot(pder[3,:], 'b')
    axs[1, 0].set_title("'cen1' 2-pt estimate'")
    axs[1, 0].plot(diff[3,:], 'g')
    axs[2, 0].set_title("Difference")
    axs[2, 0].plot(pder[3,:] - diff[3,:], 'purple')
    axs[0, 1].set_title("'cen1'+'cen2' pder functions'")
    axs[0, 1].plot(pder[3,:]+pder[4,:], 'b')
    axs[1, 1].set_title("'cen1' 2-pt estimate'")
    axs[1, 1].plot(diff[3,:], 'g')
    axs[2, 1].set_title("Diff, note 1e-7 scale")
    axs[2, 1].plot(pder[3,:]+pder[4,:] - diff[3,:], 'purple')
    plt.show()

    ###########################################################################
    # Here we optimize the model without/with a Jacobian function
    # - results are similar but not exactly the same

    min1 = Minimizer(func, pfit, fcn_args=(t, data))
    out1 = min1.least_squares()
    best_fit1 = data + out1.residual
    report_fit(out1.params)

    min2 = Minimizer(func, pfit, fcn_args=(t, data))
    out2 = min2.least_squares(jac=dfunc)
    best_fit2 = data + out2.residual
    report_fit(out2.params)

    fig, axs = plt.subplots(2,2)
    axs[0,0].set_title("fit vs data, 2-point")
    axs[0,0].plot(data,      'b-')
    axs[0,0].plot(best_fit1, 'r-')
    axs[1,0].set_title("fit vs actual and difference")
    axs[1,0].plot(data0,     'g')
    axs[1,0].plot(best_fit1, 'r-')
    axs[1,0].plot(data0 - best_fit1, 'purple')

    axs[0,1].set_title("fit vs data, jacobian")
    axs[0,1].plot(data,      'b-')
    axs[0,1].plot(best_fit2, 'r-')
    axs[1,1].set_title("fit vs actual and difference")
    axs[1,1].plot(data0,     'g')
    axs[1,1].plot(best_fit2, 'r-')
    axs[1,1].plot(data0 - best_fit2, 'purple')

    plt.show()

    bob = 10







# def pders(pars, t):
#
#     pder = []
#
#     gauss1 = np.exp(-1j * pars['cen1'] * 2 * np.pi * t) * np.exp(-1 * (t/pars['wid1'])**2)
#     gauss2 = np.exp(-1j * pars['cen2'] * 2 * np.pi * t) * np.exp(-1 * (t/pars['wid2'])**2)
#     gauss3 = np.exp(-1j * pars['cen3'] * 2 * np.pi * t) * np.exp(-1 * (t/pars['wid3'])**2)
#
#     pder.append(gauss1)
#     pder.append(gauss2)
#     pder.append(gauss3)
#     pder.append(t * 1j * pars['amp1'] * gauss1)
#     pder.append(t * 1j * pars['amp2'] * gauss2)
#     pder.append(t * 1j * pars['amp3'] * gauss3)
#     pder.append((2.0 * (t**2) / (pars['wid1']**3)) * pars['amp1'] * gauss1)
#     pder.append((2.0 * (t**2) / (pars['wid2']**3)) * pars['amp2'] * gauss2)
#     pder.append((2.0 * (t**2) / (pars['wid3']**3)) * pars['amp3'] * gauss3)
#
#     pder = [np.fft.fft(item).real for item in pder]
#
#     return pder


    # for i in range(len(fvar_vals)):
    #     fig, axs = plt.subplots(3)
    #     axs[0].plot(pder[i,:], 'b')
    #     axs[1].plot(diff[i,:], 'g')
    #     axs[2].plot(pder[i,:] - diff[i,:], 'purple')
    #     plt.show()
    #     bob = 10

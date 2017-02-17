import flavio
import flavio.statistics.fits
import numpy as np

# Wilson coefficient function
def wc_fct_smeft(ReCS, ImCS, ReCSp, ImCSp):
    return {
            'CS_bsmumu': ReCS+1j*ImCS,
            'CP_bsmumu': -(ReCS+1j*ImCS),
            'CSp_bsmumu': ReCSp+1j*ImCSp,
            'CPp_bsmumu': ReCSp+1j*ImCSp,
           }

# Bayesian fit

# priors to find the starting value
wc_start = flavio.classes.WilsonCoefficientPriors()
for coeff in ['ReCS', 'ImCS', 'ReCSp', 'ImCSp']:
    wc_start.add_constraint([coeff], flavio.statistics.probability.UniformDistribution(0, 0.01))

# Bayesian fit instance
bayesian_fit = flavio.statistics.fits.BayesianFit(
                name = "CS-CSp SMEFT Bayesian fit",
                nuisance_parameters = [ 'DeltaGamma/Gamma_Bs', 'Vcb', 'Vub',
                             'f_Bs', 'f_B0', 'gamma', 'tau_Bs', 'tau_B0', ],
                observables = [ 'BR(Bs->mumu)', 'BR(Bd->mumu)', ],
                fit_wc_function = wc_fct_smeft,
                start_wc_starts = wc_start,
                input_scale = 4.8,
                include_measurements = ['LHCb Bs->mumu 2017', 'CMS Bs->mumu 2013'],
            )

# Fast fit: present situation
fast_fit = flavio.statistics.fits.FastFit(
                name = "CS-CSp SMEFT fast fit",
                nuisance_parameters = 'all',
                observables = [ 'BR(Bs->mumu)', 'BR(Bd->mumu)', ],
                fit_wc_function = wc_fct_smeft,
                fit_wc_starts = wc_start,
                input_scale = 4.8,
                include_measurements = ['LHCb Bs->mumu 2017', 'CMS Bs->mumu 2013'],
            )

# Future fast fits

# load the run 4 and run 5 projected measurements
flavio.measurements.load('meas_future.yml')

# set the uncertainties for the future theory parameters
par_future = flavio.default_parameters.copy()
par_future.set_constraint('f_Bs', '0.2284(10)')
par_future.set_constraint('Vcb', '4.221(30)e-2')

fast_fit_future = {}
for future in ['Run 4 SM', 'Run 4 NP', 'Run 5 SM', 'Run 5 NP']:
    fast_fit_future[future] = flavio.statistics.fits.FastFit(
                name = "CS-CSp SMEFT fast fit " + future,
                par_obj = par_future,
                nuisance_parameters = 'all',
                observables = [ 'BR(Bs->mumu)', 'ADeltaGamma(Bs->mumu)', ],
                fit_wc_function = wc_fct_smeft,
                fit_wc_starts = wc_start,
                input_scale = 4.8,
                include_measurements = ['Bs->mumu ' + future],
            )

# auxiliary function to compute fit predictions e.g. for FH, S6c from Bayesian fit
def get_fit_prediction(fit, obs, x):
    par = fit.get_par_dict(x)
    wc_obj = fit.get_wc_obj(x)
    if isinstance(obs, str):
        obs_obj = flavio.classes.Observable.get_instance(obs)
        return obs_obj.prediction_par(par, wc_obj, *args, **kwargs)
    else:
        obs_obj = flavio.classes.Observable.get_instance(obs[0])
        return obs_obj.prediction_par(par, wc_obj, *obs[1:])

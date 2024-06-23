import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor
from doubleml.double_ml import DoubleML
from doubleml.double_ml_data import DoubleMLData
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.double_ml_blp import DoubleMLBLP
from doubleml._utils import _dml_cv_predict, _dml_tune
from doubleml._utils_checks import _check_score, _check_finite_predictions, _check_is_propensity
from scipy.linalg import toeplitz
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import matplotlib.pyplot as plt
import sys

class KernelTransparencyLearning(LinearScoreMixin, DoubleML):
    def __init__(self,
                 obj_dml_data,
                 ml_l,
                 ml_m,
                 ml_g,
                 ml_h,
                 n_folds=5,
                 n_rep=1,
                 score='IV-type',
                 dml_procedure='dml1',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)

        self._check_data(self._dml_data)
        self._learner = {'ml_l': ml_l, 'ml_m': ml_m, 'ml_g': ml_g, 'ml_h': ml_h}
        self._predict_method = {'ml_l': 'predict', 'ml_m': 'predict', 'ml_g': 'predict', 'ml_h': 'predict'}
        self._initialize_ml_nuisance_params()
        self._sensitivity_implemented = True
        self._external_predictions_implemented = True

    def _initialize_ml_nuisance_params(self):
        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols}
                        for learner in self._learner}

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.')
        if obj_dml_data.z_cols is not None:
            raise ValueError('Incompatible data. ' +
                             ' and '.join(obj_dml_data.z_cols) +
                             ' have been set as instrumental variable(s). '
                             'To fit a partially linear IV regression model use DoubleMLPLIV instead of DoubleMLPLR.')
        return

    def _nuisance_est(self, smpls, n_jobs_cv, external_predictions, return_models=False):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        m_external = external_predictions['ml_m'] is not None
        l_external = external_predictions['ml_l'] is not None
        g_external = external_predictions['ml_g'] is not None
        h_external = external_predictions['ml_h'] is not None
        if l_external:
            l_hat = {'preds': external_predictions['ml_l'],
                     'targets': None,
                     'models': None}
        elif self._score == "IV-type" and g_external:
            l_hat = {'preds': None,
                     'targets': None,
                     'models': None}
        else:
            l_hat = _dml_cv_predict(self._learner['ml_l'], x, y, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_l'), method=self._predict_method['ml_l'],
                                    return_models=return_models)
            _check_finite_predictions(l_hat['preds'], self._learner['ml_l'], 'ml_l', smpls)

        if m_external:
            m_hat = {'preds': external_predictions['ml_m'],
                     'targets': None,
                     'models': None}
        else:
            m_hat = _dml_cv_predict(self._learner['ml_m'], x, d, smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_m'), method=self._predict_method['ml_m'],
                                    return_models=return_models)
            _check_finite_predictions(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls)
        if self._check_learner(self._learner['ml_m'], 'ml_m', regressor=True, classifier=True):
            _check_is_propensity(m_hat['preds'], self._learner['ml_m'], 'ml_m', smpls, eps=1e-12)

        if h_external:
            h_hat = {'preds': external_predictions['ml_h'],
                     'targets': None,
                     'models': None}
        else:
            h_hat = _dml_cv_predict(self._learner['ml_h'], d.reshape(len(d), 1), d-m_hat['preds'], smpls=smpls, n_jobs=n_jobs_cv,
                                    est_params=self._get_params('ml_h'), method=self._predict_method['ml_h'],
                                    return_models=return_models)
            _check_finite_predictions(h_hat['preds'], self._learner['ml_h'], 'ml_h', smpls)
        if self._dml_data.binary_treats[self._dml_data.d_cols[self._i_treat]]:
            binary_preds = (type_of_target(m_hat['preds']) == 'binary')
            zero_one_preds = np.all((np.power(m_hat['preds'], 2) - m_hat['preds']) == 0)
            if binary_preds & zero_one_preds:
                raise ValueError(f'For the binary treatment variable {self._dml_data.d_cols[self._i_treat]}, '
                                 f'predictions obtained with the ml_m learner {str(self._learner["ml_m"])} are also '
                                 'observed to be binary with values 0 and 1. Make sure that for classifiers '
                                 'probabilities and not labels are predicted.')
        g_hat = {'preds': None, 'targets': None, 'models': None}
        if 'ml_g' in self._learner:
            if g_external:
                g_hat = {'preds': external_predictions['ml_g'],
                         'targets': None,
                         'models': None}
            else:
                # get an initial estimate for theta using the partialling out score
                psi_a = -np.multiply(d - m_hat['preds'], d - m_hat['preds'])
                psi_b = np.multiply(d - m_hat['preds'], y - l_hat['preds'])
                theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
                g_hat = _dml_cv_predict(self._learner['ml_g'], x, y - theta_initial*d, smpls=smpls, n_jobs=n_jobs_cv,
                                        est_params=self._get_params('ml_g'), method=self._predict_method['ml_g'],
                                        return_models=return_models)
                _check_finite_predictions(g_hat['preds'], self._learner['ml_g'], 'ml_g', smpls)

        psi_a, psi_b = self._score_elements(y, d, g_hat['preds'], h_hat['preds'])
        psi_elements = {'psi_a': psi_a,
                        'psi_b': psi_b}
        preds = {'predictions': {'ml_l': l_hat['preds'],
                                 'ml_m': m_hat['preds'],
                                 'ml_g': g_hat['preds'],
                                 'ml_h': h_hat['preds']},
                 'targets': {'ml_l': l_hat['targets'],
                             'ml_m': m_hat['targets'],
                             'ml_g': g_hat['targets'],
                             'ml_h': h_hat['targets']},
                 'models': {'ml_l': l_hat['models'],
                            'ml_m': m_hat['models'],
                            'ml_g': g_hat['models'],
                            'ml_h': h_hat['models']}}
        return psi_elements, preds

    def _score_elements(self, y, d, g_hat, h_hat):
        psi_a = - np.multiply(h_hat, d)
        psi_b = np.multiply(h_hat, y - g_hat)
        return psi_a, psi_b

    def _sensitivity_element_est(self, preds):
        y = self._dml_data.y
        d = self._dml_data.d
        m_hat = preds['predictions']['ml_m']
        theta = self.all_coef[self._i_treat, self._i_rep]
        assert self.score == 'IV-type'
        g_hat = preds['predictions']['ml_g']
        sigma2_score_element = np.square(y - g_hat - np.multiply(theta, d))
        sigma2 = np.mean(sigma2_score_element)
        psi_sigma2 = sigma2_score_element - sigma2
        nu2 = np.divide(1.0, np.mean(np.square(d - m_hat)))
        psi_nu2 = nu2 - np.multiply(np.square(d-m_hat), np.square(nu2))
        element_dict = {'sigma2': sigma2,
                        'nu2': nu2,
                        'psi_sigma2': psi_sigma2,
                        'psi_nu2': psi_nu2}
        return element_dict

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv,
                         search_mode, n_iter_randomized_search):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y,
                         force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d,
                         force_all_finite=False)
        if scoring_methods is None:
            scoring_methods = {'ml_l': None, 'ml_m': None, 'ml_g': None, 'ml_h': None}
        l_hat = np.full_like(y, np.nan)
        m_hat = np.full_like(d, np.nan)
        h_hat = np.full_like(d, np.nan)
        train_inds = [train_index for (train_index, _) in smpls]
        l_tune_res = _dml_tune(y, x, train_inds,
                               self._learner['ml_l'], param_grids['ml_l'], scoring_methods['ml_l'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        m_tune_res = _dml_tune(d, x, train_inds,
                               self._learner['ml_m'], param_grids['ml_m'], scoring_methods['ml_m'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        h_tune_res = _dml_tune(d-m_hat['preds'], d.reshape(len(d), 1), train_inds,
                              self._learner['ml_h'], param_grids['ml_h'], scoring_methods['ml_h'],
                              n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)

        l_best_params = [xx.best_params_ for xx in l_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        h_best_params = [xx.best_params_ for xx in h_tune_res]
        for idx, (train_index, _) in enumerate(smpls):
            l_hat[train_index] = l_tune_res[idx].predict(x[train_index, :])
            m_hat[train_index] = m_tune_res[idx].predict(x[train_index, :])
            h_hat[train_index] = h_tune_res[idx].predict(x[train_index, :])
        psi_a = -np.multiply(h_hat, d - m_hat)
        psi_b = np.multiply(h_hat, y - l_hat)
        theta_initial = -np.nanmean(psi_b) / np.nanmean(psi_a)
        g_tune_res = _dml_tune(y - theta_initial * d, x, train_inds,
                               self._learner['ml_g'], param_grids['ml_g'], scoring_methods['ml_g'],
                               n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search)
        g_best_params = [xx.best_params_ for xx in g_tune_res]
        params = {'ml_l': l_best_params,
                  'ml_m': m_best_params,
                  'ml_g': g_best_params,
                  'ml_h': h_best_params}
        tune_res = {'l_tune': l_tune_res,
                    'm_tune': m_tune_res,
                    'g_tune': g_tune_res,
                    'h_tune': h_tune_res}
        res = {'params': params,
               'tune_res': tune_res}
        return res

def algorithm(z, y, x, i):
    np.random.seed(1000 + i)
    kerneltransparencydata = DoubleMLData.from_arrays(z, y, x)
    learner = RandomForestRegressor(n_estimators=500, max_features=20, max_depth=5, min_samples_leaf=2)
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_g = clone(learner)
    ml_h = clone(learner)
    dml_plr_obj = KernelTransparencyLearning(kerneltransparencydata, ml_l, ml_m, ml_g, ml_h, score='IV-type',
                                             dml_procedure='dml1')
    dml_plr_obj.fit()
    print(dml_plr_obj)
    return dml_plr_obj.coef

data = pd.read_csv("2020-04-03_ADNI.csv")
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
data = data.drop([486, 517, 672]).values
z_index = np.array([5, 6, 11, 15]+list(range(19, 89)), dtype=int)
np.random.seed(1000)
x = data[:, 89:157]
y = data[:, 16]
z = data[:, z_index]
beta = []
for i in range(10):
    beta.append(algorithm(z, y, x, i))
    np.savetxt('result2/output'+str(i + 1)+'.txt', beta[i], fmt='%0.4f')
print(np.mean(beta, axis=0))


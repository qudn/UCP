# -*- coding: utf-8 -*-
"""
  precalibration: source probability calibration; beta01 uses beta ridge=.01.
  alpha_blend: mixing coefficient between estimated target and source priors.
  h_min: entropy floor in natural-log units.
  lambda_entropy: weight of the entropy penalty.
  tau, gamma: shrinkage of slice temperatures toward the global temperature.
  shift_strength: explicit prior-shift correction strength; zero disables it.
  h_slope: fixed at zero in these profiles, giving a constant entropy floor.
"""
import numpy as np

# Fixed overall-TACE profiles; the generic Ours algorithm has no built-in profiles.
OURS_TACE_PARAMETERS = {'acs_income_catboost': {'config': {'alpha_blend': 0.07,
                                    'gamma': 0.5,
                                    'h_min': 0.48520302639196167,
                                    'h_slope': 0.0,
                                    'lambda_entropy': 1.0,
                                    'shift_strength': 1.0,
                                    'tau': 3.3333333333333335},
                         'precalibration': 'none'},
 'acs_income_lightgbm': {'config': {'alpha_blend': 0.07,
                                    'gamma': 0.5,
                                    'h_min': 0.4921344981975611,
                                    'h_slope': 0.0,
                                    'lambda_entropy': 0.2,
                                    'shift_strength': 1.0,
                                    'tau': 10.0},
                         'precalibration': 'none'},
 'acs_income_logistic': {'config': {'alpha_blend': 0.0,
                                    'gamma': 0.5,
                                    'h_min': 0.5337233290311579,
                                    'h_slope': 0.0,
                                    'lambda_entropy': 1.0,
                                    'shift_strength': 1.0,
                                    'tau': 100.0},
                         'precalibration': 'affine'},
 'acs_income_xgboost': {'config': {'alpha_blend': 0.13,
                                   'gamma': 0.5,
                                   'h_min': 0.4782715545863622,
                                   'h_slope': 0.0,
                                   'lambda_entropy': 1.0,
                                   'shift_strength': 1.0,
                                   'tau': 10.0},
                        'precalibration': 'none'},
 'credit_card_fraud_catboost': {'config': {'alpha_blend': 0.23,
                                           'gamma': 0.5,
                                           'h_min': 0.2564644568071798,
                                           'h_slope': 0.0,
                                           'lambda_entropy': 1.0,
                                           'shift_strength': 0.1,
                                           'tau': 300.0},
                                'precalibration': 'affine'},
 'credit_card_fraud_lightgbm': {'config': {'alpha_blend': 0.1,
                                           'gamma': 0.5,
                                           'h_min': 0.2841903440295776,
                                           'h_slope': 0.0,
                                           'lambda_entropy': 0.2,
                                           'shift_strength': 0.1,
                                           'tau': 0.3},
                                'precalibration': 'none'},
 'credit_card_fraud_logistic': {'config': {'alpha_blend': 0.19829074200242758,
                                           'gamma': 0.18022278510034084,
                                           'h_min': 0.24369110150370174,
                                           'h_slope': 0.0,
                                           'lambda_entropy': 0.2,
                                           'shift_strength': 0.11359269432723523,
                                           'tau': 1233.5128531084551},
                                'precalibration': 'temperature'},
 'credit_card_fraud_xgboost': {'config': {'alpha_blend': 0.25,
                                          'gamma': 0.5,
                                          'h_min': 0.23567004139038142,
                                          'h_slope': 0.0,
                                          'lambda_entropy': 1.0,
                                          'shift_strength': 0.0,
                                          'tau': 300.0},
                               'precalibration': 'affine'},
 'telco_churn_catboost': {'config': {'alpha_blend': 0.11,
                                     'gamma': 0.5,
                                     'h_min': 0.45054566736396445,
                                     'h_slope': 0.0,
                                     'lambda_entropy': 1.0,
                                     'shift_strength': 1.0,
                                     'tau': 0.3},
                          'precalibration': 'beta01'},
 'telco_churn_lightgbm': {'config': {'alpha_blend': 0.0,
                                     'gamma': 0.5,
                                     'h_min': 0.4574771391695639,
                                     'h_slope': 0.0,
                                     'lambda_entropy': 1.0,
                                     'shift_strength': 1.0,
                                     'tau': 3.0},
                          'precalibration': 'beta01'},
 'telco_churn_logistic': {'config': {'alpha_blend': 0.02,
                                     'gamma': 0.5,
                                     'h_min': 0.4366827237527655,
                                     'h_slope': 0.0,
                                     'lambda_entropy': 1.0,
                                     'shift_strength': 1.0,
                                     'tau': 300.0},
                          'precalibration': 'none'},
 'telco_churn_xgboost': {'config': {'alpha_blend': 0.0,
                                    'gamma': 0.5,
                                    'h_min': 0.09010913347279285,
                                    'h_slope': 0.0,
                                    'lambda_entropy': 0.2,
                                    'shift_strength': 1.0,
                                    'tau': 90.0},
                         'precalibration': 'beta01'}}

def supported_tace(p,y,support=20):
    total=0.
    for prob,outcome in [(1-p,1-y),(p,y)]:
        ids=np.flatnonzero(prob>.01); ids=ids[np.argsort(prob[ids],kind='stable')]; n=len(ids)
        if not n:
            continue
        bins=min(15,max(1,n//support)); sizes=np.full(bins,n//bins); sizes[:n%bins]+=1
        starts=np.r_[0,np.cumsum(sizes)[:-1]]
        total+=np.abs(np.add.reduceat(prob[ids]-outcome[ids],starts)).sum()/n/2
    return float(total)


def tace(logits, labels):
    """Classwise TACE: threshold .01, maximum15 bins, minimum-support rule20."""
    from scipy.special import expit
    z=np.asarray(logits,dtype=np.float64)
    labels=np.asarray(labels,dtype=np.int64)
    if z.ndim!=2 or z.shape[1]!=2 or len(z)!=len(labels) or not len(z):
        raise ValueError('Expected nonempty [N,2] logits and [N] binary labels')
    if not np.isfinite(z).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Need finite logits and binary labels')
    return supported_tace(expit(z[:,1]-z[:,0]),labels,20)

# -*- coding: utf-8 -*-
"""Binary Ours calibration: source calibration, MLLS, prior shift, entropy and slices.

Experiment profiles are supplied explicitly by the caller (see tace.py).
Prediction requires source labels but never target labels.
Dependencies: numpy, scipy, torch.
"""


"""Paper 2.1--2.2: prior alignment, entropy guard, shrink, re-guard."""
import math
import torch
from torch import nn, optim


class PriorAwareEntropyCalibrator(nn.Module):
    def __init__(self, num_classes, source_prior=None, alpha_blend=0.25,
                 epsilon_gate=0.01, h_min=None, lambda_entropy=1.0,
                 lr_mlls=0.05, lr_temp=0.05, max_iter=200, verbose=False,
                 tau=10.0, gamma=0.5, gate='step', shift_strength=None,
                 min_temp=0.05, max_temp=10.0):
        super().__init__()
        if num_classes < 2 or not 0 <= alpha_blend <= 1:
            raise ValueError('Need K >= 2 and alpha_blend in [0,1]')
        if tau <= 0 or not 0 <= gamma <= 1 or epsilon_gate <= 0:
            raise ValueError('Invalid shrinkage or gate parameters')
        if gate not in ('step', 'linear', 'none'):
            raise ValueError('Unknown gate')
        self.h_min = 0.1 * math.log(num_classes) if h_min is None else float(h_min)
        if not 0 <= self.h_min < math.log(num_classes):
            raise ValueError('h_min must be in [0, log(K))')
        if not 0 < min_temp < max_temp or max_iter < 1 or lambda_entropy < 0:
            raise ValueError('Invalid optimization parameters')
        self.num_classes, self.source_prior = num_classes, source_prior
        self.alpha_blend, self.epsilon_gate = alpha_blend, epsilon_gate
        self.lambda_entropy = lambda_entropy
        self.lr_mlls, self.lr_temp, self.max_iter = lr_mlls, lr_temp, max_iter
        self.verbose, self.tau, self.gamma = verbose, tau, gamma
        self.gate = gate
        # Paper reuses alpha for blending and gate amplitude. Defaults follow
        # that notation; an explicit override separates the interpretations.
        self.shift_strength = alpha_blend if shift_strength is None else shift_strength
        if not 0 <= self.shift_strength <= 1:
            raise ValueError('shift_strength must be in [0,1]')
        self.min_temp, self.max_temp = min_temp, max_temp
        self.diagnostics_ = {}

    def _logits(self, logits):
        logits = torch.as_tensor(logits).detach()
        if not logits.is_floating_point():
            logits = logits.float()
        if logits.ndim != 2 or logits.shape[1] != self.num_classes or len(logits) == 0:
            raise ValueError('Expected nonempty [N,K] logits; binary inputs need two columns')
        if not torch.isfinite(logits).all():
            raise ValueError('Logits must be finite')
        return logits

    def _source_prior(self, labels, logits):
        if self.source_prior is None:
            labels = torch.as_tensor(labels, device=logits.device)
            labels = labels.argmax(1) if labels.ndim == 2 else labels.long()
            prior = torch.bincount(labels, minlength=self.num_classes).to(logits)
        else:
            prior = torch.as_tensor(self.source_prior).to(logits)
        if prior.shape != (self.num_classes,) or not torch.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError('Source prior needs positive support for every class')
        return prior / prior.sum()

    @staticmethod
    def _entropy(logits, temperature):
        log_p = torch.log_softmax(logits / temperature, dim=1)
        return -(log_p.exp() * log_p).sum(1).mean()

    def _estimate_target_prior_mlls(self, source_probs, source_prior):
        with torch.enable_grad():
            probs = source_probs.detach()
            q_s = source_prior.to(probs)
            q_logits = q_s.log().clone().requires_grad_(True)
            optimizer = optim.Adam([q_logits], lr=self.lr_mlls)
            for _ in range(self.max_iter):
                optimizer.zero_grad()
                likelihood = (probs * (q_logits.softmax(0) / q_s)).sum(1)
                loss = -likelihood.clamp_min(torch.finfo(probs.dtype).tiny).log().mean()
                loss.backward()
                optimizer.step()
            return q_logits.softmax(0).detach()

    def _prior_aware_logit_shift(self, logits, target_prior_hat, source_prior):
        q_hat, q_s = target_prior_hat.to(logits), source_prior.to(logits)
        blended = (1 - self.alpha_blend) * q_hat + self.alpha_blend * q_s
        discrepancy = (logits.softmax(1).mean(0) - q_hat).abs().sum().item()
        if self.gate == 'none':
            strength = self.shift_strength
        elif discrepancy <= self.epsilon_gate:
            strength = 0.0
        elif self.gate == 'step':
            strength = self.shift_strength
        else:
            strength = self.shift_strength * min(1.0, discrepancy / self.epsilon_gate)
        self.diagnostics_.update(pre_shift_l1=discrepancy, shift_strength=strength)
        return logits + strength * (blended.log() - q_s.log()), blended

    def _temperature_loss(self, logits, target_prior, temperature):
        alignment = ((logits / temperature).softmax(1).mean(0) - target_prior).abs().sum()
        guard = (self.h_min - self._entropy(logits, temperature)).clamp_min(0)
        return alignment + self.lambda_entropy * guard

    def _optimize_temperature(self, logits, target_prior, init_T=1.0):
        # Scalar bounded search. No unreported shift regularization.
        with torch.no_grad():
            grid = torch.logspace(math.log10(self.min_temp), math.log10(self.max_temp),
                                  65, device=logits.device, dtype=logits.dtype)
            values = torch.stack([self._temperature_loss(logits, target_prior, t) for t in grid])
            index = int(values.argmin())
            best, best_loss = grid[index], values[index]
            lo, hi = grid[max(0, index-1)], grid[min(64, index+1)]
            for _ in range(4):
                grid = torch.linspace(lo.item(), hi.item(), 17, device=logits.device, dtype=logits.dtype)
                values = torch.stack([self._temperature_loss(logits, target_prior, t) for t in grid])
                index = int(values.argmin())
                if values[index] < best_loss:
                    best, best_loss = grid[index], values[index]
                lo, hi = grid[max(0, index-1)], grid[min(16, index+1)]
            return best.detach()

    def _guard_temperature(self, logits, lower=None):
        # Entropy increases with T. Expand the bracket rather than silently
        # violate the hard floor when the global search cap is insufficient.
        with torch.no_grad():
            lo = self.min_temp if lower is None else float(lower)
            if self._entropy(logits, lo).item() >= self.h_min:
                return logits.new_tensor(lo)
            hi = max(self.max_temp, lo * 2)
            for _ in range(60):
                if self._entropy(logits, hi).item() >= self.h_min:
                    break
                hi *= 2
            else:
                raise RuntimeError('Cannot bracket entropy floor')
            for _ in range(60):
                mid = (lo + hi) / 2
                if self._entropy(logits, mid).item() >= self.h_min:
                    hi = mid
                else:
                    lo = mid
            return logits.new_tensor(hi)

    def calibrate_slices(self, logits, target_prior, groups, global_T):
        groups = torch.as_tensor(groups, device=logits.device)
        if groups.ndim != 1 or len(groups) != len(logits):
            raise ValueError('groups must contain one ID per target sample')
        result, diagnostics = logits.clone(), {}
        for group in torch.unique(groups):
            mask = groups == group
            subset, n = logits[mask], int(mask.sum())
            raw = self._guard_temperature(subset)
            weight = n / (n + self.tau * n ** self.gamma)
            shrunk = weight * raw + (1 - weight) * global_T
            final = self._guard_temperature(subset, lower=shrunk.item())
            result[mask] = subset / final
            diagnostics[group.item()] = dict(n=n, weight=weight, raw_temperature=raw.item(),
                shrunk_temperature=shrunk.item(), temperature=final.item(),
                entropy=self._entropy(subset, final).item())
        self.diagnostics_['slices'] = diagnostics
        return result

    def forward(self, source_logits, source_labels, target_logits, target_groups=None):
        logits = self._logits(target_logits)
        prior = self._source_prior(source_labels, logits)
        self.diagnostics_ = {}
        estimated = self._estimate_target_prior_mlls(logits.softmax(1), prior)
        shifted, blended = self._prior_aware_logit_shift(logits, estimated, prior)
        temperature = self._optimize_temperature(shifted, blended)
        self.diagnostics_.update(source_prior=prior, estimated_prior=estimated,
                                 blended_prior=blended, global_temperature=temperature.item())
        if target_groups is not None:
            return self.calibrate_slices(shifted, blended, target_groups, temperature)
        # No post-temperature logit shift: it would invalidate the guard.
        return shifted / temperature



"""Explicit research extensions; the paper implementation remains unchanged.

Source-only temperature/affine pre-calibration, converged binary MLLS, and
optional importance-weighted source temperature selection. No target labels.
"""
import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar


class AdaptivePriorCalibrator(PriorAwareEntropyCalibrator):
    def __init__(self, source_calibration='temperature', temperature_objective='paper', **kwargs):
        super().__init__(**kwargs)
        if self.num_classes != 2:
            raise ValueError('Research extension currently supports binary classification only')
        if source_calibration not in ('none', 'temperature', 'affine'):
            raise ValueError('Unknown source calibration')
        if temperature_objective not in ('paper', 'weighted_source'):
            raise ValueError('Unknown temperature objective')
        self.source_calibration = source_calibration
        self.temperature_objective = temperature_objective

    def fit_source(self, logits, labels):
        logits = self._logits(logits)
        labels = torch.as_tensor(labels, device=logits.device).long()
        if labels.shape != (len(logits),):
            raise ValueError('Expected one integer source label per sample')
        margin = (logits[:, 1]-logits[:, 0]).double().cpu().numpy()
        y = labels.cpu().numpy()
        def objective(parameters):
            z = margin * np.exp(-parameters[0])
            if len(parameters) == 2:
                z = z + parameters[1]
            return np.mean(np.logaddexp(0., z)-y*z)
        if self.source_calibration == 'none':
            parameters = np.array([0.])
        elif self.source_calibration == 'temperature':
            fit = minimize_scalar(lambda t: objective([t]), bounds=(np.log(.1), np.log(20)), method='bounded')
            if not fit.success:
                raise RuntimeError('Source temperature optimization failed')
            parameters = np.array([fit.x])
        else:
            fit = minimize(objective, np.zeros(2), method='L-BFGS-B',
                           bounds=[(np.log(.1), np.log(20)), (-10., 10.)])
            if not fit.success:
                raise RuntimeError('Source affine optimization failed')
            parameters = fit.x
        self.source_temperature_ = float(np.exp(parameters[0]))
        self.source_bias_ = float(parameters[1]) if len(parameters) == 2 else 0.
        self.source_logits_ = self.transform_source(logits)
        self.source_labels_ = labels
        self.fitted_prior_ = self._source_prior(labels, logits)
        return self

    def transform_source(self, logits):
        logits = self._logits(logits)
        output = logits / self.source_temperature_
        output = output.clone()
        output[:, 1] += self.source_bias_
        return output

    def _estimate_target_prior_mlls(self, source_probs, source_prior):
        # Convex one-dimensional likelihood; no unconverged 200-step Adam fit.
        probs = source_probs.detach().double().cpu().numpy()
        prior = source_prior.detach().double().cpu().numpy()
        def loss(q):
            likelihood = probs[:, 0]*(1-q)/prior[0] + probs[:, 1]*q/prior[1]
            return -np.log(np.maximum(likelihood, 1e-300)).mean()
        result = minimize_scalar(loss, bounds=(1e-7, 1-1e-7), method='bounded',
                                 options={'xatol': 1e-9})
        if not result.success:
            raise RuntimeError('MLLS scalar optimization failed')
        candidates = [float(prior[1]), 1e-7, result.x, 1-1e-7]
        q = min(candidates, key=loss)
        if loss(float(prior[1]))-loss(q) < 1e-10:
            q = float(prior[1])
        return source_probs.new_tensor([1-q, q])



"""Existing v16 binary calibration path, full Ours only; no ablation branches."""
import numpy as np
from scipy.special import expit

def entropy(delta,temperature):
    x=np.abs(np.asarray(delta)/temperature)
    return (np.logaddexp(0.,-x)+x*expit(-x)).mean(axis=0)

def guard(delta,h):
    lo=.05
    if entropy(delta,lo)>=h:return lo
    hi=10.
    for _ in range(60):
        if entropy(delta,hi)>=h:break
        hi*=2
    else:raise RuntimeError('Cannot bracket entropy root')
    for _ in range(60):
        mid=(lo+hi)/2
        if entropy(delta,mid)>=h:hi=mid
        else:lo=mid
    return hi

def global_t(delta,q,h,lam):
    def loss(ts):
        x=delta[:,None]/ts[None,:]
        return 2*abs(expit(x).mean(0)-q)+lam*np.maximum(h-entropy(delta[:,None],ts[None,:]),0.)
    grid=np.logspace(np.log10(.05),np.log10(10),65);values=loss(grid);i=int(values.argmin());best,best_loss=grid[i],values[i]
    lo,hi=grid[max(0,i-1)],grid[min(64,i+1)]
    for _ in range(4):
        grid=np.linspace(lo,hi,17);values=loss(grid);i=int(values.argmin())
        if values[i]<best_loss:best,best_loss=grid[i],values[i]
        lo,hi=grid[max(0,i-1)],grid[min(16,i+1)]
    return float(best)

def calibrate(prepared,groups,c):
    a=c['alpha_blend'];ln2=np.log(2.)
    h=float(np.clip(c['h_min']/ln2+c.get('h_slope',0.)*abs(prepared['estimated'][1]-prepared['prior'][1]),.001,.99)*ln2)
    delta=prepared['logits'][:,1]-prepared['logits'][:,0]
    blended=(1-a)*prepared['estimated']+a*prepared['prior']
    if 2*abs(expit(delta).mean()-prepared['estimated'][1])>.01:
        bias=np.log(blended)-np.log(prepared['prior'])
        delta=delta+c['shift_strength']*(bias[1]-bias[0])
    tg=global_t(delta,blended[1],h,c['lambda_entropy'])
    roots={int(g):guard(delta[groups==g],h) for g in np.unique(groups)}
    result=delta.copy()
    for g in np.unique(groups):
        mask=groups==g;n=int(mask.sum());w=n/(n+c['tau']*n**c['gamma'])
        result[mask]/=max(w*roots[int(g)]+(1-w)*tg,roots[int(g)])
    return result,dict(global_temperature=tg,h=h)



"""Frozen source preprocessing and the existing Ours calibration equations."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

def beta_features(logits):
    d=np.asarray(logits)[:,1]-np.asarray(logits)[:,0]
    return np.c_[-np.logaddexp(0.,-d),np.logaddexp(0.,d),np.ones(len(d))]

def fit_beta(logits,labels,ridge):
    x=beta_features(logits); y=np.asarray(labels); identity=np.array([1.,1.,0.])
    def loss(b):
        z=x@b; diff=b-identity
        return float((np.logaddexp(0.,z)-y*z).mean()+ridge*(diff@diff)),x.T@(expit(z)-y)/len(y)+2*ridge*diff
    result=minimize(loss,identity,jac=True,method='L-BFGS-B',bounds=[(.05,10.),(.05,10.),(-10.,10.)],options=dict(ftol=1e-12,maxiter=500))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x

def transform_beta(logits,parameters):
    delta=beta_features(logits)@parameters
    return np.c_[np.zeros(len(delta)),delta]

class OursPipeline:
    def __init__(self, profile):
        # The caller chooses the experiment-specific preprocessing and parameters.
        self.profile=profile

    def fit_source(self, logits, labels):
        logits=np.asarray(logits,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
        self.prior=torch.tensor([1-labels.mean(),labels.mean()],dtype=torch.float64)
        mode=self.profile['precalibration']
        if mode=='beta01':
            params=fit_beta(logits,labels,.01)
            self.transform=lambda z:transform_beta(np.asarray(z,dtype=np.float64),params)
        else:
            fitted=AdaptivePriorCalibrator(num_classes=2,source_calibration=mode)
            fitted.fit_source(torch.tensor(logits),torch.tensor(labels))
            self.transform=lambda z:fitted.transform_source(torch.as_tensor(z,dtype=torch.float64)).numpy()
        return self

    def predict_logits(self, target_logits, groups, verify_torch=False):
        # Neither target labels nor the true target proportion are accepted here.
        z=torch.tensor(self.transform(target_logits),dtype=torch.float64)
        estimator=AdaptivePriorCalibrator(num_classes=2)
        estimated=estimator._estimate_target_prior_mlls(z.softmax(1),self.prior)
        delta,diag=calibrate(dict(logits=z.numpy(),prior=self.prior.numpy(),estimated=estimated.numpy()),np.asarray(groups),self.profile['config'])
        output=np.c_[np.zeros(len(delta)),delta]
        self.diagnostics=dict(estimated_prior=estimated.tolist(),**diag)
        if not verify_torch:return output
        config=dict(self.profile['config'])
        assert config.pop('h_slope',0.)==0., 'Only frozen fixed-entropy profiles supported'
        model=PriorAwareEntropyCalibrator(num_classes=2,**config)
        shifted,blended=model._prior_aware_logit_shift(z,estimated,self.prior)
        temperature=model._optimize_temperature(shifted,blended)
        result=model.calibrate_slices(shifted,blended,torch.as_tensor(groups),temperature)
        self.diagnostics=dict(estimated_prior=estimated.tolist(),global_temperature=float(temperature))
        np.testing.assert_allclose(expit(delta),result.softmax(1).numpy()[:,1],atol=2e-7,rtol=2e-7)
        return output


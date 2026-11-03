"""V4 interactive atmospheric inversion with saved GP or legacy MLP models.

Self-contained entrypoint: no local helper modules are required.
"""
from __future__ import annotations

# 数据检查与模型推理
import csv

import hashlib

import warnings

from pathlib import Path

import numpy as np

import torch





PARAMETER_NAMES = ['T_top_K', 'T_bottom_K', 'log10_X_H2O', 'log10_X_CO', 'log10_X_CH4', 'log10_X_NH3']

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_dataset(directory):
    directory = Path(directory)
    def read(name):
        with (directory / name).open(encoding='utf-8-sig', newline='') as f:
            rows = csv.reader(f)
            header = next(rows)
            values = np.array([[float(v) for v in row] for row in rows])
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f'{name}: empty, malformed or nonfinite data')
        return header, values
    hx, tx = read('ml_inputs.csv')
    hy, ty = read('ml_log_flux.csv')
    if hx[1:] != PARAMETER_NAMES or tx.shape[1] != len(PARAMETER_NAMES) + 1:
        raise ValueError('Unexpected parameter columns/order')
    if len(tx) != len(ty) or not np.array_equal(tx[:, 0], ty[:, 0]):
        raise ValueError('Input/target sample IDs or row order do not match')
    ids = tx[:, 0].astype(np.int64)
    if not np.array_equal(ids, tx[:, 0]) or len(np.unique(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique integers')
    x, y = tx[:, 1:], ty[:, 1:]
    if len(np.unique(np.round(x, 12), axis=0)) != len(x):
        raise ValueError('Duplicate atmospheric parameters: split leakage risk')
    if np.any(x[:, :2] <= 0) or np.any(x[:, 1] <= x[:, 0]):
        raise ValueError('Require positive temperatures and T_bottom > T_top')
    nu = np.array([float(v.removeprefix('nu_').removesuffix('_cm-1')) for v in hy[1:]])
    if len(nu) != y.shape[1] or not np.isfinite(nu).all() or np.any(np.diff(nu) <= 0) or np.any(nu <= 0):
        raise ValueError('Invalid spectral grid')
    return ids, x, y, nu

def check_split(sample_ids, split):
    groups = [list(map(int, split[k])) for k in ('train', 'validation', 'test')]
    if any(not g or len(set(g)) != len(g) for g in groups):
        raise ValueError('Empty split or repeated IDs within a split')
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError('Training, validation and test IDs overlap')
    if set().union(*map(set, groups)) != set(map(int, sample_ids)):
        raise ValueError('Split does not exactly cover current sample IDs')
    lookup = {int(v): i for i, v in enumerate(sample_ids)}
    return {k: np.array([lookup[int(i)] for i in split[k]]) for k in ('train', 'validation', 'test')}

def planck(x, nu, column=1):
    n = np.asarray(nu)[None, :] * 100
    return np.pi * 2 * 6.62607015e-34 * 299792458.**2 * n**3 * 100 / np.expm1(6.62607015e-34 * 299792458. * n / (1.380649e-23 * x[:, column, None]))

def encode_target(x, y, nu, mode):
    if mode == 'log_flux':
        return y.copy()
    bottom = planck(x, nu)
    if mode == 'log_ratio':
        return y - np.log10(bottom)
    if mode == 'contrast':
        return (bottom - 10**y) / (bottom - planck(x, nu, 0))
    raise ValueError(f'Unknown target mode: {mode}')

def error_metrics(truth, prediction):
    error = prediction - truth
    relative = np.abs(np.expm1(np.log(10) * error)) * 100
    return dict(log10_rmse=float(np.sqrt(np.mean(error**2))), log10_mae=float(np.mean(np.abs(error))),
                flux_mape_percent=float(np.mean(relative)), flux_rmsre_percent=float(np.sqrt(np.mean(relative**2))),
                flux_p95_percent=float(np.percentile(relative, 95)), flux_max_percent=float(np.max(relative)))

class KernelGroup(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        for key in ('length_scale', 'alpha', 'basis'):
            self.register_buffer(key, torch.as_tensor(config[key], dtype=torch.float64))
        self.register_buffer('amplitude', torch.as_tensor(config['amplitude'], dtype=torch.float64))

class SpectralGaussianProcess(torch.nn.Module):
    """Standardized x -> standardized log10 flux; compatible with V2 retrieval.

    All buffers are float64. Gradients flow through inputs, while learned
    coefficients stay fixed during atmospheric inversion.
    """
    def __init__(self, config):
        super().__init__()
        self.mode, self.family = config['mode'], config['family']
        for key in ('x_mean', 'x_std', 'centers', 'z_mean', 'nu', 'y_mean', 'y_std'):
            self.register_buffer(key, torch.as_tensor(config[key], dtype=torch.float64))
        self.groups = torch.nn.ModuleList([KernelGroup(g) for g in config['groups']])

    def forward(self, values):
        xx = values.to(torch.float64)
        delta = xx[:, None, :] - self.centers[None, :, :]
        z = self.z_mean[None, :].expand(len(xx), -1)
        for group in self.groups:
            r2 = ((delta / group.length_scale)**2).sum(-1)
            if self.family == 'rbf':
                kernel = torch.exp(-0.5 * r2)
            else:
                r = torch.sqrt(torch.clamp(r2, min=1e-30))
                if self.family == 'matern25':
                    a = np.sqrt(5.) * r
                    kernel = (1 + a + 5 / 3 * r2) * torch.exp(-a)
                else:
                    a = np.sqrt(3.) * r
                    kernel = (1 + a) * torch.exp(-a)
            z = z + ((kernel * group.amplitude) @ group.alpha) @ group.basis
        if self.mode != 'log_flux':
            physical = xx * self.x_std + self.x_mean
            n = self.nu[None, :] * 100
            constant = np.pi * 2 * 6.62607015e-34 * 299792458.**2 * n**3 * 100
            exponent = 6.62607015e-34 * 299792458. * n / 1.380649e-23
            bottom = constant / torch.expm1(exponent / physical[:, 1, None])
            if self.mode == 'log_ratio':
                z = z + torch.log10(bottom)
            else:
                top = constant / torch.expm1(exponent / physical[:, 0, None])
                z = torch.log10(torch.clamp(bottom - z * (bottom - top), min=1e-30))
        return (z - self.y_mean) / self.y_std

    @torch.inference_mode()
    def predict(self, x, batch_size=64):
        predictions = []
        for start in range(0, len(x), batch_size):
            raw = torch.as_tensor(x[start:start+batch_size], dtype=torch.float64, device=self.x_mean.device)
            predictions.append((self((raw-self.x_mean)/self.x_std)*self.y_std+self.y_mean).cpu().numpy())
        return np.concatenate(predictions)

def load_saved_model(checkpoint, device='cpu'):
    if checkpoint.get('parameter_names') != PARAMETER_NAMES or checkpoint.get('input_dim') != len(PARAMETER_NAMES):
        raise ValueError('V4 requires a six-parameter NH3 checkpoint; regenerate data and retrain in V4.0')
    if checkpoint.get('model_type') == 'spectral_gp_v4':
        model = SpectralGaussianProcess(checkpoint['gp_config']).to(device)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(len(PARAMETER_NAMES), 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, int(checkpoint['output_dim']))).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

def validate_checkpoint_data(checkpoint, directory):
    """Reject contaminated splits, altered data, and contaminated fit statistics."""
    if checkpoint.get('parameter_names') != PARAMETER_NAMES or checkpoint.get('input_dim') != len(PARAMETER_NAMES):
        raise ValueError('V4 requires a six-parameter NH3 checkpoint; regenerate data and retrain in V4.0')
    ids, x, y, _ = read_dataset(directory)
    if checkpoint.get('model_type') != 'spectral_gp_v4':
        train, test = map(set, (checkpoint['train_sample_ids'], checkpoint['test_sample_ids']))
        if train & test:
            raise ValueError('Legacy checkpoint has training/test overlap')
        return
    split = {k: list(checkpoint[k + '_sample_ids']) for k in ('train', 'validation', 'test')}
    rows = check_split(ids, split)
    tr = rows['train']
    for name, expected in checkpoint['data_sha256'].items():
        if file_hash(Path(directory) / name) != expected:
            raise ValueError(f'Checkpoint dataset SHA256 mismatch: {name}')
    for key, expected in [('x_mean', x[tr].mean(0)), ('x_std', np.where(x[tr].std(0)>0, x[tr].std(0), 1.)),
                          ('y_mean', y[tr].mean(0)), ('y_std', np.maximum(y[tr].std(0), 1e-12))]:
        if not np.allclose(checkpoint[key], expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f'{key} is not fitted exclusively on training rows')
    expected = (x[tr] - checkpoint['x_mean']) / checkpoint['x_std']
    if not np.allclose(checkpoint['gp_config']['centers'], expected, rtol=1e-12, atol=1e-12):
        raise ValueError('Kernel centers differ from training rows')
    config = checkpoint['gp_config']
    encoded = encode_target(x[tr], y[tr], config['nu'], config['mode'])
    if not np.allclose(config['z_mean'], encoded.mean(0), rtol=1e-12, atol=1e-12):
        raise ValueError('PCA mean is not training-only')
    for key in ('x_mean', 'x_std', 'y_mean', 'y_std', 'centers', 'z_mean', 'nu'):
        saved = checkpoint['model_state_dict'][key].detach().cpu().numpy()
        if not np.array_equal(saved, np.asarray(config[key])):
            raise ValueError(f'Inference state/config mismatch: {key}')
        if key in checkpoint and not np.array_equal(np.asarray(checkpoint[key]), np.asarray(config[key])):
            raise ValueError(f'Checkpoint normalization/config mismatch: {key}')

# V4 解析梯度与多起点最小二乘
import argparse,json,re,time

from pathlib import Path

import numpy as np

from scipy.optimize import least_squares

from threadpoolctl import threadpool_limits

import torch

LOW=np.array([400.,900.,-5.5,-5.5,-7.,-7.])

HIGH=np.array([800.,1800.,-2.5,-3.,-3.5,-3.5])

NAMES=['T_top_K','T_bottom_K','X_H2O','X_CO','X_CH4', 'X_NH3']

class NumpyGP:
    def __init__(self,config):
        self.config=config
        self.independent=all(g['alpha'].shape[1]==1 for g in config['groups'])
        if self.independent:
            self.lengths=np.array([g['length_scale'] for g in config['groups']])
            self.amplitude=np.array([g['amplitude'] for g in config['groups']])
            self.alpha=np.stack([g['alpha'][:,0] for g in config['groups']])
            self.basis=np.concatenate([g['basis'] for g in config['groups']],axis=0)

    def predict_jacobian(self,theta):
        c=self.config
        xx=(theta-c['x_mean'])/c['x_std']
        delta=xx[None,:]-c['centers']
        def kernel(lengths,amplitude):
            delta_scaled=delta[None,:,:]/lengths[:,None,:]
            r2=np.sum(delta_scaled**2,axis=-1)
            if c['family']=='rbf':
                k=amplitude[:,None]*np.exp(-.5*r2); factor=-k
            else:
                r=np.sqrt(np.maximum(r2,1e-30))
                if c['family']=='matern25':
                    a=np.sqrt(5)*r; exp=amplitude[:,None]*np.exp(-a)
                    k=(1+a+5/3*r2)*exp; factor=-5/3*(1+a)*exp
                else:
                    a=np.sqrt(3)*r; exp=amplitude[:,None]*np.exp(-a)
                    k=(1+a)*exp; factor=-3*exp
            dk=factor[:,:,None]*delta[None,:,:]/lengths[:,None,:]**2/c['x_std']
            return k,dk
        if self.independent:
            k,dk=kernel(self.lengths,self.amplitude)
            scores=np.einsum('gn,gn->g',k,self.alpha)
            dscore=np.einsum('gnp,gn->gp',dk,self.alpha)
            z=c['z_mean']+scores@self.basis
            jac=self.basis.T@dscore
        else:
            z=c['z_mean'].copy(); jac=np.zeros((len(z),len(PARAMETER_NAMES)))
            for g in c['groups']:
                k,dk=kernel(g['length_scale'][None,:],np.array([g['amplitude']]))
                z+=(k[0]@g['alpha'])@g['basis']
                jac+=g['basis'].T@(dk[0].T@g['alpha']).T
        if c['mode']!='log_flux':
            n=c['nu']*100; constant=np.pi*2*6.62607015e-34*299792458.**2*n**3*100
            a=6.62607015e-34*299792458.*n/1.380649e-23
            def blackbody(t):
                b=constant/np.expm1(a/t)
                derivative=b*a/t**2/(-np.expm1(-a/t))
                return b,derivative
            bottom,db=blackbody(theta[1])
            if c['mode']=='log_ratio':
                z+=np.log10(bottom); jac[:,1]+=db/bottom/np.log(10)
            else:
                top,dt=blackbody(theta[0]); flux=bottom-z*(bottom-top)
                raw_jac=-jac*(bottom-top)[:,None]
                raw_jac[:,0]+=z*dt; raw_jac[:,1]+=(1-z)*db
                valid=flux>1e-30
                jac=np.where(valid[:,None],raw_jac/np.maximum(flux,1e-30)[:,None]/np.log(10),0)
                z=np.log10(np.maximum(flux,1e-30))
        return z,jac

def retrieve(models,observed,x_train,y_train,y_scale,*,starts=5,max_nfev=120,loss_scaling='standardized',channel_weights=None):
    """No validation/test parameter truths are accepted by this API."""
    weights=1/np.maximum(y_scale,1e-12) if loss_scaling=='standardized' else np.ones_like(y_scale)
    # channel_weights are LOSS weights (squared-residual weights), hence sqrt on the residual.
    if channel_weights is not None: weights=weights*np.sqrt(channel_weights)
    distances=np.mean(((y_train-observed)*weights)**2,axis=1)
    neighbors=x_train[np.argsort(distances)[:starts]]
    # Additional midpoint is a fixed physical prior, independent of held-out truths.
    initial=np.vstack([neighbors,(LOW+HIGH)/2])
    best=None
    for theta0 in initial:
        cached_u=None; cached_result=None
        def evaluate(u):
            nonlocal cached_u,cached_result
            if cached_u is None or not np.array_equal(u,cached_u):
                theta=LOW+u*(HIGH-LOW)
                values=[m.predict_jacobian(theta) for m in models]
                predicted=np.concatenate([v[0] for v in values]); jac=np.vstack([v[1] for v in values])
                cached_result=((predicted-observed)*weights,jac*weights[:,None]*(HIGH-LOW)[None,:])
                cached_u=u.copy()
            return cached_result
        result=least_squares(lambda u:evaluate(u)[0],np.clip((theta0-LOW)/(HIGH-LOW),1e-8,1-1e-8),
                             jac=lambda u:evaluate(u)[1],bounds=(np.zeros(len(PARAMETER_NAMES)),np.ones(len(PARAMETER_NAMES))),
                             max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        if best is None or result.cost<best.cost: best=result
    return LOW+best.x*(HIGH-LOW),float(best.cost),int(best.nfev),bool(best.success)

def physical(x):
    values=x.copy(); values[:,2:]=10**values[:,2:]; return values

def save_inversion_spectra(output_dir, sample_ids, window_names, grids, truth, prediction):
    """Save all retrieved spectra, without a second retrieval or model update."""
    np.savez_compressed(
        Path(output_dir)/'inversion_spectra.npz',
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        window_names=np.asarray(window_names, dtype=str),
        window_sizes=np.asarray([len(nu) for nu in grids], dtype=np.int64),
        wavenumber=np.concatenate(grids),
        true_log_flux=np.asarray(truth, dtype=np.float64),
        retrieved_log_flux=np.asarray(prediction, dtype=np.float64),
        parameter_csv_sha256=np.array(file_hash(Path(output_dir)/'inversion_error_by_sample.csv')),
    )


def parse_window_weights(text):
    """'1200-1400=1,1900-2300=2' -> {'window_1200_1400':1.0,'window_1900_2300':2.0}."""
    out={}
    for item in filter(None,(t.strip() for t in (text or '').split(','))):
        m=re.fullmatch(r'([0-9.]+)-([0-9.]+)=([0-9.eE+-]+)',item)
        if not m: raise ValueError(f'窗口权重格式应为 1200-1400=2.0，收到：{item!r}')
        w=float(m.group(3))
        if not np.isfinite(w) or w<0: raise ValueError(f'窗口权重必须是非负有限数：{item!r}')
        out[f'window_{float(m.group(1)):g}_{float(m.group(2)):g}']=w
    return out

def run_gp_retrieval(input_dir,output_dir,split_name='test',starts=5,max_nfev=120,loss_scaling='standardized',sample_limit=None,window_weights=None):
    input_dir,output_dir=Path(input_dir),Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    payload=[]; ref=None
    windows=sorted(p for p in input_dir.glob('window_*') if (p/'surrogate_model.pt').is_file())
    for window in windows or [input_dir]:
        if not (window/'surrogate_model.pt').is_file(): continue
        ck=torch.load(window/'surrogate_model.pt',map_location='cpu',weights_only=False)
        if ck.get('model_type')!='spectral_gp_v4': raise ValueError('Requires frozen V4 GP checkpoints')
        validate_checkpoint_data(ck,window)
        ids,x,y,nu=read_dataset(window)
        split={k:ck[k+'_sample_ids'].tolist() for k in ['train','validation','test']}
        rows=check_split(ids,split)
        if ref is None: ref=(ids,x,split,rows)
        elif not np.array_equal(ids,ref[0]) or not np.array_equal(x,ref[1]) or split!=ref[2]: raise ValueError('Window data/split mismatch')
        payload.append((window,ck,y,nu))
    if not payload: raise ValueError('No window checkpoints found')
    ids,x,split,rows=ref; tr=rows['train']; ev=rows[split_name]
    if sample_limit is not None:
        if sample_limit < 1 or sample_limit > len(ev): raise ValueError('Invalid evaluation sample count')
        ev=ev[:sample_limit]
    models=[NumpyGP(p[1]['gp_config']) for p in payload]
    y=np.concatenate([p[2] for p in payload],axis=1)
    y_scale=np.concatenate([p[1]['y_std'] for p in payload])
    channel_weights=None
    if window_weights:
        unknown=set(window_weights)-{p[0].name for p in payload}
        if unknown: raise ValueError(f'窗口权重指定了不存在的窗口：{sorted(unknown)}；可用窗口：{[p[0].name for p in payload]}')
        channel_weights=np.concatenate([np.full(p[2].shape[1],window_weights.get(p[0].name,1.0)) for p in payload])
        channel_weights=channel_weights/np.mean(channel_weights)   # keep overall loss scale
        print('窗口权重（归一化后均值=1）：',{p[0].name:round(float(window_weights.get(p[0].name,1.0)/np.mean(np.concatenate([np.full(q[2].shape[1],window_weights.get(q[0].name,1.0)) for q in payload]))),4) for p in payload},flush=True)
    y_train=y[tr].copy(); x_train=x[tr].copy()
    observed=y[ev].copy(); del y
    estimates=[]; diagnostics=[]; spectra=[]
    started=time.perf_counter()
    with threadpool_limits(limits=1):
      for rank,obs in enumerate(observed):
        theta,cost,nfev,success=retrieve(models,obs,x_train,y_train,y_scale,starts=starts,max_nfev=max_nfev,loss_scaling=loss_scaling,channel_weights=channel_weights)
        estimates.append(theta); diagnostics.append([int(ids[ev[rank]]),cost,nfev,int(success)])
        spectra.append(np.concatenate([m.predict_jacobian(theta)[0] for m in models]))
        if rank%10==0: print(f'{split_name} retrieval {rank+1}/{len(ev)} cost={cost:.5g}',flush=True)
    estimates=np.array(estimates); predicted=np.array(spectra)
    # Ground-truth atmospheric parameters are accessed for scoring only here.
    truth=x[ev]; pt,pe=physical(truth),physical(estimates)
    rel=(pe-pt)/pt*100
    metrics={name:dict(mae_percent=float(np.abs(rel[:,j]).mean()),rmse_percent=float(np.sqrt(np.mean(rel[:,j]**2))),
                       p95_abs_percent=float(np.percentile(np.abs(rel[:,j]),95))) for j,name in enumerate(NAMES)}
    report=dict(split=split_name,fit_rows=len(tr),evaluation_rows=len(ev),train_evaluation_overlap=0,
                starts=starts,max_nfev=max_nfev,loss_scaling=loss_scaling,window_weights=window_weights or {},
                model_weights_updated=False,initialization_library='training_rows_only',
                parameters=metrics,normalized_parameter_rmse=float(np.sqrt(np.mean(((estimates-truth)/(HIGH-LOW))**2))),
                log10_abundance_rmse=float(np.sqrt(np.mean((estimates[:,2:]-truth[:,2:])**2))),
                spectrum=error_metrics(observed,predicted),optimizer_success_rate=float(np.mean(np.array(diagnostics)[:,-1])),
                elapsed_seconds=time.perf_counter()-started,
                checkpoint_sha256={p[0].name:file_hash(p[0]/'surrogate_model.pt') for p in payload})
    (output_dir/'inversion_metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    header=['sample_id','spectral_cost','nfev','optimizer_success']+[f'true_{n}' for n in NAMES]+[f'retrieved_{n}' for n in NAMES]+[f'{n}_error_percent' for n in NAMES]
    np.savetxt(output_dir/'inversion_error_by_sample.csv',np.column_stack([diagnostics,pt,pe,rel]),delimiter=',',header=','.join(header),comments='',fmt='%.12g')
    lines=['Frozen V4 GP joint-window inversion',f'evaluation_split={split_name}',f'training_library_samples={len(tr)}',
           f'inversion_samples={len(ev)}','validation_or_test_rows_in_training_library=false','model_weights_updated=false']
    for name,v in metrics.items():
        for key,value in v.items(): lines.append(f'parameter_{name}_{key}={value:.10e}')
    lines.extend([f'normalized_parameter_rmse={report["normalized_parameter_rmse"]:.10e}',
                  f'elapsed_seconds={report["elapsed_seconds"]:.3f}',
                  'scope=Noiseless synthetic spectra from the existing forward solver; not experimental retrieval accuracy.'])
    (output_dir/'inversion_error_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_inversion_spectra(output_dir, ids[ev], [p[0].name for p in payload],
                           [p[3] for p in payload], observed, predicted)
    offset = 0
    window_records = []
    for p in payload:
        sl = slice(offset, offset + len(p[3]))
        offset += len(p[3])
        wm = error_metrics(observed[:, sl], predicted[:, sl])
        window_records.append([p[0].name, wm['log10_rmse'], wm['flux_mape_percent'], wm['flux_rmsre_percent']])
    import csv
    with (output_dir/'inversion_error_by_window.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['window','log10_rmse','flux_mape_percent','flux_rms_relative_percent']); writer.writerows(window_records)
    print(json.dumps(report),flush=True)
    return report

# 终端交互、V2 模型兼容与统一入口
import argparse

import csv

import math

import time

from pathlib import Path

from typing import Any




import numpy as np

H = 6.62607015e-34

C = 299792458.0

K_B = 1.380649e-23

PARAMETER_NAMES = [
    "T_top_K",
    "T_bottom_K",
    "log10_X_H2O",
    "log10_X_CO",
    "log10_X_CH4",
    "log10_X_NH3",
]

PARAMETER_BOUNDS = np.asarray(
    [
        [400.0, 800.0],
        [900.0, 1800.0],
        [-5.5, -2.5],
        [-5.5, -3.0],
        [-7.0, -3.5],
        [-7.0, -3.5],
    ],
    dtype=np.float32,
)

REPORT_PARAMETER_NAMES = ["T_top_K", "T_bottom_K", "X_H2O", "X_CO", "X_CH4", "X_NH3"]

def physical_parameters(values: np.ndarray) -> np.ndarray:
    """Convert model log10 abundances to dimensionless number fractions."""
    result = np.array(values, dtype=np.float64, copy=True)
    result[..., 2:] = np.power(10.0, result[..., 2:])
    return result

def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV file: {path}") from exc
        values = np.asarray([[float(value) for value in row] for row in reader], dtype=float)
    return header, values

def resolve_sample_count(value: int | None, available: int) -> int:
    if value is None:
        try:
            raw = input(
                f"请输入用于反演误差计算的测试样本数（1-{available}，输入 0 使用全部）："
            ).strip()
        except EOFError as exc:
            raise ValueError(
                "未读取到反演样本数；非交互运行请使用 --inversion-samples N"
            ) from exc
        if not raw:
            raise ValueError("反演样本数不能为空")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("反演样本数必须是整数") from exc
    if value == 0:
        return available
    if value < 0 or value > available:
        raise ValueError(f"反演样本数必须为 0 或位于 1-{available} 之间")
    return value

def resolve_torch_device(torch: Any, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if requested == "gpu":
        raise RuntimeError("PyTorch CUDA 不可用；请使用 --device cpu")
    return torch.device("cpu")

def brightness_temperature(nu_cm: np.ndarray, spectral_flux: np.ndarray) -> np.ndarray:
    nu_m = np.asarray(nu_cm, dtype=float)[None, :] * 100.0
    flux = np.asarray(spectral_flux, dtype=float)
    b_per_m = np.maximum(flux / math.pi / 100.0, np.finfo(float).tiny)
    numerator = H * C * nu_m / K_B
    denominator = np.log1p(2.0 * H * C**2 * nu_m**3 / b_per_m)
    return numerator / denominator

def build_model(torch: Any, output_channels: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(len(PARAMETER_NAMES), 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, output_channels),
    )

def theta_to_unconstrained(torch: Any, theta: np.ndarray, low: Any, high: Any) -> Any:
    fraction = (torch.as_tensor(theta, dtype=torch.float32, device=low.device) - low) / (high - low)
    fraction = torch.clamp(fraction, 1.0e-5, 1.0 - 1.0e-5)
    return torch.log(fraction / (1.0 - fraction))

def optimize_one_sample(
    torch: Any,
    model: Any,
    observed_log_flux: np.ndarray,
    x_mean: Any,
    x_std: Any,
    y_mean: Any,
    y_std: Any,
    low: Any,
    high: Any,
    starts: int,
    steps: int,
    learning_rate: float,
    seed: int,
    tolerance: float,
    nearest_parameters: np.ndarray | None = None,
    nearest_count: int = 3,
    lbfgs_steps: int = 30,
    prior_parameters: np.ndarray | None = None,
    prior_weight: float = 0.0,
    trust_radius: float = 0.0,
    spectral_channel_weights: Any | None = None,
) -> tuple[np.ndarray, float, float, int, bool]:
    """Optimize one spectrum with data-guided starts and local refinement.

    The nearest training spectra provide valid initialization points without
    using the held-out sample's true parameters. Adam explores the bounded
    parameter space, while LBFGS refines the best Adam state. The best state
    seen during every stage is retained instead of blindly returning the last
    optimizer iterate.
    """
    rng = np.random.default_rng(seed)
    span = high.detach().cpu().numpy() - low.detach().cpu().numpy()
    global_lower = low.detach().cpu().numpy() + 0.01 * span
    global_upper = high.detach().cpu().numpy() - 0.01 * span
    trust_enabled = prior_parameters is not None and trust_radius > 0.0
    if trust_enabled:
        prior_center = np.clip(np.asarray(prior_parameters), global_lower, global_upper)
        lower = np.maximum(global_lower, prior_center - trust_radius * span)
        upper = np.minimum(global_upper, prior_center + trust_radius * span)
    else:
        prior_center = None
        lower = global_lower
        upper = global_upper
    initial_points = [0.5 * (lower + upper)]
    if prior_parameters is not None:
        initial_points.append(np.clip(prior_parameters, lower, upper))
    if nearest_parameters is not None and nearest_count > 0:
        for candidate in np.asarray(nearest_parameters)[:nearest_count]:
            initial_points.append(np.clip(candidate, lower, upper))
    while len(initial_points) < starts:
        initial_points.append(rng.uniform(lower, upper, size=len(PARAMETER_NAMES)))
    initial_points = initial_points[:starts]
    observed = torch.as_tensor(observed_log_flux, dtype=torch.float32, device=low.device)
    observed_scaled = (observed - y_mean) / y_std

    prior_tensor = None
    if prior_parameters is not None and prior_weight > 0.0:
        prior_tensor = torch.as_tensor(
            np.clip(prior_parameters, global_lower, global_upper),
            dtype=torch.float32,
            device=low.device,
        )

    trust_center_tensor = None
    trust_span_tensor = None
    if trust_enabled:
        trust_center_tensor = torch.as_tensor(prior_center, dtype=torch.float32, device=low.device)
        trust_span_tensor = torch.as_tensor(
            trust_radius * span, dtype=torch.float32, device=low.device
        )

    def theta_for(variable: Any) -> Any:
        if trust_center_tensor is not None and trust_span_tensor is not None:
            theta = trust_center_tensor + trust_span_tensor * torch.tanh(variable)
            return torch.maximum(
                torch.minimum(theta, high), low
            )
        return low + (high - low) * torch.sigmoid(variable)

    def variable_for(theta: np.ndarray) -> Any:
        clipped = np.clip(theta, lower, upper)
        if trust_center_tensor is not None and trust_span_tensor is not None:
            fraction = (
                torch.as_tensor(clipped, dtype=torch.float32, device=low.device)
                - trust_center_tensor
            ) / trust_span_tensor
            return torch.atanh(torch.clamp(fraction, -0.999, 0.999))
        return theta_to_unconstrained(torch, clipped, low, high)

    def spectral_loss_for(variable: Any) -> Any:
        theta = theta_for(variable)
        x_scaled = (theta - x_mean) / x_std
        prediction_scaled = model(x_scaled[None, :])[0]
        squared_error = (prediction_scaled - observed_scaled) ** 2
        if spectral_channel_weights is None:
            return torch.mean(squared_error)
        return torch.sum(spectral_channel_weights * squared_error) / torch.sum(
            spectral_channel_weights
        )

    def loss_for(variable: Any) -> Any:
        spectral_loss = spectral_loss_for(variable)
        if prior_tensor is None:
            return spectral_loss
        theta = theta_for(variable)
        normalized_distance = torch.mean(((theta - prior_tensor) / (high - low)) ** 2)
        return spectral_loss + prior_weight * normalized_distance

    best_theta: np.ndarray | None = None
    best_loss = float("inf")
    best_initial_loss = float("inf")
    best_steps = 0
    best_converged = False
    best_selection_score = float("inf")
    parameter_span = high.detach().cpu().numpy() - low.detach().cpu().numpy()

    if prior_parameters is not None:
        prior_variable = variable_for(prior_parameters)
        prior_spectral_loss = float(spectral_loss_for(prior_variable).detach())
        best_theta = np.clip(np.asarray(prior_parameters), global_lower, global_upper).copy()
        best_loss = prior_spectral_loss
        best_initial_loss = prior_spectral_loss
        best_steps = 0
        best_converged = prior_spectral_loss <= tolerance
        best_selection_score = prior_spectral_loss

    for initial_theta in initial_points:
        unconstrained = variable_for(initial_theta)
        variable = torch.nn.Parameter(unconstrained)
        optimizer = torch.optim.Adam([variable], lr=learning_rate)
        initial_loss = None
        local_best_loss = float("inf")
        local_best_variable = variable.detach().clone()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(variable)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss_value = float(loss.detach())
            if loss_value < local_best_loss:
                local_best_loss = loss_value
                local_best_variable = variable.detach().clone()
            loss.backward()
            optimizer.step()
        variable.data.copy_(local_best_variable)

        if lbfgs_steps > 0:
            refinement = torch.optim.LBFGS(
                [variable],
                lr=0.8,
                max_iter=lbfgs_steps,
                tolerance_grad=1.0e-8,
                tolerance_change=1.0e-10,
                history_size=10,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Any:
                refinement.zero_grad()
                refinement_loss = loss_for(variable)
                refinement_loss.backward()
                return refinement_loss

            refinement.step(closure)
            refined_loss = float(loss_for(variable).detach())
            if refined_loss < local_best_loss:
                local_best_loss = refined_loss
                local_best_variable = variable.detach().clone()
            else:
                variable.data.copy_(local_best_variable)

        final_loss = local_best_loss
        converged = final_loss <= tolerance
        actual_steps = steps + max(0, lbfgs_steps)
        theta_np = (
            theta_for(local_best_variable)
        ).detach().cpu().numpy()
        prior_distance = 0.0
        if prior_parameters is not None and prior_weight > 0.0:
            prior_distance = float(
                np.mean(((theta_np - np.asarray(prior_parameters)) / parameter_span) ** 2)
            )
        selection_score = final_loss + prior_weight * prior_distance
        if selection_score < best_selection_score:
            best_theta = theta_np
            best_loss = final_loss
            best_initial_loss = float(initial_loss if initial_loss is not None else final_loss)
            best_steps = actual_steps
            best_converged = converged
            best_selection_score = selection_score
        if best_converged:
            # A solution below tolerance is already sufficient for this sample.
            break
    if best_theta is None:
        raise RuntimeError("No inversion result was produced")
    return best_theta, best_initial_loss, best_loss, best_steps, best_converged

def save_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

def relative_error_percent(error: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Signed pointwise relative error; zero truth has no defined percentage."""
    truth = np.asarray(truth, dtype=np.float64)
    if np.any(~np.isfinite(truth)) or np.any(truth == 0.0):
        raise ValueError("百分比误差要求真实值为有限的非零数")
    return 100.0 * np.asarray(error, dtype=np.float64) / np.abs(truth)

def parameter_metrics(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "bias": float(np.mean(values)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95.0)),
        "max_abs": float(np.max(absolute)),
    }

def load_window_surrogate(window_dir: Path, torch: Any, device: Any) -> dict[str, Any]:
    """Load one independently trained spectral-window surrogate."""
    _, x_table = read_numeric_csv(window_dir / "ml_inputs.csv")
    flux_header, y_table = read_numeric_csv(window_dir / "ml_log_flux.csv")
    if x_table.shape[1] != len(PARAMETER_NAMES) + 1 or y_table.shape[1] < 2:
        raise SystemExit(f"窗口输入格式无效：{window_dir}")
    if len(x_table) != len(y_table):
        raise SystemExit(f"窗口输入样本数不一致：{window_dir}")
    sample_ids = x_table[:, 0].astype(int)
    x = x_table[:, 1:].astype(np.float32)
    y = y_table[:, 1:].astype(np.float32)
    model_path = window_dir / "surrogate_model.pt"
    if not model_path.is_file():
        raise SystemExit(
            f"窗口 {window_dir.name} 缺少 surrogate_model.pt；请先运行 train_surrogate.py"
        )
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise SystemExit(f"窗口替代模型检查点格式无效：{model_path}")
    input_dim = int(checkpoint.get("input_dim", x.shape[1]))
    output_dim = int(checkpoint.get("output_dim", y.shape[1]))
    if input_dim != x.shape[1] or output_dim != y.shape[1]:
        raise SystemExit(
            f"窗口 {window_dir.name} 的模型与 CSV 维度不一致："
            f"checkpoint=({input_dim},{output_dim}), csv=({x.shape[1]},{y.shape[1]})"
        )
    try:
        x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
        x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
        y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
        y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
        train_sample_ids = np.asarray(checkpoint["train_sample_ids"], dtype=np.int64)
        test_sample_ids = np.asarray(checkpoint["test_sample_ids"], dtype=np.int64)
    except KeyError as exc:
        raise SystemExit(f"窗口检查点缺少字段 {exc.args[0]}：{model_path}") from exc
    if x_mean.shape != (input_dim,) or x_std.shape != (input_dim,):
        raise SystemExit(f"窗口输入归一化参数维度无效：{model_path}")
    if y_mean.shape != (output_dim,) or y_std.shape != (output_dim,):
        raise SystemExit(f"窗口输出归一化参数维度无效：{model_path}")
    if np.any(x_std <= 0.0) or np.any(y_std <= 0.0):
        raise SystemExit(f"窗口检查点中的归一化标准差必须为正：{model_path}")
    validate_checkpoint_data(checkpoint, window_dir)
    model = load_saved_model(checkpoint, device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"无法加载窗口替代模型权重：{model_path}") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    sample_id_to_row = {int(sample_id): row for row, sample_id in enumerate(sample_ids)}
    try:
        train_rows = np.asarray(
            [sample_id_to_row[int(sample_id)] for sample_id in train_sample_ids], dtype=int
        )
        test_rows = np.asarray(
            [sample_id_to_row[int(sample_id)] for sample_id in test_sample_ids], dtype=int
        )
    except KeyError as exc:
        raise SystemExit(
            f"窗口 {window_dir.name} 缺少检查点记录的 sample_id={exc.args[0]}"
        ) from exc
    nu_grid = np.asarray(
        [float(field.removeprefix("nu_").removesuffix("_cm-1")) for field in flux_header[1:]],
        dtype=float,
    )
    return {
        "name": window_dir.name,
        "directory": window_dir,
        "model": model,
        "checkpoint": checkpoint,
        "model_path": model_path,
        "sample_ids": sample_ids,
        "x": x,
        "y": y,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_sample_ids": train_sample_ids,
        "test_sample_ids": test_sample_ids,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "nu_grid": nu_grid,
    }

def run_joint_window_inversion(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    window_dirs: list[Path],
) -> None:
    """Jointly invert the concatenated losses from independent windows."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("当前 Python 环境没有安装 PyTorch") from exc
    try:
        device = resolve_torch_device(torch, args.device)
    except RuntimeError as exc:
        raise SystemExit(f"GPU 初始化失败：{exc}") from exc

    payloads = [load_window_surrogate(path, torch, device) for path in window_dirs]
    reference = payloads[0]
    for payload in payloads[1:]:
        if not np.array_equal(payload["sample_ids"], reference["sample_ids"]):
            raise SystemExit("各波数窗口的 sample_id 不一致，不能联合反演")
        if not np.array_equal(payload["train_sample_ids"], reference["train_sample_ids"]):
            raise SystemExit("各波数窗口的训练集划分不一致，请使用同一批正演数据重新训练")
        if not np.array_equal(payload["test_sample_ids"], reference["test_sample_ids"]):
            raise SystemExit("各波数窗口的测试集划分不一致，请使用同一批正演数据重新训练")
        if not np.allclose(payload["x"], reference["x"], rtol=0.0, atol=1.0e-6):
            raise SystemExit("各波数窗口的大气参数输入不一致，不能联合反演")

    class JointWindowModel(torch.nn.Module):
        def __init__(self, models: list[Any]) -> None:
            super().__init__()
            self.models = torch.nn.ModuleList(models)

        def forward(self, values: Any) -> Any:
            return torch.cat([model(values) for model in self.models], dim=1)

    model = JointWindowModel([payload["model"] for payload in payloads]).to(device)
    x = reference["x"]
    y = np.concatenate([payload["y"] for payload in payloads], axis=1)
    x_train = x[reference["train_rows"]]
    x_test = x[reference["test_rows"]]
    y_train = np.concatenate(
        [payload["y"][payload["train_rows"]] for payload in payloads], axis=1
    )
    y_test = np.concatenate(
        [payload["y"][payload["test_rows"]] for payload in payloads], axis=1
    )
    x_mean = reference["x_mean"]
    x_std = reference["x_std"]
    y_mean = np.concatenate([payload["y_mean"] for payload in payloads])
    y_std = np.concatenate([payload["y_std"] for payload in payloads])
    channel_weights = np.concatenate(
        [
            np.full(
                payload["y"].shape[1],
                args.co_window_weight if payload["name"] == "window_1900_2300" else 1.0,
                dtype=np.float32,
            )
            for payload in payloads
        ]
    )
    # Keep the overall loss scale comparable to the unweighted case while
    # increasing the relative influence of the CO-dominated window.
    channel_weights /= np.mean(channel_weights)
    test_sample_ids = reference["test_sample_ids"]
    if len(x) < 5:
        raise SystemExit("至少需要 5 个正演样本才能进行联合反演")
    if np.any(x < PARAMETER_BOUNDS[:, 0]) or np.any(x > PARAMETER_BOUNDS[:, 1]):
        raise SystemExit("输入参数超出当前反演参数边界")

    inversion_count = resolve_sample_count(args.inversion_samples, len(x_test))
    selected_indices = np.arange(inversion_count)
    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_tensor = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    y_mean_tensor = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_tensor = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    channel_weights_tensor = torch.as_tensor(
        channel_weights, dtype=torch.float32, device=device
    )
    low_tensor = torch.as_tensor(PARAMETER_BOUNDS[:, 0], dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(PARAMETER_BOUNDS[:, 1], dtype=torch.float32, device=device)

    nearest_parameters_by_test = None
    prior_parameters_by_test = None
    if args.nearest_neighbors > 0:
        y_train_scaled = (y_train - y_mean) / y_std
        y_test_scaled = (y_test - y_mean) / y_std
        weighted_train_scaled = y_train_scaled * channel_weights[None, :]
        train_norm = np.sum(y_train_scaled * weighted_train_scaled, axis=1)
        test_norm = np.sum(y_test_scaled**2 * channel_weights[None, :], axis=1)
        distances = (
            test_norm[:, None]
            + train_norm[None, :]
            - 2.0 * (y_test_scaled @ weighted_train_scaled.T)
        )
        neighbor_count = min(args.nearest_neighbors, len(x_train))
        nearest_result = np.argsort(distances, axis=1)[:, :neighbor_count]
        nearest_distances = np.take_along_axis(distances, nearest_result, axis=1)
        nearest_parameters_by_test = x_train[nearest_result]
        nearest_weights = 1.0 / np.maximum(nearest_distances, 1.0e-8)
        nearest_weights /= np.sum(nearest_weights, axis=1, keepdims=True)
        prior_parameters_by_test = np.sum(
            nearest_parameters_by_test * nearest_weights[:, :, None], axis=1
        )

    started = time.perf_counter()
    print(
        f"Loaded {len(payloads)} window surrogates on "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    rows: list[list[Any]] = []
    retrieved_parameters = []
    retrieved_spectra = []
    spectral_flux_errors = []
    spectral_brightness_errors = []
    spectral_brightness_relative_errors = []
    window_error_rows: list[list[Any]] = []
    for rank, test_index in enumerate(selected_indices):
        inversion_started = time.perf_counter()
        true_parameters = x_test[test_index]
        prior = None if prior_parameters_by_test is None else prior_parameters_by_test[test_index]
        nearest = None if nearest_parameters_by_test is None else nearest_parameters_by_test[test_index]
        if args.retrieval_mode == "empirical-knn":
            if prior is None:
                raise RuntimeError("empirical-knn 模式没有可用的训练集近邻参数")
            retrieved = prior.astype(np.float32, copy=True)
            with torch.inference_mode():
                theta_tensor = torch.as_tensor(retrieved, dtype=torch.float32, device=device)
                predicted_scaled = model(((theta_tensor - x_mean_tensor) / x_std_tensor)[None, :])[0]
                observed_scaled = (
                    torch.as_tensor(y_test[test_index], dtype=torch.float32, device=device)
                    - y_mean_tensor
                ) / y_std_tensor
                final_loss = float(
                    torch.sum(
                        channel_weights_tensor
                        * (predicted_scaled - observed_scaled) ** 2
                    )
                    / torch.sum(channel_weights_tensor)
                )
            initial_loss = final_loss
            steps_used = 0
            converged = True
        else:
            retrieved, initial_loss, final_loss, steps_used, converged = optimize_one_sample(
                torch,
                model,
                y_test[test_index],
                x_mean_tensor,
                x_std_tensor,
                y_mean_tensor,
                y_std_tensor,
                low_tensor,
                high_tensor,
                args.starts,
                args.inverse_steps,
                args.inverse_learning_rate,
                args.seed + int(test_sample_ids[test_index]) + rank,
                args.inverse_tolerance,
                nearest,
                args.nearest_neighbors,
                args.lbfgs_steps,
                prior,
                args.prior_weight,
                args.trust_radius,
                channel_weights_tensor,
            )
        with torch.inference_mode():
            retrieved_tensor = torch.as_tensor(retrieved, dtype=torch.float32, device=device)
            predicted_log_flux = (
                model(((retrieved_tensor - x_mean_tensor) / x_std_tensor)[None, :])[0]
                * y_std_tensor
                + y_mean_tensor
            ).detach().cpu().numpy()
        true_log_flux = y_test[test_index]
        true_flux = np.power(10.0, true_log_flux.astype(np.float64))
        retrieved_spectra.append(predicted_log_flux.copy())
        predicted_flux = np.power(10.0, predicted_log_flux.astype(np.float64))
        flux_error = predicted_flux - true_flux
        retrieved_parameters.append(retrieved)
        spectral_flux_errors.append(predicted_flux - true_flux)
        window_brightness_relative_errors = []
        for payload_index, payload in enumerate(payloads):
            start = sum(item["y"].shape[1] for item in payloads[:payload_index])
            stop = start + payload["y"].shape[1]
            nu_grid = payload["nu_grid"]
            true_flux_window = true_flux[start:stop]
            predicted_flux_window = predicted_flux[start:stop]
            bt_error = brightness_temperature(nu_grid, predicted_flux_window[None, :])[0] - brightness_temperature(
                nu_grid, true_flux_window[None, :]
            )[0]
            true_bt_window = brightness_temperature(nu_grid, true_flux_window[None, :])[0]
            relative_bt_window = relative_error_percent(bt_error, true_bt_window)
            window_brightness_relative_errors.append(relative_bt_window)
            window_error_rows.append(
                [
                    int(test_sample_ids[test_index]),
                    payload["name"],
                    float(np.sqrt(np.mean(relative_error_percent(flux_error[start:stop], true_flux_window) ** 2))),
                    float(np.mean(np.abs(relative_error_percent(flux_error[start:stop], true_flux_window)))),
                    float(np.mean(100.0 * np.abs(predicted_flux_window - true_flux_window) / np.maximum(true_flux_window, np.finfo(float).tiny))),
                    float(np.sqrt(np.mean(relative_bt_window**2))),
                ]
            )
        spectral_brightness_relative_errors.append(np.concatenate(window_brightness_relative_errors))
        spectral_brightness_errors.append(
            np.concatenate(
                [
                    brightness_temperature(payload["nu_grid"], predicted_flux[sum(item["y"].shape[1] for item in payloads[:index]):sum(item["y"].shape[1] for item in payloads[:index + 1])][None, :])[0]
                    - brightness_temperature(payload["nu_grid"], true_flux[sum(item["y"].shape[1] for item in payloads[:index]):sum(item["y"].shape[1] for item in payloads[:index + 1])][None, :])[0]
                    for index, payload in enumerate(payloads)
                ]
            )
        )
        true_physical = physical_parameters(true_parameters)
        retrieved_physical = physical_parameters(retrieved)
        parameter_error = retrieved_physical - true_physical
        rows.append(
            [
                int(test_sample_ids[test_index]),
                int(rank),
                *true_physical,
                *retrieved_physical,
                *parameter_error,
                *np.abs(parameter_error),
                float(np.sqrt(np.mean(relative_error_percent(flux_error, true_flux)**2))),
                float(np.mean(np.abs(relative_error_percent(flux_error, true_flux)))),
                float(np.mean(100.0 * np.abs(predicted_flux - true_flux) / np.maximum(true_flux, np.finfo(float).tiny))),
                float(np.sqrt(np.mean(spectral_brightness_relative_errors[-1] ** 2))),
                initial_loss,
                final_loss,
                steps_used,
                int(converged),
                time.perf_counter() - inversion_started,
            ]
        )
        print(
            f"joint inversion {rank + 1}/{inversion_count} sample={test_sample_ids[test_index]} loss={final_loss:.6e}",
            flush=True,
        )

    retrieved_array = np.asarray(retrieved_parameters)
    true_physical_array = physical_parameters(x_test[selected_indices])
    retrieved_physical_array = physical_parameters(retrieved_array)
    parameter_errors = retrieved_physical_array - true_physical_array
    flux_errors = np.asarray(spectral_flux_errors)
    brightness_errors = np.asarray(spectral_brightness_errors)
    brightness_relative_errors = np.asarray(spectral_brightness_relative_errors)
    true_flux_selected = np.power(10.0, y_test[selected_indices].astype(np.float64))
    relative_flux_errors = 100.0 * np.abs(flux_errors) / np.maximum(
        true_flux_selected, np.finfo(float).tiny
    )
    header = [
        "sample_id",
        "test_rank",
        *[f"true_{name}" for name in REPORT_PARAMETER_NAMES],
        *[f"retrieved_{name}" for name in REPORT_PARAMETER_NAMES],
        *[f"signed_error_{name}" for name in REPORT_PARAMETER_NAMES],
        *[f"absolute_error_{name}" for name in REPORT_PARAMETER_NAMES],
        "spectral_linear_flux_rmse_percent",
        "spectral_linear_flux_mae_percent",
        "spectral_linear_flux_mean_relative_error_percent",
        "spectral_brightness_temperature_rmse_percent",
        "best_initial_loss",
        "best_final_loss",
        "optimization_steps",
        "converged",
        "elapsed_seconds",
    ]
    save_csv(output_dir / "inversion_error_by_sample.csv", header, rows)
    save_csv(
        output_dir / "inversion_error_by_window.csv",
        [
            "sample_id",
            "window",
            "spectral_linear_flux_rmse_percent",
            "spectral_linear_flux_mae_percent",
            "spectral_linear_flux_mean_relative_error_percent",
            "spectral_brightness_temperature_rmse_percent",
        ],
        window_error_rows,
    )

    parameter_units = ["K", "K"] + ["dimensionless_number_fraction"] * (len(PARAMETER_NAMES) - 2)
    summary_lines = [
        "Joint independent-window MLP surrogate inversion",
        f"input_dir={input_dir}",
        f"window_count={len(payloads)}",
        f"windows={','.join(payload['name'] for payload in payloads)}",
        f"total_forward_samples={len(x)}",
        f"surrogate_train_samples={len(x_train)}",
        f"surrogate_test_samples={len(x_test)}",
        f"inversion_samples={inversion_count}",
        f"retrieval_mode={args.retrieval_mode}",
        f"inverse_starts={args.starts}",
        f"inverse_steps={args.inverse_steps}",
        f"inverse_nearest_neighbors={args.nearest_neighbors}",
        f"inverse_lbfgs_steps={args.lbfgs_steps}",
        f"inverse_prior_weight={args.prior_weight:.8e}",
        f"inverse_trust_radius={args.trust_radius:.8e}",
        f"inverse_tolerance={args.inverse_tolerance:.8e}",
        "co_window=window_1900_2300",
        f"co_window_weight={args.co_window_weight:.6f}",
        "window_loss_weight_normalization=mean_1.0",
        f"compute_backend={'pytorch-cuda' if device.type == 'cuda' else 'pytorch-cpu'}",
        f"compute_device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        f"inversion_success_rate_percent={100.0 * np.mean(np.isfinite(retrieved_array).all(axis=1)):.6f}",
        f"inversion_converged_rate_percent={100.0 * np.mean(np.asarray([row[-2] for row in rows], dtype=float)):.6f}",
        "error_metric_space=physical_linear",
        "mae_rmse_unit=percent",
        "mae_rmse_definition=pointwise_error_divided_by_absolute_truth_times_100",
        "optimization_loss_space=standardized_log10_flux",
        "spectral_linear_flux_unit=W m^-2 (cm^-1)^-1",
        f"spectral_linear_flux_rmse_percent={np.sqrt(np.mean(relative_flux_errors**2)):.8e}",
        f"spectral_linear_flux_mae_percent={np.mean(relative_flux_errors):.8e}",
        f"spectral_linear_flux_mean_relative_error_percent={np.mean(relative_flux_errors):.8e}",
        f"spectral_brightness_temperature_rmse_percent={np.sqrt(np.mean(brightness_relative_errors**2)):.8e}",
        f"spectral_brightness_temperature_mae_percent={np.mean(np.abs(brightness_relative_errors)):.8e}",
    ]
    for index, (name, unit) in enumerate(zip(REPORT_PARAMETER_NAMES, parameter_units)):
        metrics = parameter_metrics(parameter_errors[:, index])
        percent_metrics = parameter_metrics(relative_error_percent(
            parameter_errors[:, index], true_physical_array[:, index]
        ))
        summary_lines.extend(
            [
                f"parameter_{name}_unit={unit}",
                f"parameter_{name}_bias={metrics['bias']:.8e}",
                f"parameter_{name}_mae_percent={percent_metrics['mae']:.8e}",
                f"parameter_{name}_rmse_percent={percent_metrics['rmse']:.8e}",
                f"parameter_{name}_median_abs={metrics['median_abs']:.8e}",
                f"parameter_{name}_p95_abs={metrics['p95_abs']:.8e}",
                f"parameter_{name}_max_abs={metrics['max_abs']:.8e}",
            ]
        )
    summary_lines.append(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    (output_dir / "inversion_error_report.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    save_inversion_spectra(
        output_dir, test_sample_ids[selected_indices],
        [p['name'] for p in payloads], [p['nu_grid'] for p in payloads],
        y_test[selected_indices], np.asarray(retrieved_spectra),
    )

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="saved surrogate checkpoint; default is input-dir/surrogate_model.pt",
    )
    parser.add_argument("--inversion-samples", type=int, default=None)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "least-squares", "empirical-knn", "optimize", "hybrid"),
        default="auto",
        help="auto uses bounded least-squares for V4 GP and empirical-knn for legacy MLP",
    )
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=120, help="V4 least-squares evaluations per start")
    parser.add_argument("--inverse-steps", type=int, default=1000)
    parser.add_argument("--inverse-learning-rate", type=float, default=5.0e-2)
    parser.add_argument(
        "--inverse-tolerance",
        type=float,
        default=1.0e-4,
        help="convergence threshold for standardized spectral MSE",
    )
    parser.add_argument(
        "--nearest-neighbors",
        type=int,
        default=3,
        help="training spectra used as data-guided inversion starts",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=30,
        help="LBFGS refinement steps after Adam; use 0 to disable",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=2.0e-2,
        help="weak empirical-prior weight for data-guided parameter regularization",
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=0.25,
        help="maximum parameter distance from the empirical prior as a fraction of each bound span; 0 disables",
    )
    parser.add_argument(
        "--window-weights",
        default="",
        help="V4 GP retrieval: per-window loss weights, e.g. '1200-1400=1,1900-2300=2,2800-3200=1' "
             "(normalised to mean 1; unlisted windows keep weight 1)",
    )
    parser.add_argument(
        "--co-window-weight",
        type=float,
        default=1.25,
        help="relative loss weight for the CO-dominated 1900-2300 cm-1 window",
    )
    args = parser.parse_args()

    try:
        if args.starts <= 0 or args.inverse_steps <= 0 or args.max_nfev <= 0:
            raise ValueError("starts 和 inverse-steps 必须大于 0")
        if (
            args.inverse_learning_rate <= 0.0
            or args.inverse_tolerance <= 0.0
            or args.nearest_neighbors < 0
            or args.lbfgs_steps < 0
            or args.prior_weight < 0.0
            or args.trust_radius < 0.0
            or args.co_window_weight <= 0.0
        ):
            raise ValueError(
                "inverse-learning-rate 和 inverse-tolerance 必须大于 0，"
                "nearest-neighbors、lbfgs-steps、prior-weight 和 trust-radius 不能小于 0，"
                "co-window-weight 必须大于 0"
            )
        if args.retrieval_mode == "empirical-knn" and args.nearest_neighbors <= 0:
            raise ValueError("empirical-knn 模式要求 nearest-neighbors 大于 0")
    except ValueError as exc:
        raise SystemExit(f"输入错误：{exc}") from exc

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    window_dirs = sorted(
        path
        for path in input_dir.glob("window_*")
        if path.is_dir()
        and (path / "ml_inputs.csv").is_file()
        and (path / "ml_log_flux.csv").is_file()
    )
    if args.retrieval_mode in ("auto", "least-squares"):
        import torch
        first_checkpoint_path = (window_dirs[0] if window_dirs else input_dir) / "surrogate_model.pt"
        if args.model_path is None and first_checkpoint_path.is_file():
            first_checkpoint = torch.load(first_checkpoint_path, map_location="cpu", weights_only=False)
            if first_checkpoint.get("model_type") == "spectral_gp_v4":
                count = resolve_sample_count(args.inversion_samples, len(first_checkpoint["test_sample_ids"]))
                print("V4 inversion backend: CPU float64 analytic-Jacobian least squares", flush=True)
                try:
                    window_weights = parse_window_weights(args.window_weights)
                except ValueError as exc:
                    raise SystemExit(f"输入错误：{exc}") from exc
                run_gp_retrieval(input_dir, output_dir, "test", args.starts, args.max_nfev, "standardized", count, window_weights)
                return
        if args.retrieval_mode == "least-squares":
            raise SystemExit("least-squares requires V4 GP checkpoints at input-dir/surrogate_model.pt")
        args.retrieval_mode = "empirical-knn"
    if window_dirs:
        if args.model_path is not None:
            raise SystemExit(
                "多窗口联合反演会自动加载各窗口的 surrogate_model.pt，不能同时指定 --model-path"
            )
        run_joint_window_inversion(args, input_dir, output_dir, window_dirs)
        return
    _, x_table = read_numeric_csv(input_dir / "ml_inputs.csv")
    flux_header, y_table = read_numeric_csv(input_dir / "ml_log_flux.csv")
    if x_table.shape[1] != len(PARAMETER_NAMES) + 1 or y_table.shape[1] < 2:
        raise SystemExit("输入 CSV 格式不符合当前含 NH3 的 6 参数数据结构")
    if len(x_table) != len(y_table):
        raise SystemExit("ml_inputs.csv 和 ml_log_flux.csv 的样本数不一致")

    sample_ids = x_table[:, 0].astype(int)
    x = x_table[:, 1:].astype(np.float32)
    y = y_table[:, 1:].astype(np.float32)
    if len(x) < 5:
        raise SystemExit("至少需要 5 个正演样本才能进行反演")
    if np.any(x < PARAMETER_BOUNDS[:, 0]) or np.any(x > PARAMETER_BOUNDS[:, 1]):
        raise SystemExit("输入参数超出当前反演参数边界")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("当前 Python 环境没有安装 PyTorch") from exc
    try:
        device = resolve_torch_device(torch, args.device)
    except RuntimeError as exc:
        raise SystemExit(f"GPU 初始化失败：{exc}") from exc
    model_path = (args.model_path or (input_dir / "surrogate_model.pt")).resolve()
    if not model_path.is_file():
        raise SystemExit(
            f"未找到替代模型检查点：{model_path}\n"
            "请先运行 train_surrogate.py 训练并保存 surrogate_model.pt"
        )
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise SystemExit(f"替代模型检查点格式无效：{model_path}")

    input_dim = int(checkpoint.get("input_dim", x.shape[1]))
    output_dim = int(checkpoint.get("output_dim", y.shape[1]))
    if input_dim != x.shape[1] or output_dim != y.shape[1]:
        raise SystemExit(
            "替代模型与当前 CSV 的维度不一致："
            f"checkpoint=({input_dim},{output_dim}), csv=({x.shape[1]},{y.shape[1]})"
        )
    try:
        x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
        x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
        y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
        y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
        train_sample_ids = np.asarray(checkpoint["train_sample_ids"], dtype=np.int64)
        saved_test_sample_ids = np.asarray(checkpoint["test_sample_ids"], dtype=np.int64)
    except KeyError as exc:
        raise SystemExit(
            f"替代模型检查点缺少 {exc.args[0]}；请重新运行 train_surrogate.py"
        ) from exc
    if x_mean.shape != (input_dim,) or x_std.shape != (input_dim,):
        raise SystemExit("检查点中的输入归一化参数维度无效")
    if y_mean.shape != (output_dim,) or y_std.shape != (output_dim,):
        raise SystemExit("检查点中的输出归一化参数维度无效")
    if np.any(x_std <= 0.0) or np.any(y_std <= 0.0):
        raise SystemExit("检查点中的归一化标准差必须为正")

    sample_id_to_row = {int(sample_id): row for row, sample_id in enumerate(sample_ids)}
    if len(sample_id_to_row) != len(sample_ids):
        raise SystemExit("ml_inputs.csv 中存在重复 sample_id，无法匹配模型检查点")
    try:
        train_rows = np.asarray(
            [sample_id_to_row[int(sample_id)] for sample_id in train_sample_ids],
            dtype=int,
        )
        test_rows = np.asarray(
            [sample_id_to_row[int(sample_id)] for sample_id in saved_test_sample_ids],
            dtype=int,
        )
    except KeyError as exc:
        raise SystemExit(
            f"当前 CSV 缺少训练时保存的 sample_id={exc.args[0]}；"
            "请用同一批正演输出重新训练模型"
        ) from exc
    if len(train_rows) == 0 or len(test_rows) == 0:
        raise SystemExit("检查点中的训练集或测试集为空")
    x_train = x[train_rows]
    x_test = x[test_rows]
    y_train = y[train_rows]
    y_test = y[test_rows]
    test_sample_ids = saved_test_sample_ids

    inversion_count = resolve_sample_count(args.inversion_samples, len(x_test))
    selected_indices = np.arange(inversion_count)

    started = time.perf_counter()
    print(
        f"Loaded surrogate checkpoint={model_path} on "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        flush=True,
    )
    validate_checkpoint_data(checkpoint, input_dir)
    model = load_saved_model(checkpoint, device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"无法加载替代模型权重：{model_path}") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"checkpoint_epochs={checkpoint.get('epochs', 'unknown')} "
        f"checkpoint_train_samples={len(train_sample_ids)} "
        f"checkpoint_test_samples={len(test_sample_ids)} "
        f"nearest_neighbors={args.nearest_neighbors} "
        f"lbfgs_steps={args.lbfgs_steps}",
        flush=True,
    )

    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_tensor = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    y_mean_tensor = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_tensor = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    low_tensor = torch.as_tensor(PARAMETER_BOUNDS[:, 0], dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(PARAMETER_BOUNDS[:, 1], dtype=torch.float32, device=device)
    nu_grid = np.asarray(
        [float(field.removeprefix("nu_").removesuffix("_cm-1")) for field in flux_header[1:]],
        dtype=float,
    )

    # Find nearby training spectra in the same standardized spectral space as
    # the inversion loss. This uses no held-out parameter labels and gives the
    # optimizer physically plausible starting points near the observation.
    nearest_parameters_by_test = None
    prior_parameters_by_test = None
    if args.nearest_neighbors > 0:
        y_train_scaled_tensor = torch.as_tensor(
            (y_train - y_mean) / y_std, dtype=torch.float32, device=device
        )
        y_test_scaled_tensor = torch.as_tensor(
            (y_test - y_mean) / y_std, dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            train_norm = torch.sum(y_train_scaled_tensor**2, dim=1)
            test_norm = torch.sum(y_test_scaled_tensor**2, dim=1)
            distances = (
                test_norm[:, None]
                + train_norm[None, :]
                - 2.0 * (y_test_scaled_tensor @ y_train_scaled_tensor.T)
            )
            neighbor_count = min(args.nearest_neighbors, len(x_train))
            nearest_result = torch.topk(
                distances,
                k=neighbor_count,
                largest=False,
                dim=1,
            ).indices.cpu().numpy()
            nearest_distances = np.take_along_axis(
                distances.detach().cpu().numpy(), nearest_result, axis=1
            )
        nearest_parameters_by_test = x_train[nearest_result]
        nearest_weights = 1.0 / np.maximum(nearest_distances, 1.0e-8)
        nearest_weights /= np.sum(nearest_weights, axis=1, keepdims=True)
        prior_parameters_by_test = np.sum(
            nearest_parameters_by_test * nearest_weights[:, :, None], axis=1
        )

    rows: list[list[Any]] = []
    retrieved_parameters = []
    retrieved_spectra = []
    spectral_flux_errors = []
    spectral_brightness_errors = []
    spectral_brightness_relative_errors = []
    for rank, test_index in enumerate(selected_indices):
        inversion_started = time.perf_counter()
        true_parameters = x_test[test_index]
        if args.retrieval_mode == "empirical-knn":
            if prior_parameters_by_test is None:
                raise RuntimeError("empirical-knn 模式没有可用的训练集近邻参数")
            retrieved = prior_parameters_by_test[test_index].astype(np.float32, copy=True)
            with torch.inference_mode():
                retrieved_tensor = torch.as_tensor(retrieved, dtype=torch.float32, device=device)
                retrieved_prediction = model(
                    ((retrieved_tensor - x_mean_tensor) / x_std_tensor)[None, :]
                )[0]
                observed_scaled = (
                    torch.as_tensor(y_test[test_index], dtype=torch.float32, device=device)
                    - y_mean_tensor
                ) / y_std_tensor
                final_loss = float(torch.mean((retrieved_prediction - observed_scaled) ** 2))
            initial_loss = final_loss
            steps_used = 0
            # This mode is a completed non-iterative retrieval, so the
            # iterative optimizer convergence threshold does not apply.
            converged = True
        else:
            retrieved, initial_loss, final_loss, steps_used, converged = optimize_one_sample(
                torch,
                model,
                y_test[test_index],
                x_mean_tensor,
                x_std_tensor,
                y_mean_tensor,
                y_std_tensor,
                low_tensor,
                high_tensor,
                args.starts,
                args.inverse_steps,
                args.inverse_learning_rate,
                args.seed + int(test_sample_ids[test_index]) + rank,
                args.inverse_tolerance,
                None
                if nearest_parameters_by_test is None
                else nearest_parameters_by_test[test_index],
                args.nearest_neighbors,
                args.lbfgs_steps,
                None
                if prior_parameters_by_test is None
                else prior_parameters_by_test[test_index],
                args.prior_weight,
                args.trust_radius,
            )
        with torch.inference_mode():
            retrieved_tensor = torch.as_tensor(retrieved, dtype=torch.float32, device=device)
            predicted_scaled = model(((retrieved_tensor - x_mean_tensor) / x_std_tensor)[None, :])[0]
            predicted_log_flux = (predicted_scaled * y_std_tensor + y_mean_tensor).cpu().numpy()
        true_log_flux = y_test[test_index]
        true_flux = np.power(10.0, true_log_flux.astype(np.float64))
        retrieved_spectra.append(predicted_log_flux.copy())
        predicted_flux = np.power(10.0, predicted_log_flux.astype(np.float64))
        flux_error = predicted_flux - true_flux
        true_bt = brightness_temperature(nu_grid, true_flux[None, :])[0]
        predicted_bt = brightness_temperature(nu_grid, predicted_flux[None, :])[0]
        bt_error = predicted_bt - true_bt
        true_physical = physical_parameters(true_parameters)
        retrieved_physical = physical_parameters(retrieved)
        parameter_error = retrieved_physical - true_physical
        retrieved_parameters.append(retrieved)
        spectral_flux_errors.append(predicted_flux - true_flux)
        spectral_brightness_errors.append(bt_error)
        spectral_brightness_relative_errors.append(relative_error_percent(bt_error, true_bt))
        rows.append(
            [
                int(test_sample_ids[test_index]),
                int(rank),
                *true_physical,
                *retrieved_physical,
                *parameter_error,
                *np.abs(parameter_error),
                float(np.sqrt(np.mean(relative_error_percent(flux_error, true_flux)**2))),
                float(np.mean(np.abs(relative_error_percent(flux_error, true_flux)))),
                float(np.mean(100.0 * np.abs(predicted_flux - true_flux) / np.maximum(true_flux, np.finfo(float).tiny))),
                float(np.sqrt(np.mean(spectral_brightness_relative_errors[-1]**2))),
                initial_loss,
                final_loss,
                steps_used,
                int(converged),
                time.perf_counter() - inversion_started,
            ]
        )
        print(
            f"inversion {rank + 1}/{inversion_count} "
            f"sample={test_sample_ids[test_index]} "
            f"loss={final_loss:.6e}",
            flush=True,
        )

    retrieved_array = np.asarray(retrieved_parameters)
    true_physical_array = physical_parameters(x_test[selected_indices])
    retrieved_physical_array = physical_parameters(retrieved_array)
    parameter_errors = retrieved_physical_array - true_physical_array
    flux_errors = np.asarray(spectral_flux_errors)
    brightness_errors = np.asarray(spectral_brightness_errors)
    brightness_relative_errors = np.asarray(spectral_brightness_relative_errors)
    true_flux_selected = np.power(10.0, y_test[selected_indices].astype(np.float64))
    relative_flux_errors = 100.0 * np.abs(flux_errors) / np.maximum(
        true_flux_selected, np.finfo(float).tiny
    )

    header = [
        "sample_id",
        "test_rank",
        *[f"true_{name}" for name in REPORT_PARAMETER_NAMES],
        *[f"retrieved_{name}" for name in REPORT_PARAMETER_NAMES],
        *[f"signed_error_{name}" for name in REPORT_PARAMETER_NAMES],
        *[f"absolute_error_{name}" for name in REPORT_PARAMETER_NAMES],
        "spectral_linear_flux_rmse_percent",
        "spectral_linear_flux_mae_percent",
        "spectral_linear_flux_mean_relative_error_percent",
        "spectral_brightness_temperature_rmse_percent",
        "best_initial_loss",
        "best_final_loss",
        "optimization_steps",
        "converged",
        "elapsed_seconds",
    ]
    save_csv(output_dir / "inversion_error_by_sample.csv", header, rows)

    parameter_units = ["K", "K"] + ["dimensionless_number_fraction"] * (len(PARAMETER_NAMES) - 2)
    summary_lines = [
        "MLP surrogate inversion error against held-out LBL spectra",
        f"total_forward_samples={len(x)}",
        f"surrogate_train_samples={len(x_train)}",
        f"surrogate_test_samples={len(x_test)}",
        f"inversion_samples={inversion_count}",
        f"retrieval_mode={args.retrieval_mode}",
        f"surrogate_checkpoint={model_path}",
        f"surrogate_checkpoint_epochs={checkpoint.get('epochs', 'unknown')}",
        f"inverse_starts={args.starts}",
        f"inverse_steps={args.inverse_steps}",
        f"inverse_nearest_neighbors={args.nearest_neighbors}",
        f"inverse_lbfgs_steps={args.lbfgs_steps}",
        f"inverse_prior_weight={args.prior_weight:.8e}",
        f"inverse_trust_radius={args.trust_radius:.8e}",
        f"inverse_tolerance={args.inverse_tolerance:.8e}",
        f"compute_backend={'pytorch-cuda' if device.type == 'cuda' else 'pytorch-cpu'}",
        f"compute_device={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}",
        f"inversion_success_rate_percent={100.0 * np.mean(np.isfinite(retrieved_array).all(axis=1)):.6f}",
        f"inversion_converged_rate_percent={100.0 * np.mean(np.asarray([row[-2] for row in rows], dtype=float)):.6f}",
        f"inversion_convergence_definition={'non_iterative_retrieval_completed' if args.retrieval_mode == 'empirical-knn' else 'final_standardized_spectral_mse_below_tolerance'}",
        "error_metric_space=physical_linear",
        "mae_rmse_unit=percent",
        "mae_rmse_definition=pointwise_error_divided_by_absolute_truth_times_100",
        "optimization_loss_space=standardized_log10_flux",
        "spectral_linear_flux_unit=W m^-2 (cm^-1)^-1",
        f"spectral_linear_flux_rmse_percent={np.sqrt(np.mean(relative_flux_errors**2)):.8e}",
        f"spectral_linear_flux_mae_percent={np.mean(relative_flux_errors):.8e}",
        f"spectral_linear_flux_mean_relative_error_percent={np.mean(relative_flux_errors):.8e}",
        f"spectral_linear_flux_median_relative_error_percent={np.median(relative_flux_errors):.8e}",
        f"spectral_linear_flux_p95_relative_error_percent={np.percentile(relative_flux_errors, 95.0):.8e}",
        f"spectral_brightness_temperature_rmse_percent={np.sqrt(np.mean(brightness_relative_errors**2)):.8e}",
        f"spectral_brightness_temperature_mae_percent={np.mean(np.abs(brightness_relative_errors)):.8e}",
    ]
    for index, (name, unit) in enumerate(zip(REPORT_PARAMETER_NAMES, parameter_units)):
        metrics = parameter_metrics(parameter_errors[:, index])
        percent_metrics = parameter_metrics(relative_error_percent(
            parameter_errors[:, index], true_physical_array[:, index]
        ))
        summary_lines.extend(
            [
                f"parameter_{name}_unit={unit}",
                f"parameter_{name}_bias={metrics['bias']:.8e}",
                f"parameter_{name}_mae_percent={percent_metrics['mae']:.8e}",
                f"parameter_{name}_rmse_percent={percent_metrics['rmse']:.8e}",
                f"parameter_{name}_median_abs={metrics['median_abs']:.8e}",
                f"parameter_{name}_p95_abs={metrics['p95_abs']:.8e}",
                f"parameter_{name}_max_abs={metrics['max_abs']:.8e}",
            ]
        )
    summary_lines.append(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    (output_dir / "inversion_error_report.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    save_inversion_spectra(
        output_dir, test_sample_ids[selected_indices], [input_dir.name], [nu_grid],
        y_test[selected_indices], np.asarray(retrieved_spectra),
    )

    print("\n".join(summary_lines), flush=True)

if __name__ == "__main__":
    main()
    print("数据已保存；运行 visualize_results.py 生成图像。", flush=True)

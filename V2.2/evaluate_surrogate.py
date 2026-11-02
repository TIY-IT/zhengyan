"""V4 evaluation of already-frozen models; no training or model selection.

Self-contained entrypoint: no local helper modules are required.
"""
from __future__ import annotations

import argparse
import json

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

def load_model(checkpoint, device='cpu'):
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

# 误差计算与报告
from pathlib import Path

import json

import numpy as np




def brightness_temperature(nu, flux):
    n = nu[None, :] * 100
    return (6.62607015e-34 * 299792458. * n / 1.380649e-23) / np.log1p(2 * 6.62607015e-34 * 299792458.**2 * n**3 * np.pi * 100 / flux)

def write_evaluation(directory, ids, nu, truth, prediction, *, split='test', baseline=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metric = error_metrics(truth, prediction)
    flux, predicted_flux = 10**truth, 10**prediction
    log_abs = np.abs(prediction - truth)
    flux_abs = np.abs(predicted_flux - flux)
    relative = flux_abs / flux * 100
    tb, tbp = brightness_temperature(nu, flux), brightness_temperature(nu, predicted_flux)
    te = tbp - tb
    metric.update(samples=len(ids), channels=len(nu), linear_flux_mae=float(flux_abs.mean()),
                  linear_flux_rmse=float(np.sqrt(np.mean(flux_abs**2))),
                  brightness_temperature_mae_K=float(np.abs(te).mean()),
                  brightness_temperature_rmse_K=float(np.sqrt(np.mean(te**2))))
    (directory / f'{split}_metrics.json').write_text(json.dumps(metric, indent=2)+'\n', encoding='utf-8')
    per_sample = np.column_stack([ids, np.sqrt(np.mean(log_abs**2, 1)), log_abs.mean(1),
                                 log_abs.max(1), np.sqrt(np.mean(flux_abs**2, 1)), flux_abs.mean(1),
                                 flux_abs.max(1), relative.mean(1), relative.max(1),
                                 np.sqrt(np.mean(te**2, 1)), np.abs(te).mean(1), np.abs(te).max(1)])
    sample_header='sample_id,log10_flux_rmse,log10_flux_mae,log10_flux_max_absolute_error,linear_flux_rmse,linear_flux_mae,linear_flux_max_absolute_error,linear_flux_mean_relative_error_percent,linear_flux_max_relative_error_percent,brightness_temperature_rmse_K,brightness_temperature_mae_K,brightness_temperature_max_absolute_error_K'
    prefix = 'surrogate' if split == 'test' else split
    np.savetxt(directory / f'{prefix}_error_by_sample.csv', per_sample, delimiter=',', header=sample_header, comments='', fmt='%.12g')
    channel = np.column_stack([nu, np.sqrt(np.mean(log_abs**2, 0)), log_abs.mean(0),
                               np.sqrt(np.mean(flux_abs**2, 0)), flux_abs.mean(0), relative.mean(0),
                               relative.max(0), np.sqrt(np.mean(te**2, 0)), np.abs(te).mean(0), np.abs(te).max(0)])
    np.savetxt(directory / f'{prefix}_error_by_wavenumber.csv', channel, delimiter=',',
               header='wavenumber_cm-1,log10_flux_rmse,log10_flux_mae,linear_flux_rmse,linear_flux_mae,mean_linear_flux_relative_error_percent,max_linear_flux_relative_error_percent,brightness_temperature_rmse_K,brightness_temperature_mae_K,brightness_temperature_max_absolute_error_K', comments='', fmt='%.12g')
    if split == 'test':
        points = np.column_stack([np.repeat(np.arange(len(ids)),len(nu)), np.repeat(ids,len(nu)), np.tile(nu,len(ids)),
                                  truth.ravel(),prediction.ravel(),log_abs.ravel(),flux.ravel(),predicted_flux.ravel(),
                                  flux_abs.ravel(),relative.ravel(),tb.ravel(),tbp.ravel(),np.abs(te).ravel()])
        np.savetxt(directory / 'surrogate_predictions.csv', points, delimiter=',',
                   header='test_rank,sample_id,wavenumber_cm-1,true_log10_flux,predicted_log10_flux,absolute_log10_flux_error,true_linear_flux,predicted_linear_flux,absolute_linear_flux_error,linear_flux_relative_error_percent,true_brightness_temperature_K,predicted_brightness_temperature_K,brightness_temperature_absolute_error_K', comments='', fmt='%.12g')
        lines=['Frozen V4 surrogate: held-out six-parameter atmospheres',
               f'surrogate_error_test_samples={len(ids)}', f'surrogate_error_spectral_channels={len(nu)}',
               f'surrogate_error_log10_flux_rmse={metric["log10_rmse"]:.10e}',
               f'surrogate_error_log10_flux_mae={metric["log10_mae"]:.10e}',
               f'surrogate_error_linear_flux_mean_rel_percent={metric["flux_mape_percent"]:.10e}',
               f'surrogate_error_linear_flux_rms_relative_percent={metric["flux_rmsre_percent"]:.10e}',
               f'surrogate_error_linear_flux_p95_rel_percent={metric["flux_p95_percent"]:.10e}',
               f'surrogate_error_linear_flux_max_rel_percent={metric["flux_max_percent"]:.10e}',
               f'surrogate_error_brightness_temperature_rmse_K={metric["brightness_temperature_rmse_K"]:.10e}',
               'scope=Errors against the existing synthetic forward solver; not experimental or independent-physics validation.',
               'test_used_for_training=false', 'test_used_for_hyperparameter_selection=false', 'validation_refitted=false']
        (directory/'surrogate_error_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        baseline_path = directory / 'surrogate_baseline_plot_data.npz'
        if baseline is not None:
            np.savez_compressed(
                baseline_path, sample_ids=ids, wavenumber=nu,
                true_log_flux=truth, baseline_log_flux=baseline,
                predictions_sha256=np.array(file_hash(directory/'surrogate_predictions.csv')),
            )
        elif baseline_path.exists():
            baseline_path.unlink()
    return metric

# 冻结模型评估
def evaluate_frozen(output,device='cpu',baseline_root=None):
    """Evaluation only: never updates, refits, or selects a model."""
    output=Path(output)
    frozen=json.loads((output/'frozen_selection.json').read_text(encoding='utf-8'))
    comparison=[]
    for item in frozen['models']:
        directory=Path(item['directory']); ckpath=directory/'surrogate_model.pt'
        if file_hash(ckpath)!=item['checkpoint_sha256']: raise ValueError('Model changed after selection freeze')
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        validate_checkpoint_data(ck,directory)
        ids,x,y,nu=read_dataset(directory)
        split={k:ck[k+'_sample_ids'].tolist() for k in ['train','validation','test']}
        te=check_split(ids,split)['test']
        prediction=load_model(ck,device).predict(x[te])
        base_prediction=None; bm=None
        if baseline_root:
            basepath=Path(baseline_root)/item['window']
            bc=torch.load(basepath/'surrogate_model.pt',map_location='cpu',weights_only=False)
            bi,bx,by,bnu=read_dataset(basepath)
            if not np.array_equal(ids,bi) or not np.array_equal(x,bx) or not np.array_equal(y,by) or not np.array_equal(nu,bnu):
                raise ValueError('Baseline comparison requires identical data')
            if not np.array_equal(ck['test_sample_ids'],bc['test_sample_ids']):
                raise ValueError('Baseline test IDs differ')
            base=load_model(bc,device)
            # V2 trained/predicted with float64 normalization then float32 tensors.
            br={int(i):r for r,i in enumerate(bi)}
            btr=np.array([br[int(i)] for i in bc['train_sample_ids']])
            xm,xs=bx[btr].mean(0),bx[btr].std(0)
            ym,ys=by[btr].mean(0),by[btr].std(0)
            with torch.inference_mode():
                scaled=torch.tensor((x[te]-xm)/xs,dtype=torch.float32,device=device)
                base_prediction=base(scaled).cpu().numpy()*ys+ym
            bm=error_metrics(y[te],base_prediction)
        metric=write_evaluation(directory,ids[te],nu,y[te],prediction,baseline=base_prediction)
        report_path=directory/'surrogate_report.txt'
        if report_path.exists():
            training_text=report_path.read_text(encoding='utf-8').split('\nFINAL_FROZEN_TEST\n')[0]
            report_path.write_text(training_text.rstrip()+'\n\nFINAL_FROZEN_TEST\n'+
                                  '\n'.join(f'{k}={v}' for k,v in metric.items())+'\n',encoding='utf-8')
        row=dict(window=item['window'],v4=metric,baseline=bm)
        if bm:
            row['log10_rmse_reduction_percent']=100*(1-metric['log10_rmse']/bm['log10_rmse'])
            row['flux_mape_reduction_percent']=100*(1-metric['flux_mape_percent']/bm['flux_mape_percent'])
        comparison.append(row); print('FINAL_TEST',json.dumps(row),flush=True)
    (output/'optimization_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
    return comparison

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--baseline-dir',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    args=parser.parse_args()
    if args.device=='gpu' and not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    device='cuda' if args.device!='cpu' and torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    evaluate_frozen(args.input_dir,device,args.baseline_dir)

if __name__ == "__main__":
    main()
    print("数据已保存；运行 visualize_results.py 生成图像。", flush=True)

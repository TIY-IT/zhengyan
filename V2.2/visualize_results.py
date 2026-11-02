"""V4 独立绘图入口：读取已保存结果，标记绝对相对误差严格大于阈值的点。

直接运行即可处理 outputs，默认阈值为 5%，图像写入 outputs/visualizations。
不训练模型、不执行参数反演、不修改输入数据或正式误差报告。
旧版未保存反演光谱时，使用匹配的冻结 GP 和已保存反演参数补算光谱。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

NAMES = ['T_top_K', 'T_bottom_K', 'X_H2O', 'X_CO', 'X_CH4', 'X_NH3']
LABELS = ['Top temperature', 'Bottom temperature', 'H$_2$O', 'CO', 'CH$_4$', 'NH$_3$']
BLUE, RED, GREY = '#2563a6', '#c43c39', '#687582'
FLUX_LABEL = r'Flux [W m$^{-2}$ (cm$^{-1}$)$^{-1}$]'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_error(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Truth/prediction shapes or finite values are invalid')
    if np.any(truth == 0):
        raise ValueError('Relative error is undefined for zero truth')
    return (prediction - truth) / np.abs(truth) * 100


def over_threshold(error, threshold):
    """Both signs count; a value equal to the threshold does not count."""
    return np.abs(error) > threshold


def integer_ids(values):
    result = np.asarray(values, dtype=np.int64)
    if not np.array_equal(result, values):
        raise ValueError('Sample IDs must be integers')
    return result


def write_csv(path, header, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class Visualizer:
    def __init__(self, input_dir, output_dir, threshold=5.0, dpi=180, model_dir=None):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.model_dir = Path(model_dir or input_dir).resolve()
        self.threshold = threshold
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sources = {}
        self.figures = []
        self.notes = []
        self.spectral_flags = []
        self.summary = {'threshold_percent': threshold, 'comparison': 'absolute_relative_error > threshold',
                        'model_training_performed': False, 'parameter_retrieval_performed': False}

    def record_source(self, path):
        path = Path(path).resolve()
        self.sources[str(path)] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}

    def read_table(self, path):
        self.record_source(path)
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            values = np.asarray([[float(v) for v in row] for row in reader], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(header) or not np.isfinite(values).all():
            raise ValueError(f'Empty, malformed or nonfinite CSV: {path}')
        return dict(zip(header, values.T))

    def save(self, fig, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=self.dpi, facecolor='white')
        plt.close(fig)
        self.figures.append(str(destination.relative_to(self.output_dir)))

    def note(self, message):
        self.notes.append(message)
        print(message, flush=True)

    def forward(self, directory, destination):
        """Original plots 01--06, reconstructed from saved numerical data."""
        destination.mkdir(parents=True, exist_ok=True)
        archive = directory / 'line_plot_data.npz'
        if archive.exists():
            self.record_source(archive)
            with np.load(archive, allow_pickle=False) as data:
                fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
                for ax, species in zip(axes, ['H2O', 'CO', 'CH4', 'NH3']):
                    nu, strength = data[f'{species}_nu'], data[f'{species}_strength']
                    if np.any(strength <= 0) or not np.isfinite(strength).all():
                        raise ValueError('Invalid line strengths')
                    ax.scatter(nu, np.log10(strength), s=3, alpha=.5, color=BLUE, linewidths=0, rasterized=True)
                    ax.set_ylabel(f'{species}\nlog10 S (296 K)')
                    ax.grid(alpha=.15)
                axes[-1].set_xlabel(r'Wavenumber [cm$^{-1}$]')
                fig.suptitle(f'{directory.name}: line list')
                self.save(fig, destination/'01_hitran_line_list.png')
        else:
            # Old runs retained only the image and summary statistics, not the
            # individual line strengths. Preserve that image without inventing data.
            original = directory/'01_hitran_line_list.png'
            if original.exists():
                self.record_source(original)
                target = destination/original.name
                if target.resolve() != original.resolve():
                    shutil.copy2(original, target)
                self.figures.append(str(target.relative_to(self.output_dir)))
                self.note(f'{directory.name}: reused original line-list image (legacy run has no line_plot_data.npz).')
            else:
                self.note(f'{directory.name}: no line-list data; skipped figure 01.')

        profile_path = directory/'atmosphere_profile.csv'
        profile = self.read_table(profile_path) if profile_path.exists() else None
        if profile is not None:
            fig, ax = plt.subplots(figsize=(6.5, 6.5), layout='constrained')
            ax.plot(profile['temperature_K'], profile['p_mid_Pa'], color=BLUE, lw=2)
            ax.set(yscale='log', xlabel='Temperature [K]', ylabel='Pressure [Pa]', title='Atmospheric profile')
            ax.invert_yaxis()
            ax.grid(alpha=.2, which='both')
            self.save(fig, destination/'02_temperature_pressure.png')

        opacity_path = directory/'example_opacity_all_layers.csv'
        if opacity_path.exists() and profile is not None:
            data = self.read_table(opacity_path)
            nu = data['wavenumber_cm-1']
            selected = np.unique([np.argmin(np.abs(profile['p_mid_Pa']-p)) for p in [3.3, 59., 2.7e3, 7.9e4]])
            fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
            for layer in selected:
                ax.semilogy(nu, np.maximum(data[f'kappa_layer_{layer:02d}_m2_kg-1'], 1e-30), lw=1,
                            label=f"p={profile['p_mid_Pa'][layer]:.2g} Pa, T={profile['temperature_K'][layer]:.0f} K")
            ax.set(xlabel=r'Wavenumber [cm$^{-1}$]', ylabel=r'Mass opacity [m$^2$ kg$^{-1}$]', title='Opacity at selected layers')
            ax.legend(fontsize=9)
            ax.grid(alpha=.2)
            self.save(fig, destination/'03_opacity_selected_layers.png')

        spectrum_path = directory/'example_spectrum.csv'
        if spectrum_path.exists():
            data = self.read_table(spectrum_path)
            nu = data['wavenumber_cm-1']
            fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
            ax.plot(nu, data['outgoing_flux_W_m-2_per_cm-1'], label='Two-stream outgoing flux', color=BLUE)
            n = nu*100
            for temperature in [550., 1000., 1450.]:
                blackbody = np.pi*2*6.62607015e-34*299792458.**2*n**3*100/np.expm1(6.62607015e-34*299792458.*n/(1.380649e-23*temperature))
                ax.plot(nu, blackbody, lw=1, label=f'Blackbody {temperature:.0f} K')
            ax.set(xlabel=r'Wavenumber [cm$^{-1}$]', ylabel=FLUX_LABEL, title='LBL + two-stream emission spectrum')
            ax.legend(fontsize=9)
            ax.grid(alpha=.2)
            self.save(fig, destination/'04_emission_spectrum.png')
            fig, ax = plt.subplots(figsize=(11, 5.2), layout='constrained')
            ax.plot(nu, data['brightness_temperature_K'], color=BLUE)
            ax.set(xlabel=r'Wavenumber [cm$^{-1}$]', ylabel='Brightness temperature [K]', title='Brightness temperature')
            ax.grid(alpha=.2)
            self.save(fig, destination/'05_brightness_temperature.png')

        ml_path = directory/'ml_log_flux.csv'
        if ml_path.exists():
            data = self.read_table(ml_path)
            ids = integer_ids(data['sample'])
            columns = [c for c in data if c.startswith('nu_')]
            nu = np.array([float(c.removeprefix('nu_').removesuffix('_cm-1')) for c in columns])
            selected = np.unique(np.linspace(0, len(ids)-1, min(5, len(ids)), dtype=int))
            fig, ax = plt.subplots(figsize=(11, 5.8), layout='constrained')
            for index in selected:
                ax.plot(nu, [data[c][index] for c in columns], lw=1, label=f'Sample {ids[index]}')
            ax.set(xlabel=r'Wavenumber [cm$^{-1}$]', ylabel='log10 outgoing flux',
                   title=f'Synthetic spectra: {len(selected)} evenly spaced examples of {len(ids)} samples')
            ax.legend(fontsize=9)
            ax.grid(alpha=.2)
            self.save(fig, destination/'06_ml_synthetic_spectra.png')

    def surrogate(self, directory, destination):
        source = directory/'surrogate_predictions.csv'
        if not source.exists():
            self.note(f'{directory.name}: no surrogate_predictions.csv; skipped surrogate figures.')
            return
        data = self.read_table(source)
        ids = integer_ids(data['sample_id'])
        nu = data['wavenumber_cm-1']
        signed = relative_error(data['true_linear_flux'], data['predicted_linear_flux'])
        error = np.abs(signed)
        stored = data['linear_flux_relative_error_percent']
        if not np.allclose(error, stored, atol=1e-7, rtol=1e-7):
            raise ValueError(f'{source}: stored percentage disagrees with flux values')
        # Preserve the saved higher-precision error for threshold decisions.
        error = stored
        mask = over_threshold(error, self.threshold)
        if len(np.unique(np.column_stack([ids, nu]), axis=0)) != len(ids):
            raise ValueError('Duplicate sample/wavenumber pair in predictions')
        for i in np.flatnonzero(mask):
            self.spectral_flags.append([directory.name, int(ids[i]), float(nu[i]),
                                       float(data['true_linear_flux'][i]), float(data['predicted_linear_flux'][i]), float(error[i])])
        unique_ids = np.unique(ids)
        rmse = np.array([np.sqrt(np.mean((data['predicted_log10_flux'][ids == sid]-data['true_log10_flux'][ids == sid])**2)) for sid in unique_ids])
        examples = [int(ids[0]), int(unique_ids[np.argmax(rmse)])]
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
        for ax, sid, label in zip(axes, examples, ['First saved test sample', 'Largest log10 spectral RMSE']):
            rows = np.flatnonzero(ids == sid)
            rows = rows[np.argsort(nu[rows])]
            flagged = rows[mask[rows]]
            ax.plot(nu[rows], data['true_log10_flux'][rows], lw=1.2, color=GREY, label='Forward solver')
            ax.plot(nu[rows], data['predicted_log10_flux'][rows], lw=1, ls='--', color=BLUE, label='Surrogate')
            ax.scatter(nu[flagged], data['predicted_log10_flux'][flagged], s=40, facecolors='none', edgecolors=RED,
                       linewidths=1.4, zorder=5, label=f'Flux error > {self.threshold:g}%: {len(flagged)} points')
            ax.set(title=f'{label}: sample {sid}', ylabel='log10 flux', xlabel=r'Wavenumber [cm$^{-1}$]')
            ax.legend(fontsize=9)
            ax.grid(alpha=.18)
        self.save(fig, destination/'07_surrogate_prediction.png')

        grid = np.unique(nu)
        mean_error = np.array([error[nu == x].mean() for x in grid])
        max_error = np.array([error[nu == x].max() for x in grid])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
        axes[0].plot(grid, mean_error, color=BLUE, label='Mean across samples')
        axes[0].plot(grid, max_error, color=GREY, lw=.9, alpha=.7, label='Maximum across samples')
        axes[0].scatter(nu[mask], error[mask], color=RED, s=24, zorder=4, label=f'All {mask.sum()} points > {self.threshold:g}%')
        axes[0].axhline(self.threshold, color=RED, ls='--', lw=1)
        axes[0].set(xlabel=r'Wavenumber [cm$^{-1}$]', ylabel='Absolute flux relative error [%]')
        axes[1].plot(np.maximum(np.sort(error), 1e-12), np.arange(1,len(error)+1)/len(error)*100, color=BLUE, label='Surrogate')
        axes[1].axvline(self.threshold, color=RED, ls='--', label=f'{self.threshold:g}% threshold')
        axes[1].set(xscale='log', xlabel='Absolute flux relative error [%]', ylabel='Cumulative spectral points [%]')
        baseline = directory/'surrogate_baseline_plot_data.npz'
        if baseline.exists():
            self.record_source(baseline)
            with np.load(baseline, allow_pickle=False) as b:
                if str(b['predictions_sha256'].item()) != file_hash(source):
                    raise ValueError('Baseline plot data belongs to different predictions; rerun evaluation')
                be = np.abs(np.expm1(np.log(10)*(b['baseline_log_flux']-b['true_log_flux'])))*100
                axes[0].plot(b['wavenumber'], be.mean(0), color='#aa8c4f', label='V2 mean')
                axes[1].plot(np.maximum(np.sort(be.ravel()),1e-12), np.arange(1,be.size+1)/be.size*100, color='#aa8c4f', label='V2')
        for ax in axes:
            ax.grid(alpha=.18)
            ax.legend(fontsize=9)
        fig.suptitle(f'{directory.name}: {mask.sum()} / {len(error)} spectral points exceed {self.threshold:g}%')
        self.save(fig, destination/'08_surrogate_error.png')
        self.summary.setdefault('surrogate_windows', {})[directory.name] = {
            'total_points': len(error), 'flagged_points': int(mask.sum()),
            'flagged_sample_ids': np.unique(ids[mask]).tolist(),
        }

    def spectral_flag_figure(self):
        rows = sorted(self.spectral_flags, key=lambda r: (r[0], r[1], r[2]))
        write_csv(self.output_dir/'spectral_points_over_threshold.csv',
                  ['window','sample_id','wavenumber_cm-1','true_linear_flux','predicted_linear_flux','absolute_relative_error_percent'], rows)
        # Paginate instead of dropping observations when another run has many flags.
        pages = max(1, (len(rows)+39)//40)
        for page in range(pages):
            chunk = rows[page*40:(page+1)*40]
            fig, ax = plt.subplots(figsize=(11, max(3, .27*len(chunk)+1.8)), layout='constrained')
            if chunk:
                positions = np.arange(len(chunk))
                values = np.array([r[-1] for r in chunk])
                ax.barh(positions, values, height=.65, color=RED, alpha=.85)
                ax.set_yticks(positions, [f'Sample {r[1]} | {r[2]:g} cm$^{{-1}}$' for r in chunk], fontsize=9)
                for y, value in zip(positions, values):
                    ax.text(value+.12, y, f'{value:.2f}%', va='center', fontsize=9)
                ax.set_xlim(0, max(values.max()*1.16, self.threshold*1.3))
                ax.invert_yaxis()
            else:
                ax.text(.5,.5,'No spectral points exceed the threshold', transform=ax.transAxes, ha='center')
                ax.set_yticks([])
            ax.axvline(self.threshold, color=GREY, ls='--', lw=1)
            ax.set(xlabel='Absolute linear-flux relative error [%]',
                   title=f'All spectral points > {self.threshold:g}%: {len(rows)} points (page {page+1}/{pages})')
            ax.grid(axis='x',alpha=.15)
            suffix = '' if pages == 1 else f'_{page+1:02d}'
            self.save(fig, self.output_dir/f'12_spectral_points_over_threshold{suffix}.png')
        self.summary['spectral_flagged_points'] = len(rows)
        self.summary['spectral_flagged_sample_ids'] = sorted({r[1] for r in rows})

    def load_inversion(self):
        path = self.input_dir/'inversion_error_by_sample.csv'
        data = self.read_table(path)
        ids = integer_ids(data['sample_id'])
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate inversion sample IDs')
        truth = np.column_stack([data['true_'+name] for name in NAMES])
        prediction = np.column_stack([data['retrieved_'+name] for name in NAMES])
        error = relative_error(truth, prediction)
        for j,name in enumerate(NAMES):
            if name+'_error_percent' in data:
                stored = data[name+'_error_percent']
                if not np.allclose(stored, error[:,j], atol=1e-7, rtol=1e-7):
                    raise ValueError(f'{name}: saved percentage does not match parameters')
                error[:,j] = stored
        return ids, truth, prediction, error

    def inversion(self):
        path = self.input_dir/'inversion_error_by_sample.csv'
        if not path.exists():
            self.note('No inversion_error_by_sample.csv; skipped inversion figures.')
            return
        ids, truth, prediction, error = self.load_inversion()
        mask = over_threshold(error, self.threshold)
        self.summary['inversion'] = {'samples': len(ids),
            'flagged_samples_by_parameter': {name:ids[mask[:,j]].tolist() for j,name in enumerate(NAMES)},
            'gas_flagged_sample_ids': sorted(ids[mask[:,2:].any(1)].tolist())}
        fig, axes = plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
        for j,ax in enumerate(axes.ravel()[:len(NAMES)]):
            lo,hi = min(truth[:,j].min(),prediction[:,j].min()), max(truth[:,j].max(),prediction[:,j].max())
            ax.plot([lo,hi],[lo,hi],color=GREY,lw=1,label='Exact recovery')
            ax.plot([lo,hi],np.array([lo,hi])*(1+self.threshold/100),color=GREY,lw=.8,ls=':')
            if self.threshold < 100:
                ax.plot([lo,hi],np.array([lo,hi])*(1-self.threshold/100),color=GREY,lw=.8,ls=':')
            ax.scatter(truth[~mask[:,j],j],prediction[~mask[:,j],j],s=16,alpha=.65,color=BLUE)
            ax.scatter(truth[mask[:,j],j],prediction[mask[:,j],j],s=45,facecolors='none',edgecolors=RED,linewidths=1.5,zorder=4)
            # IDs for every flagged sample are printed without overlap in figures 13/14.
            worst = int(np.argmax(np.abs(error[:,j])))
            if mask[worst,j]:
                ax.annotate(f'ID {ids[worst]}', (truth[worst,j],prediction[worst,j]), xytext=(5,10), textcoords='offset points', color=RED, fontsize=9)
            if j>=2:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set(title=f'{LABELS[j]}: {mask[:,j].sum()}/{len(ids)} exceed {self.threshold:g}%',
                   xlabel='True value'+(' [K]' if j<2 else ' [number fraction]'),
                   ylabel='Retrieved value'+(' [K]' if j<2 else ' [number fraction]'))
            ax.grid(alpha=.15)
        fig.suptitle(f'Frozen-model inversion: {len(ids)} samples; red circles: |error| > {self.threshold:g}%')
        self.save(fig,self.output_dir/'09_inversion_parameter_parity.png')

        fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
        ax.boxplot([np.abs(error[:,j]) for j in range(len(NAMES))],tick_labels=LABELS,showfliers=False)
        # Deterministic jitter displays all observations; no random data added.
        jitter = np.linspace(-.12,.12,len(ids))
        for j in range(len(NAMES)):
            ax.scatter(j+1+jitter, np.abs(error[:,j]), s=14, c=np.where(mask[:,j],RED,BLUE), alpha=.7, zorder=3)
        ax.axhline(self.threshold,color=RED,ls='--',label=f'{self.threshold:g}% threshold')
        ax.set_yscale('symlog', linthresh=.01)
        ax.set_ylim(0, max(float(np.abs(error).max()), self.threshold)*1.4)
        ax.set(ylabel='Absolute parameter relative error [%]',title='Parameter errors: every test sample shown')
        ax.legend()
        ax.grid(axis='y',alpha=.2)
        self.save(fig,self.output_dir/'10_inversion_parameter_error.png')
        self.gas_flags(ids,truth,prediction,error,mask)
        self.temperature(ids,truth,prediction,error,mask)
        self.inversion_spectra(ids,truth,prediction)

    def gas_flags(self,ids,truth,prediction,error,mask):
        selected = np.flatnonzero(mask[:,2:].any(1))
        selected = selected[np.argsort(ids[selected])]
        rows = [[int(ids[i]),NAMES[j],float(truth[i,j]),float(prediction[i,j]),float(error[i,j]),float(abs(error[i,j]))]
                for i in np.argsort(ids) for j in range(len(NAMES)) if mask[i,j]]
        write_csv(self.output_dir/'parameters_over_threshold.csv',
                  ['sample_id','parameter','true_value','retrieved_value','signed_error_percent','absolute_error_percent'], rows)
        pages = max(1,(len(selected)+34)//35)
        for page in range(pages):
            chunk=selected[page*35:(page+1)*35]
            fig,ax=plt.subplots(figsize=(8,max(3,.32*len(chunk)+1.8)),layout='constrained')
            if len(chunk):
                flags=mask[chunk,2:]
                ax.imshow(flags.astype(float),cmap=ListedColormap(['#f0f2f4','#f7d8d5']),vmin=0,vmax=1,aspect='auto')
                ax.set_yticks(np.arange(len(chunk)),[f'Sample {sid}' for sid in ids[chunk]])
                ax.set_xticks(range(len(NAMES)-2),LABELS[2:])
                for row,i in enumerate(chunk):
                    for col,j in enumerate(range(2,len(NAMES))):
                        ax.text(col,row,f'{error[i,j]:+.2f}%',ha='center',va='center',fontsize=10,
                                color=RED if mask[i,j] else GREY,fontweight='bold' if mask[i,j] else 'normal')
                ax.tick_params(length=0)
            else:
                ax.text(.5,.5,'No gas samples exceed the threshold',transform=ax.transAxes,ha='center')
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f'Gas abundance errors: {len(selected)}/{len(ids)} flagged samples\nRed cells: |error| > {self.threshold:g}%; + overestimate, - underestimate',pad=16)
            suffix='' if pages==1 else f'_{page+1:02d}'
            self.save(fig,self.output_dir/f'13_gas_samples_over_threshold{suffix}.png')

    def temperature(self,ids,truth,prediction,error,mask):
        delta=prediction[:,:2]-truth[:,:2]
        fig,axes=plt.subplots(2,1,figsize=(12,7.5),sharex=True,layout='constrained')
        statistics={}
        for j,ax in enumerate(axes):
            mae=float(np.abs(delta[:,j]).mean())
            rmse=float(np.sqrt(np.mean(delta[:,j]**2)))
            statistics[NAMES[j]]={'mae_K':mae,'rmse_K':rmse,'mae_percent':float(np.abs(error[:,j]).mean()),
                                  'rmse_percent':float(np.sqrt(np.mean(error[:,j]**2))),
                                  'flagged_sample_ids':ids[mask[:,j]].tolist()}
            ax.scatter(ids[~mask[:,j]],delta[~mask[:,j],j],s=22,color=BLUE,alpha=.7)
            ax.scatter(ids[mask[:,j]],delta[mask[:,j],j],s=55,color=RED,zorder=4)
            for i in np.flatnonzero(mask[:,j]):
                ax.annotate(f'ID {ids[i]}: {delta[i,j]:+.2f} K ({error[i,j]:+.2f}%)',
                            (ids[i],delta[i,j]),xytext=(8,12),textcoords='offset points',fontsize=9,color=RED)
            ax.axhline(0,color=GREY,lw=.9)
            ax.set(ylabel='Retrieved - true [K]',
                   title=f'{LABELS[j]}: MAE {mae:.4f} K, RMSE {rmse:.4f} K | {mask[:,j].sum()} samples exceed {self.threshold:g}%')
            ax.margins(y=.25,x=.06)
            ax.grid(alpha=.18)
        axes[-1].set_xlabel('Original sample ID')
        self.save(fig,self.output_dir/'14_temperature_errors.png')
        self.summary['temperature']=statistics
        write_csv(self.output_dir/'temperature_errors.csv',
                  ['sample_id','true_T_top_K','retrieved_T_top_K','T_top_error_K','T_top_error_percent',
                   'true_T_bottom_K','retrieved_T_bottom_K','T_bottom_error_K','T_bottom_error_percent'],
                  [[int(ids[i]),truth[i,0],prediction[i,0],delta[i,0],error[i,0],truth[i,1],prediction[i,1],delta[i,1],error[i,1]] for i in np.argsort(ids)])

    def inversion_spectra(self,ids,truth,parameters):
        archive=self.input_dir/'inversion_spectra.npz'
        if archive.exists():
            self.record_source(archive)
            with np.load(archive,allow_pickle=False) as data:
                if str(data['parameter_csv_sha256'].item()) != file_hash(self.input_dir/'inversion_error_by_sample.csv'):
                    raise ValueError('inversion_spectra.npz belongs to different parameter results')
                if not np.array_equal(data['sample_ids'],ids):
                    raise ValueError('Inversion spectra and parameters have different sample IDs')
                names=data['window_names'].copy()
                sizes=data['window_sizes'].copy()
                grid=data['wavenumber'].copy()
                observed=data['true_log_flux'].copy()
                predicted=data['retrieved_log_flux'].copy()
            mode='saved_inversion_spectra'
        else:
            reconstructed=self.reconstruct_legacy_gp_spectra(ids,truth,parameters)
            if reconstructed is None:
                return
            names,sizes,grid,observed,predicted=reconstructed
            mode='frozen_gp_prediction_at_saved_parameters'
        if observed.shape!=predicted.shape or observed.shape!=(len(ids),len(grid)) or sum(sizes)!=len(grid) or len(names)!=len(sizes):
            raise ValueError('Invalid inversion spectral array shapes')
        if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise ValueError('Nonfinite inversion spectra')
        error=np.abs(np.expm1(np.log(10)*(predicted-observed)))*100
        # Show the largest spectral RMSRE, explicitly labelled; every point and
        # sample still contributes to the reports, parameter plots and counts.
        worst=int(np.argmax(np.sqrt(np.mean(error**2,axis=1))))
        fig,axes=plt.subplots(len(names),2,figsize=(14,3.3*len(names)),squeeze=False,layout='constrained')
        offset=0
        point_rows=[]
        for index,(name,size) in enumerate(zip(names,sizes)):
            sl=slice(offset,offset+int(size))
            nu=grid[sl]
            values=error[worst,sl]
            flagged=over_threshold(values,self.threshold)
            ax,err_ax=axes[index]
            ax.plot(nu,observed[worst,sl],color=GREY,lw=1,label='Observed forward spectrum')
            ax.plot(nu,predicted[worst,sl],color=BLUE,lw=1,ls='--',label='Retrieved spectrum')
            ax.scatter(nu[flagged],predicted[worst,sl][flagged],s=35,facecolors='none',edgecolors=RED,zorder=4)
            ax.set(title=str(name),ylabel='log10 flux',xlabel=r'Wavenumber [cm$^{-1}$]')
            ax.legend(fontsize=8)
            err_ax.plot(nu,values,color=BLUE,lw=1)
            err_ax.scatter(nu[flagged],values[flagged],color=RED,s=28,zorder=4)
            err_ax.axhline(self.threshold,color=RED,ls='--',lw=1)
            err_ax.set(title=f'Sample {ids[worst]}: {flagged.sum()} points > {self.threshold:g}%',
                       ylabel='Absolute flux relative error [%]',xlabel=r'Wavenumber [cm$^{-1}$]')
            for sample,col in np.argwhere(over_threshold(error[:,sl],self.threshold)):
                point_rows.append([str(name),int(ids[sample]),float(nu[col]),float(error[sample,offset+col])])
            for panel in [ax,err_ax]:
                panel.grid(alpha=.15)
            offset+=int(size)
        fig.suptitle(f'Inversion spectrum fit: sample {ids[worst]} (largest spectral RMSRE)')
        self.save(fig,self.output_dir/'11_inversion_spectrum_fit.png')
        write_csv(self.output_dir/'inversion_spectral_points_over_threshold.csv',
                  ['window','sample_id','wavenumber_cm-1','absolute_relative_error_percent'],point_rows)
        self.summary['inversion_spectra']={'source':mode,'displayed_sample_id':int(ids[worst]),
                                         'flagged_points':len(point_rows),'total_points':int(error.size)}

    def reconstruct_legacy_gp_spectra(self,ids,truth,parameters):
        """Compatibility only: predict with frozen models; never run retrieval."""
        report_path=self.input_dir/'inversion_metrics.json'
        if not report_path.exists():
            self.note('Legacy inversion has no spectral archive or GP hash record; figure 11 needs a fresh inversion result.')
            return None
        self.record_source(report_path)
        report=json.loads(report_path.read_text(encoding='utf-8'))
        hashes=report.get('checkpoint_sha256',{})
        if not hashes:
            self.note('No frozen GP hash record; skipped inversion spectrum reconstruction.')
            return None
        # Reuse the existing numerical implementation, not a second model formula.
        import torch
        from threadpoolctl import threadpool_limits
        from invert_surrogate import NumpyGP, read_dataset, validate_checkpoint_data
        self.note('Legacy results: predicting spectra at saved parameters with verified frozen GP models (no optimization).')
        theta=parameters.copy()
        theta[:,2:]=np.log10(theta[:,2:])
        names=[]; grids=[]; observations=[]; predictions=[]
        for name,digest in hashes.items():
            directory=self.model_dir/name
            if not directory.is_dir() and self.model_dir.name==name:
                directory=self.model_dir
            checkpoint=directory/'surrogate_model.pt'
            if file_hash(checkpoint)!=digest:
                raise ValueError(f'{name}: current checkpoint differs from the inversion model')
            self.record_source(checkpoint)
            self.record_source(directory/'ml_inputs.csv')
            self.record_source(directory/'ml_log_flux.csv')
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            validate_checkpoint_data(ck,directory)
            saved_ids,x,y,nu=read_dataset(directory)
            lookup={int(sid):i for i,sid in enumerate(saved_ids)}
            rows=np.array([lookup[int(sid)] for sid in ids])
            physical=x[rows].copy()
            physical[:,2:]=10**physical[:,2:]
            if not np.allclose(physical,truth,atol=1e-12,rtol=1e-9):
                raise ValueError('Inversion truth differs from the matching forward dataset')
            model=NumpyGP(ck['gp_config'])
            with threadpool_limits(limits=1):
                prediction=np.array([model.predict_jacobian(row)[0] for row in theta])
            names.append(name); grids.append(nu); observations.append(y[rows]); predictions.append(prediction)
        observed=np.concatenate(observations,axis=1)
        predicted=np.concatenate(predictions,axis=1)
        mape=np.abs(np.expm1(np.log(10)*(predicted-observed))).mean()*100
        if not np.isclose(mape,report['spectrum']['flux_mape_percent'],rtol=1e-6,atol=1e-8):
            raise ValueError('Reconstructed spectra do not reproduce the saved inversion error')
        return np.array(names),np.array([len(g) for g in grids]),np.concatenate(grids),observed,predicted

    def run(self,stage='all'):
        children=sorted(p for p in self.input_dir.glob('window_*') if p.is_dir())
        windows=children or [self.input_dir]
        for window in windows:
            destination=self.output_dir/window.name if children else self.output_dir
            if stage in ('all','forward'):
                self.forward(window,destination)
            if stage in ('all','surrogate'):
                self.surrogate(window,destination)
        if stage in ('all','surrogate'):
            self.spectral_flag_figure()
        if stage in ('all','inversion'):
            self.inversion()
        self.summary.update(figures=self.figures,sources=self.sources,notes=self.notes,stage=stage)
        (self.output_dir/'visualization_summary.json').write_text(json.dumps(self.summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'完成：{len(self.figures)} 张图像，输出目录：{self.output_dir}',flush=True)
        if 'spectral_flagged_points' in self.summary:
            print(f'光谱超标点：{self.summary["spectral_flagged_points"]}',flush=True)
        if 'inversion' in self.summary:
            print(f'气体丰度超标样本：{len(self.summary["inversion"]["gas_flagged_sample_ids"])}',flush=True)
        return self.summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--output-dir',type=Path,help='默认：输入目录/visualizations')
    parser.add_argument('--model-dir',type=Path,help='仅旧版缺少反演光谱时，用于查找对应冻结模型')
    parser.add_argument('--threshold',type=float,default=5.,help='绝对相对误差阈值，单位百分比；默认 5，严格大于')
    parser.add_argument('--stage',choices=['all','forward','surrogate','inversion'],default='all')
    parser.add_argument('--dpi',type=int,default=180)
    args=parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold<=0:
        parser.error('--threshold must be finite and positive')
    if args.dpi<50:
        parser.error('--dpi must be at least 50')
    if not args.input_dir.is_dir():
        parser.error('--input-dir does not exist')
    output=args.output_dir or args.input_dir/'visualizations'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'legend.frameon':False,'axes.titlepad':10})
    Visualizer(args.input_dir,output,args.threshold,args.dpi,args.model_dir).run(args.stage)


if __name__=='__main__':
    main()

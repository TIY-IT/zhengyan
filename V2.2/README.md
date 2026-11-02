# V1.1 HITRAN LBL + Two-stream + ML reproduction

This directory contains a standalone reproduction of the workflow shown in the teacher-provided images. The physical forward model reads only `../dataset/6aa4b041.par`; it does **not** read `code/ml_test_dataset.npz` or `code/run_summary.json`.

## What is reproduced

1. Parse H2O, CO, and CH4 records from the HITRAN 160-character line file.
2. Correct 296 K reference line strengths to each atmospheric-layer temperature.
3. Combine Doppler and pressure broadening with a Voigt profile.
4. Sum lines and species to obtain layer-by-layer mass opacity.
5. Convert opacity to optical depth in 24 pressure layers.
6. Integrate upward and downward flux with a pure-absorption two-stream solver.
7. Calculate top-of-atmosphere emission and brightness temperature.
8. Generate a manually selected number of new input/spectrum pairs in CSV format.
9. Train a fixed 5-to-20001-output PyTorch MLP as a workflow demonstration.
10. Run both batched forward modeling and surrogate training on CUDA, with CPU fallback.

## Run from the project root

本机已验证的 GPU Python 是：

```text
D:\DL_Environment\pytorch\.venv\Scripts\python.exe
```

注意 `DL_Environment` 是一个完整目录名，不是 `DL\_Environment` 两级目录。建议从项目根目录运行：

```powershell
$gpuPython = "D:\DL_Environment\pytorch\.venv\Scripts\python.exe"
& $gpuPython "V1.1\reproduce_lbl_twostream.py" --device gpu
& $gpuPython "V1.1\train_surrogate.py" --device gpu
```

运行第一条命令后，程序会提示手动输入正演样本组数；第二条命令会提示手动输入 MLP 训练次数：

```text
请输入要生成的正演样本组数（输入 0 跳过）：25
请输入 MLP 训练次数（epochs，例如 2000）：2000
```

每组大气参数仍使用固定种子 `20260903` 的 Latin-hypercube 方法生成，因此相同组数会得到相同输入。

自动化或脚本运行时仍可直接指定组数，不出现交互提示：

```powershell
& $gpuPython "V1.1\reproduce_lbl_twostream.py" --device gpu --ml-samples 25
```

只生成参考大气、不生成批量样本：

```powershell
& $gpuPython "V1.1\reproduce_lbl_twostream.py" --device gpu --ml-samples 0
```

由于 V1.1 使用 0.05 cm^-1 网格、输出通道数为 20001，上千组样本会明显增加显存和存储需求。建议先使用较小样本数验证流程，并根据显存调整批大小：

```powershell
& $gpuPython "V1.1\reproduce_lbl_twostream.py" --device gpu --gpu-dtype float32 --gpu-batch-size 8 --ml-samples 1000
& $gpuPython "V1.1\train_surrogate.py" --device gpu --input-dir "V1.1\outputs"
```

训练完成后会在 `V1.1/outputs/surrogate_model.pt` 保存 MLP 权重、输入输出归一化参数以及训练/测试样本划分。

自动化运行时可以同时指定样本数和训练次数，不出现交互提示：

```powershell
& $gpuPython "V1.1\reproduce_lbl_twostream.py" --device gpu --ml-samples 1000
& $gpuPython "V1.1\train_surrogate.py" --device gpu --input-dir "V1.1\outputs" --epochs 2000
```

使用 MLP 替代模型计算参数反演误差：

```powershell
& $gpuPython "V1.1\invert_surrogate.py" --device gpu --input-dir "V1.1\outputs" --inverse-learning-rate 0.05
```

反演脚本会直接加载 `V1.1/outputs/surrogate_model.pt`，不会重新训练 MLP；默认梯度下降学习率为 `0.05`，也可通过 `--inverse-learning-rate` 修改。如检查点不在默认位置，可通过 `--model-path` 指定。省略 `--inversion-samples` 时会手动输入用于反演的测试样本数，输入 `0` 表示使用全部测试样本。反演结果写入 `inversion_error_by_sample.csv`、`inversion_error_report.txt` 以及三张反演图。

`--device auto` 会在当前 PyTorch 能访问 CUDA 时自动使用 GPU，否则回退 CPU；`--device gpu` 则会在 CUDA 不可用时明确报错。`float32` 用于高吞吐数据生成；需要严格复核时可使用 `--gpu-dtype float64`。

GPU 运行会自动使用同一组参考大气再做一次精确 CPU/SciPy 计算，并在终端、`run_summary.txt` 和 `gpu_error_report.txt` 中输出误差。逐波数结果写入 `gpu_reference_error.csv`。上千组样本时该检查仍只比较一个参考大气；如确认过环境并希望省去这部分时间，可添加 `--skip-gpu-error`。

Use `& $gpuPython "V1.1\reproduce_lbl_twostream.py" --help` to inspect configurable temperatures, abundances, layer count, gravity, mean molecular mass, GPU batch size, precision, and output location.

## Verified GPU behavior

- Environment: Python 3.12.10, PyTorch 2.6.0+cu124.
- Device: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB class.
- The HITRAN tensors and spectral grids stay resident on CUDA across sample batches; only completed top-of-atmosphere spectra are copied back for CSV output.
- On the identical 16-sample test, CUDA float32 with batch size 8 took 10.633 s versus 25.783 s on CPU, including the reference atmosphere, plots, and CSV files: 2.43x end-to-end speedup.
- Across all 16 spectra, CUDA float32 versus exact CPU log10-flux RMSE was 1.52e-5 and maximum absolute difference was 6.79e-4.
- CUDA float64 matched the CPU reference extrema at printed precision, but is slower on this consumer GPU.

## Reference configuration

| Quantity | Value |
|---|---:|
| Wavenumber interval | 1500-2500 cm^-1 |
| Wavenumber points | 20001 |
| Wavenumber spacing | 0.05 cm^-1 |
| Pressure edges | 1-100000 Pa, log spaced |
| Atmospheric layers | 24 |
| Top/bottom temperatures | 550/1450 K |
| log10 H2O abundance | -3.2 |
| log10 CO abundance | -4.1 |
| log10 CH4 abundance | -5.0 |
| Gravity | 9.8 m s^-2 |
| Mean molecular mass | 28.97 amu |
| Two-stream representative cosine | 1/sqrt(3) |

## Outputs

All generated files are written to `outputs/`.

| File | Meaning |
|---|---|
| `01_hitran_line_list.png` | Real line positions and 296 K strengths by species |
| `02_temperature_pressure.png` | Reference temperature-pressure profile |
| `03_opacity_selected_layers.png` | Mass opacity at four atmospheric levels |
| `04_emission_spectrum.png` | Top-of-atmosphere two-stream spectrum and blackbody curves |
| `05_brightness_temperature.png` | Brightness temperature spectrum |
| `06_ml_synthetic_spectra.png` | Up to five of the newly generated ML spectra |
| `07_surrogate_prediction.png` | One held-out truth/prediction comparison |
| `08_surrogate_error.png` | Surrogate RMSE and MAE versus wavenumber |
| `line_inventory.csv` | Parsed molecule counts and coverage |
| `atmosphere_profile.csv` | Pressure and temperature of every layer |
| `example_opacity_all_layers.csv` | 24-layer mass opacity matrix |
| `example_spectrum.csv` | Wavenumber, outgoing flux, and brightness temperature |
| `ml_inputs.csv` | The selected number of rows with five atmospheric parameters |
| `ml_log_flux.csv` | Corresponding fixed 20001-channel log10 flux targets |
| `surrogate_predictions.csv` | Held-out truth and MLP predictions in long format |
| `gpu_reference_error.csv` | Per-wavenumber GPU-versus-exact-CPU flux, temperature, and opacity errors |
| `gpu_error_report.txt` | Aggregate GPU numerical MAE, RMSE, maximum absolute, and relative errors |
| `surrogate_error_by_wavenumber.csv` | Per-channel surrogate RMSE, MAE, and physical-flux relative errors |
| `surrogate_error_by_sample.csv` | Aggregate log-flux, physical-flux, and brightness-temperature errors for every held-out atmosphere |
| `run_summary.txt` | Plain-text physical-run summary |
| `surrogate_report.txt` | Plain-text ML metrics |
| `surrogate_error_report.txt` | Dedicated aggregate surrogate-error report for the held-out test set |

## Modeling assumptions

- Line strength uses the teacher document's approximation `Q(296)/Q(T) = (296/T)^1.5`, because no TIPS partition-function tables are present.
- Air and self broadening from the HITRAN file are used. The background is therefore treated as an air-like mixture with mean molecular mass 28.97 amu.
- Abundances are volume/mole fractions.
- HITRAN cross sections in cm^2/molecule are converted to mass opacity in m^2/kg before applying `delta_tau = kappa * delta_p / g`.
- The lower boundary emits as a blackbody at the bottom-layer temperature; no radiation enters at the top.
- Scattering, clouds, non-LTE effects, collision-induced absorption, continuum absorption, and line mixing are excluded.
- The input file contains no CH4 lines below about 2097.8 cm^-1.
- Only transitions whose centers are in the downloaded 1500-2500 cm^-1 file are available, so line-wing contributions from transitions just outside the interval are absent.
- V1.1 uses a fixed 20001-point grid over 1500-2500 cm^-1 with 0.05 cm^-1 spacing. The MLP output dimension is fixed at 20001 and must match the generated CSV files.

## Interpretation of the ML result

The surrogate is a CUDA-capable PyTorch network with standardized inputs and per-wavenumber targets. A scientific emulator should be trained on thousands or more forward atmospheres and evaluated on a separate, sufficiently large test set; its accuracy must be reported from that large run rather than an earlier tiny demonstration.

误差文件中，`log10_flux_*` 衡量网络在对数光谱上的误差；`linear_flux_*` 先把预测值还原到物理通量；`brightness_temperature_*` 将通量误差转换成亮温误差。`median` 表示中位数，`p95` 表示95%的测试点不超过该误差，通常比单个极端 `max` 更能描述整体精度。GPU误差与替代模型误差应分开解读：前者来自数值精度和Voigt近似，后者来自网络拟合与数据覆盖。

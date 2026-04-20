# v2 Baseline Results

## Changes from v1
1. `Config.low_hz`: 0.6 → 0.75 Hz (bandpass lower bound)
2. `GREEN.green()`: removed `detrend_poly2`, bandpass handles low-freq removal
3. `SignalQualityController`: added calibration phase (first 20s marked `[CALIBRATING]`)
4. `RPPGAlgorithms.cbcr_pos()`: new Cb/Cr native algorithm added to fusion
5. `FusionEngine._remove_outliers()`: median-based outlier filter with temporal
   continuity protection before fusion voting

## 3-3.mp4 (GT: 79-86 BPM, 30fps, 720x1280)
MediaPipe Tasks face detection, hybrid7 ROI preset.

### Error Summary (5-29s, excluding warmup)
| Algorithm | MAE   | RMSE  | Max   |
|-----------|-------|-------|-------|
| FUSED     | 3.4   | 4.2   | 8.2   |
| GREEN     | 6.8   | 10.6  | 26.1  |
| CHROM     | 6.8   | 8.1   | 19.4  |
| POS       | 8.8   | 10.6  | 21.2  |
| CBCR      | 8.0   | 9.4   | 22.4  |

### vs v1
| Algorithm | v1 MAE | v2 MAE | Change |
|-----------|--------|--------|--------|
| GREEN     | 10.5   | 6.8    | -35%   |
| CHROM     | 12.7   | 6.8    | -46%   |
| POS       | 11.9   | 8.8    | -26%   |
| CBCR      | 11.3   | 8.0    | -29%   |
| FUSED     | 3.3    | 3.4    | ~same  |

Note: v1 FUSED=3.3 was a lucky value (GREEN occasionally correct + temporal
smoothing compensating other algorithms' systematic bias). v2 FUSED=3.4 is
on a more stable foundation with all four algorithms improved.

## 3-2.mp4 (GT: 100-114 BPM, 30fps, 720x1280)
### Error Summary (5-29s)
| Algorithm | MAE   | RMSE  | Max   |
|-----------|-------|-------|-------|
| FUSED     | 5.6   | 7.4   | 24.9  |
| GREEN     | 35.2  | 36.3  | 48.2  |
| CHROM     | 33.7  | 34.5  | 51.4  |
| POS       | 33.3  | 34.9  | 50.3  |
| CBCR      | 34.0  | 34.8  | 49.6  |

### Known issue: 3-2 sub-harmonic
Raw green channel PSD has no peak near 104 BPM (1.73 Hz).
Strongest peak is at 54.4 BPM (≈ GT/2), likely due to:
- 30fps + portrait orientation → low per-beat pixel SNR
- Pulse waveform dicrotic notch creates half-frequency envelope
- Not fixable by signal processing alone; needs acquisition-level changes.

## Diagnostic findings
- 0.75 Hz spurious peak was caused by poly2 detrend residual oscillation
- Confirmed: raw green PSD has correct peak at #2 (83.4 BPM), poly2 creates
  artifact at #1 (44.8 BPM) that pushes correct peak to #3
- poly1 has the same problem — mechanism is polynomial fitting itself, not order
- Chrominance projection (CHROM/POS/CBCR) eliminates HR peak entirely in
  low-SNR conditions (3-3 and 3-2), only GREEN preserves it
- highpass(0.75Hz) via bandpass lower bound is sufficient to remove the artifact

## Systematic bias analysis
- Single-ROI welch_hr estimates are accurate: 5 good ROIs average +2.9 BPM bias
  in stable period (20-29s), within 1 frequency bin
- Parabolic interpolation verified correct: corr(asymmetry, interp_shift) = 0.929
- Hann leakage and bandpass group delay ruled out as bias sources
- FUSED bias (-3.4 BPM) comes from CHROM/POS/CBCR candidates being
  systematically low (~10 BPM) due to chrominance projection destroying HR peak
  in low-SNR conditions; these outvote GREEN's correct values in median fusion
- Remaining bias is the ceiling for outlier filter; needs chrominance projection
  gating (per-ROI peak-frequency consistency check) to resolve

## Outlier filter
- Removes candidates deviating >20 BPM from group median
- Temporal continuity protection: candidates near last_valid_hr are preserved
  even if far from current median (prevents filtering correct values when
  majority of algorithms are wrong, as in 3-2 sub-harmonic scenario)
- Effect: FUSED MAE 3.85 → 3.41 on 3-3, neutral on 3-2

## Cold start analysis
- Single ROI (forehead_center) has no correct HR peak in PSD for first 18s
- Multi-ROI median fusion is what makes 5-15s usable (other ROIs have correct peaks)
- **Critical**: reducing ROI count will degrade cold-start accuracy significantly
- 15-21s "collapse" in single-ROI analysis is caused by sliding window (11.6s)
  containing early unstable signal, not by external disturbance
- Camera AGC/AWB produces ±0.8 G-channel drift in first 10s; bandpass transient
  response from this drift creates in-band artifacts (102 BPM false peak)
- Exponential time-weighting tested and rejected: does not recover correct peak,
  introduces spectral leakage, breaks previously correct windows

## Calibration phase
- `SignalQualityController.is_calibrating` = True for first 20s from t0
- t0 counted from process start (first frame), not from first face detection
- HR values still computed and published during calibration; flag is advisory
- UI should display "calibrating" and optionally hide numeric HR during this phase
- 20s threshold is empirical; multi-ROI fusion makes readings usable from ~15s
  but single-ROI instability extends to ~18s

## Known limitations
| Limitation | Nature | Next step |
|-----------|--------|-----------|
| 3-2 high HR sub-harmonic | Signal acquisition, algorithm cannot fix | Hardware/acquisition strategy |
| Cold start depends on multi-ROI | Design constraint, calibrating tag mitigates | Monitor, don't reduce ROI count |
| CHROM/POS/CBCR low-SNR projection | Algorithm, structural bias source | See "Chrominance projection gating" below |
| `_remove_outliers` temporal protection may delay response to rapid HR changes (>20 BPM jump) | Edge case | Monitor in real-world testing |
| `_cluster_values` gap=10 BPM | Structural: when GREEN correct value and projection wrong value are <10 BPM apart, they merge into one cluster, making weighting and gating ineffective | Evaluate gap increase after more video samples |

## Chrominance projection gating — explored approaches (all closed)

### Approach 1: Peak frequency consistency (GREEN vs CHROM)
- Proposed: discard CHROM when |CHROM_peak - GREEN_peak| > threshold
- **Result: Invalid assumption.** IMG_0061 shows GREEN fails on 3/7 ROIs while
  CHROM is correct. Cannot assume GREEN is the baseline. Would discard correct
  CHROM values on good video.

### Approach 2: Peak SNR as quality signal
- Proposed: discard candidates with low peak_power / median_power
- **Result: No discriminative power.** Correct and wrong peaks have overlapping
  SNR distributions. 3-3: correct range [1.3, 3.9], wrong range [1.2, 3.0].
  3-2: wrong peaks have HIGHER SNR (6.9) than correct peaks (1.6) because
  sub-harmonics are strong signals, just at wrong frequency.

### Approach 3: Projection internal consistency (proj_std)
- Proposed: if std(CHROM, POS, CBCR) is high, projection is unstable → discard
- **Result: Partial.** High std (>2.5) reliably marks unstable projections, but
  these are already caught by outlier filter. Low std does NOT prove correctness —
  3-3/forehead_left has std=0.42 with all three algorithms converged to ~53 BPM
  (wrong). "Stable but wrong" is the hard case and proj_std cannot detect it.

### Approach 4: GREEN 2x voting weight
- Proposed: duplicate GREEN candidates in fusion to outvote 3 projection algos
- **Result: Zero effect.** `_cluster_values` with gap=10 BPM merges GREEN's
  correct values (~83) with projection's wrong values (~70) into one cluster.
  Adding votes to a merged cluster does not change the cluster median. This is
  a structural limitation of the clustering approach, not a weighting issue.

### Residual bias analysis
- FUSED bias = -3.4 BPM on 3-3 (stable period 20-29s)
- Single-ROI welch_hr is accurate: 5 good ROIs average +2.9 BPM (within 1 bin)
- Bias comes from CHROM/POS/CBCR outvoting GREEN in fusion (3:1 per ROI)
- This is the structural ceiling under current architecture
- **Next step: test on more video samples** to determine if this bias is
  specific to 30fps portrait videos or a general phenomenon before investing
  in architectural changes

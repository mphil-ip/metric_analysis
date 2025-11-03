### Anomaly Detection Methods

- **Overview**  
  - Describe how robust statistics identify outliers in noisy signals.  
  - Emphasize resilience of median-based estimators to skewed or contaminated data.

- **Median Absolute Deviation (MAD)**  
  - Definition: MAD = median(|xᵢ − median(x)|); uses the median as the central location.  
  - Properties: 50% breakdown point, immune to extreme spikes; scale estimate (σ̂) = 1.4826 × MAD for large-sample normal data.  
  - Usage: Benchmark variability when variance is inflated by outliers; drives thresholds for point anomalies.

- **Robust Z-Score**  
  - Formula: zᵣ = (x − median) / (1.4826 × MAD); substitutes mean/std with robust analogs.  
  - Interpretation: |zᵣ| ≥ k flags anomalies (k ≈ 3–3.5 for normal-like noise); handles heavy tails and level shifts better than classical z-scores.

- **Baseline Median**  
  - Concept: Reference level representing long-run central tendency; typically a rolling or seasonal median.  
  - Calculation Strategy:  
    - Rolling window (e.g., prior N points) to track slow drifts.  
    - Seasonal buckets (same weekday/hour) when periodic patterns exist.  
  - Role: Aligns comparisons to recent “normal” state; reduces drift-induced false alarms.

- **Baseline MAD / Robust Deviation**  
  - Compute with the same baseline window to capture prevailing volatility.  
  - Optional smoothing (exponential or median filtering) prevents threshold chatter.  
  - Apply minimum floors to avoid division by near-zero noise.

- **Thresholding Approaches**  
  - Static k × MAD on top of baseline.  
  - Adaptive quantiles (e.g., 99th percentile within window) for asymmetric metrics.  
  - Two-sided vs. one-sided detection depending on business impact (spikes vs. drops).

- **Implementation Notes**  
  - Guard against sparse windows: require minimum sample count before evaluating.  
  - Handle missing data: impute conservatively or skip to avoid biasing medians.  
  - Combine with persistence rules (M-of-N exceedances) for noise suppression.

- **Extensions**  
  - Seasonal hybrid: combine baseline median per seasonal bucket with overall trend median.  
  - Multivariate: compute robust Mahalanobis distance using median/MAD per feature.  
  - Streaming: update medians/MAD via reservoir sampling or online robust estimators.

- **Validation & Monitoring**  
  - Backtest on historical incidents; tune k to balance precision/recall.  
  - Track alert volume and distribution of |zᵣ| to catch drift in variability.  
  - Document assumptions (stationarity, independence) and known failure modes.

- **Key Takeaways**  
  - MAD-based methods excel when data are contaminated by outliers or non-Gaussian noise.  
  - Baseline medians anchor detection to the current regime.  
  - Robust z-scores provide interpretable, scalable anomaly scores with minimal tuning.
# DJ Prep Dashboard

A Streamlit dashboard for reviewing batch audio analysis data, flagging risk tracks, and recommending DJ transitions based on BPM and key metrics.

## Features
- Upload a CSV of batch analysis results
- Summary metrics for BPM, loudness, and duration
- Risk detection for loud tracks, clipping risk, and low-confidence analysis
- Key and BPM distribution charts
- Track table with filtering and sorting
- DJ transition recommendations based on BPM and key compatibility

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV expectations
The app is designed to work with CSV files similar to the batch analysis exports produced by audio analysis tools. Common columns include:

- filename
- bpm_est / bpm_estimate
- key_est / key_estimate
- duration_s / duration_seconds
- integrated_lufs / loudness_integrated_lufs
- true_peak_dbfs / true_peak_dbtp
- key_score / key_confidence

If a column is missing, the dashboard will still load the data where possible and mark unavailable fields as `NaN`.

## Recommended use
Use this dashboard to:
- identify risky or problematic tracks
- find mixes that are BPM/Key compatible
- prioritize transition-ready tracks for DJ sets
- export a final shortlist for sequencing

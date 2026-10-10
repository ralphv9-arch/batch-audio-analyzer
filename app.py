import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="DJ Prep Dashboard", page_icon="🎧", layout="wide")


COLUMN_ALIASES = {
    "file_name": "filename",
    "track": "filename",
    "title": "filename",
    "track_name": "filename",
    "song": "filename",
    "bpm": "bpm_est",
    "tempo": "bpm_est",
    "bpm_estimate": "bpm_est",
    "bpm_estimated": "bpm_est",
    "key": "key_est",
    "key_estimate": "key_est",
    "key_guess": "key_est",
    "duration": "duration_s",
    "duration_seconds": "duration_s",
    "length_seconds": "duration_s",
    "integrated_lufs": "integrated_lufs",
    "loudness_integrated_lufs": "integrated_lufs",
    "lufs": "integrated_lufs",
    "true_peak_dbfs": "true_peak_dbfs",
    "true_peak_dbtp": "true_peak_dbfs",
    "peak_dbfs": "true_peak_dbfs",
    "key_score": "key_score",
    "key_confidence": "key_score",
    "confidence": "key_score",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={old: new for old, new in COLUMN_ALIASES.items() if old in df.columns})

    for col in ["filename", "key_est"]:
        if col not in df.columns:
            if col == "filename":
                df[col] = [f"track_{i + 1}" for i in range(len(df))]
            else:
                df[col] = "Unknown"

    for col in ["bpm_est", "duration_s", "integrated_lufs", "true_peak_dbfs", "key_score"]:
        if col not in df.columns:
            df[col] = float("nan")

    df["filename"] = df["filename"].astype(str)
    return df


def key_distance(key_a: str, key_b: str) -> float:
    if pd.isna(key_a) or pd.isna(key_b):
        return 999.0

    key_map = {
        "A": 0, "Am": 0, "A minor": 0,
        "Bb": 1, "A#": 1,
        "B": 2, "Bm": 2,
        "C": 3, "Cm": 3,
        "Db": 4, "C#": 4,
        "D": 5, "Dm": 5,
        "Eb": 6, "D#": 6,
        "E": 7, "Em": 7,
        "F": 8, "Fm": 8,
        "Gb": 9, "F#": 9,
        "G": 10, "Gm": 10,
        "Ab": 11, "G#": 11,
    }

    def clean(value):
        value = str(value).strip().lower().replace(" ", "")
        mapping = {
            "aminor": "Am",
            "am": "Am",
            "bb": "Bb",
            "a#": "A#",
            "b#": "C",
            "db": "Db",
            "c#": "C#",
            "eb": "Eb",
            "d#": "D#",
            "gb": "Gb",
            "f#": "F#",
            "ab": "Ab",
            "g#": "G#",
        }
        return mapping.get(value, value.capitalize())

    a = clean(key_a)
    b = clean(key_b)
    if a not in key_map or b not in key_map:
        return 999.0
    diff = abs(key_map[a] - key_map[b])
    return min(diff, 12 - diff)


def risk_score(row: pd.Series) -> float:
    score = 0.0
    if pd.notna(row.get("integrated_lufs")):
        if row["integrated_lufs"] > -4:
            score += 30
        elif row["integrated_lufs"] < -14:
            score += 20
    if pd.notna(row.get("true_peak_dbfs")) and row["true_peak_dbfs"] > -1.5:
        score += 25
    if pd.notna(row.get("key_score")) and row["key_score"] < 0.6:
        score += 15
    if pd.notna(row.get("bpm_est")) and (row["bpm_est"] < 60 or row["bpm_est"] > 220):
        score += 10
    return round(min(score, 100), 1)


def risk_flags(row: pd.Series) -> list:
    flags = []
    if pd.notna(row.get("integrated_lufs")):
        if row["integrated_lufs"] > -4:
            flags.append("Very loud")
        elif row["integrated_lufs"] < -14:
            flags.append("Very quiet")
    if pd.notna(row.get("true_peak_dbfs")) and row["true_peak_dbfs"] > -1.5:
        flags.append("Peak risk")
    if pd.notna(row.get("key_score")) and row["key_score"] < 0.6:
        flags.append("Low key confidence")
    if pd.notna(row.get("bpm_est")) and (row["bpm_est"] < 60 or row["bpm_est"] > 220):
        flags.append("Unusual BPM")
    return flags


def transition_recommendations(df: pd.DataFrame, ref_index: int = 0, bpm_tolerance: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["track", "bpm_est", "key_est", "compatibility_score", "note"])

    ref = df.iloc[ref_index]
    ref_bpm = ref.get("bpm_est", math.nan)
    ref_key = ref.get("key_est", "Unknown")

    rows = []
    for idx, row in df.iterrows():
        if idx == ref_index:
            continue

        bpm_val = row.get("bpm_est")
        bpm_gap = abs(float(bpm_val) - float(ref_bpm)) if pd.notna(ref_bpm) and pd.notna(bpm_val) else 999
        key_gap = key_distance(ref_key, row.get("key_est", "Unknown"))
        score = max(0, 100 - (bpm_gap * 6) - (key_gap * 8))

        if pd.notna(bpm_val):
            if bpm_gap <= bpm_tolerance and key_gap <= 2:
                note = "Excellent transition"
            elif bpm_gap <= bpm_tolerance + 4:
                note = "Good transition"
            else:
                note = "Break / energy shift"
        else:
            note = "Insufficient tempo data"

        rows.append({
            "track": row.get("filename", f"track_{idx + 1}"),
            "bpm_est": row.get("bpm_est", math.nan),
            "key_est": row.get("key_est", "Unknown"),
            "compatibility_score": round(max(0, min(100, score)), 1),
            "note": note,
        })

    return pd.DataFrame(rows).sort_values("compatibility_score", ascending=False).head(10)


def load_data() -> pd.DataFrame:
    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] is not None:
        df = pd.read_csv(st.session_state["uploaded_file"])
    else:
        default_path = Path("data/sample_batch_analysis.csv")
        if default_path.exists():
            df = pd.read_csv(default_path)
        else:
            df = pd.DataFrame([
                {"filename": "track_01.mp3", "bpm_est": 124, "key_est": "A minor", "duration_s": 215, "integrated_lufs": -8.5, "true_peak_dbfs": -1.2, "key_score": 0.92},
                {"filename": "track_02.mp3", "bpm_est": 126, "key_est": "C major", "duration_s": 230, "integrated_lufs": -7.2, "true_peak_dbfs": -1.8, "key_score": 0.88},
                {"filename": "track_03.mp3", "bpm_est": 128, "key_est": "F major", "duration_s": 198, "integrated_lufs": -10.2, "true_peak_dbfs": -2.3, "key_score": 0.71},
                {"filename": "track_04.mp3", "bpm_est": 116, "key_est": "E minor", "duration_s": 245, "integrated_lufs": -11.8, "true_peak_dbfs": -2.0, "key_score": 0.84},
                {"filename": "track_05.mp3", "bpm_est": 132, "key_est": "G major", "duration_s": 200, "integrated_lufs": -5.4, "true_peak_dbfs": -0.9, "key_score": 0.68},
            ])

    df = normalize_columns(df)
    df["risk_score"] = df.apply(risk_score, axis=1)
    df["risk_flags"] = df.apply(risk_flags, axis=1)
    df["risk_count"] = df["risk_flags"].apply(len)
    return df


def render_dashboard(df: pd.DataFrame):
    st.title("🎧 DJ Prep Dashboard")
    st.caption("Review batch analysis, detect risky tracks, and shortlist mixing-ready candidates.")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        st.session_state["uploaded_file"] = uploaded

    if df.empty:
        st.warning("No data available for the dashboard.")
        return

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks", len(df))
    c2.metric("Avg BPM", f"{df['bpm_est'].mean():.1f}" if pd.notna(df['bpm_est'].mean()) else "N/A")
    c3.metric("Avg LUFS", f"{df['integrated_lufs'].mean():.1f}" if pd.notna(df['integrated_lufs'].mean()) else "N/A")
    c4.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}" if pd.notna(df['risk_score'].mean()) else "N/A")

    st.subheader("Filters")
    filters = st.columns(4)
    min_bpm = filters[0].number_input("Min BPM", min_value=60, max_value=220, value=80)
    max_bpm = filters[1].number_input("Max BPM", min_value=60, max_value=220, value=180)
    key_filter = filters[2].selectbox("Key", ["All"] + sorted(df["key_est"].dropna().unique().tolist()))
    risk_filter = filters[3].selectbox("Risk Filter", ["All", "Risky", "Clean"])

    filtered = df.copy()
    filtered = filtered[filtered["bpm_est"].between(min_bpm, max_bpm, inclusive="both")]
    if key_filter != "All":
        filtered = filtered[filtered["key_est"] == key_filter]
    if risk_filter == "Risky":
        filtered = filtered[filtered["risk_score"] >= 25]
    elif risk_filter == "Clean":
        filtered = filtered[filtered["risk_score"] < 25]

    st.subheader("BPM Distribution")
    bpm_fig = px.histogram(df.dropna(subset=["bpm_est"]), x="bpm_est", nbins=25, title="Estimated BPM Distribution")
    st.plotly_chart(bpm_fig, use_container_width=True)

    st.subheader("Key Distribution")
    key_counts = df["key_est"].value_counts().reset_index()
    key_counts.columns = ["Key", "Count"]
    key_fig = px.bar(key_counts, x="Key", y="Count", title="Key Distribution")
    st.plotly_chart(key_fig, use_container_width=True)

    st.subheader("Risk Review")
    risk_df = df[["filename", "bpm_est", "key_est", "integrated_lufs", "true_peak_dbfs", "risk_score", "risk_flags"]].copy()
    risk_df["risk_flags"] = risk_df["risk_flags"].apply(lambda x: ", ".join(x) if x else "None")
    st.dataframe(risk_df.sort_values("risk_score", ascending=False), use_container_width=True)

    st.subheader("Filtered Track List")
    st.dataframe(filtered[["filename", "bpm_est", "key_est", "integrated_lufs", "true_peak_dbfs", "risk_score"]], use_container_width=True)

    if not filtered.empty:
        st.subheader("Recommended DJ Shortlist")
        shortlist = transition_recommendations(filtered, ref_index=0)
        if shortlist.empty:
            st.info("Not enough compatible tracks for recommendations.")
        else:
            st.dataframe(shortlist, use_container_width=True)

    st.subheader("Suggested Set Order")
    suggested = filtered.sort_values(["bpm_est", "risk_score"], ascending=[True, False]).head(10)
    st.dataframe(suggested[["filename", "bpm_est", "key_est", "risk_score", "integrated_lufs"]], use_container_width=True)


def main():
    try:
        df = load_data()
        render_dashboard(df)
    except Exception as exc:
        st.error(f"Failed to load dashboard: {exc}")


if __name__ == "__main__":
    main()

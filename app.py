import math
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def compatibility_score(track_a: pd.Series, track_b: pd.Series, bpm_tolerance: int = 3) -> float:
    """Calculate compatibility between two tracks (0-100)."""
    bpm_a = track_a.get("bpm_est")
    bpm_b = track_b.get("bpm_est")
    key_a = track_a.get("key_est", "Unknown")
    key_b = track_b.get("key_est", "Unknown")

    if pd.isna(bpm_a) or pd.isna(bpm_b):
        return 0.0

    bpm_gap = abs(float(bpm_a) - float(bpm_b))
    key_gap = key_distance(key_a, key_b)
    score = max(0, 100 - (bpm_gap * 6) - (key_gap * 8))
    return round(max(0, min(100, score)), 1)


def build_compatibility_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compatibility matrix between all tracks."""
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    n = len(df)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = compatibility_score(df.iloc[i], df.iloc[j])
            else:
                matrix[i][j] = 0.0

    return pd.DataFrame(matrix, index=df["filename"].values, columns=df["filename"].values)


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

        score = compatibility_score(ref, row, bpm_tolerance)
        bpm_val = row.get("bpm_est")
        bpm_gap = abs(float(bpm_val) - float(ref_bpm)) if pd.notna(ref_bpm) and pd.notna(bpm_val) else 999

        if pd.notna(bpm_val):
            if bpm_gap <= bpm_tolerance:
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
            "compatibility_score": score,
            "note": note,
        })

    return pd.DataFrame(rows).sort_values("compatibility_score", ascending=False).head(10)


def build_optimal_set_order(df: pd.DataFrame, max_tracks: int = 10) -> list:
    """Build an optimal DJ set order using a greedy algorithm."""
    if df.empty:
        return []

    clean_df = df[df["risk_score"] < 50].copy()
    if clean_df.empty:
        clean_df = df.copy()

    if len(clean_df) > max_tracks:
        clean_df = clean_df.sort_values("risk_score").head(max_tracks)

    if len(clean_df) < 2:
        return clean_df.index.tolist()

    # Start with lowest risk track
    order = [clean_df["risk_score"].idxmin()]
    remaining = set(clean_df.index) - {order[0]}

    while remaining:
        current_idx = order[-1]
        current_track = clean_df.loc[current_idx]

        best_next = None
        best_score = -1

        for next_idx in remaining:
            next_track = clean_df.loc[next_idx]
            compat = compatibility_score(current_track, next_track)

            if compat > best_score:
                best_score = compat
                best_next = next_idx

        if best_next is not None:
            order.append(best_next)
            remaining.remove(best_next)
        else:
            break

    return order


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
                {"filename": "track_06.mp3", "bpm_est": 120, "key_est": "D minor", "duration_s": 220, "integrated_lufs": -9.2, "true_peak_dbfs": -1.6, "key_score": 0.79},
            ])

    df = normalize_columns(df)
    df["risk_score"] = df.apply(risk_score, axis=1)
    df["risk_flags"] = df.apply(risk_flags, axis=1)
    df["risk_count"] = df["risk_flags"].apply(len)
    return df


def render_dashboard(df: pd.DataFrame):
    st.title("🎧 DJ Prep Dashboard")
    st.caption("Review batch analysis, detect risky tracks, build compatible sets, and export shortlists.")

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

    st.subheader("Compatibility Matrix")
    if len(filtered) > 1:
        matrix = build_compatibility_matrix(filtered)
        fig = px.imshow(matrix, title="Track Compatibility Scores (0-100)", color_continuous_scale="Viridis", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 tracks to build compatibility matrix.")

    st.subheader("Recommended DJ Set Order")
    if len(filtered) > 1:
        optimal_order = build_optimal_set_order(filtered, max_tracks=10)
        set_tracks = filtered.loc[optimal_order, ["filename", "bpm_est", "key_est", "risk_score"]].reset_index(drop=True)
        set_tracks.index = set_tracks.index + 1
        set_tracks.index.name = "Position"
        st.dataframe(set_tracks, use_container_width=True)

        csv_buffer = StringIO()
        set_tracks.to_csv(csv_buffer)
        st.download_button(
            label="📥 Download Recommended Set as CSV",
            data=csv_buffer.getvalue(),
            file_name="recommended_dj_set.csv",
            mime="text/csv"
        )
    else:
        st.info("Not enough clean tracks to build a set.")

    st.subheader("Export Shortlist")
    shortlist = filtered[["filename", "bpm_est", "key_est", "duration_s", "integrated_lufs", "risk_score"]].copy()
    shortlist = shortlist.sort_values("risk_score")
    if not shortlist.empty:
        csv_buffer = StringIO()
        shortlist.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Shortlist as CSV",
            data=csv_buffer.getvalue(),
            file_name="track_shortlist.csv",
            mime="text/csv"
        )
    else:
        st.info("No tracks match the current filters.")


def main():
    try:
        df = load_data()
        render_dashboard(df)
    except Exception as exc:
        st.error(f"Failed to load dashboard: {exc}")


if __name__ == "__main__":
    main()

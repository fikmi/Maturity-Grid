import io
import json
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dateutil import parser


ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]
SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}

COLUMN_ALIASES: Dict[str, List[str]] = {
    "team": ["team", "equipe", "équipe", "squad", "tribe"],
    "axis": ["axis", "axe", "category", "dimension"],
    "metric": ["metric", "kpi", "type", "item", "subject"],
    "status": ["status", "statut", "result", "etat", "state"],
    "value": ["value", "taille", "points", "story_points", "size"],
    "created_at": ["created_at", "created", "start", "start_date", "opened", "timestamp", "submitted_at"],
    "ended_at": [
        "ended_at",
        "done",
        "closed",
        "closed_at",
        "completed_at",
        "merged_at",
        "resolutiondate",
        "finished_at",
    ],
    "in_progress_at": ["in_progress_at", "progress_at", "started_at"],
    "team_lead": ["lead", "manager", "owner"],
}

AXIS_VALUE_ALIASES = {
    "conception": "Conception",
    "design": "Conception",
    "discovery": "Conception",
    "développement": "Développement",
    "developpement": "Développement",
    "development": "Développement",
    "dev": "Développement",
    "implementation": "Développement",
    "test": "Test",
    "qa": "Test",
    "quality": "Test",
    "release": "Release",
    "deploy": "Release",
    "deployment": "Release",
    "deploiement": "Release",
    "déploiement": "Release",
}

STATUS_VALUE_ALIASES = {
    "ready": "ready",
    "prêt": "ready",
    "pret": "ready",
    "done": "done",
    "completed": "done",
    "closed": "done",
    "finished": "done",
    "in_progress": "in_progress",
    "progress": "in_progress",
    "wip": "in_progress",
    "pass": "pass",
    "ok": "pass",
    "success": "pass",
    "fail": "fail",
    "failed": "fail",
    "ko": "fail",
    "incident": "incident",
    "hotfix": "hotfix",
    "open": "open",
    "todo": "open",
}

DEFAULT_AXIS_ORDER = ["Conception", "Développement", "Test", "Release"]


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value


def detect_separator(sample: str) -> Optional[str]:
    candidates = [",", ";", "\t", "|"]
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return None
    scores = {}
    for sep in candidates:
        counts = [line.count(sep) for line in lines[:5]]
        score = sum(counts)
        if all(c > 0 for c in counts[: min(len(counts), 3)]):
            score += 5
        scores[sep] = score
    best_sep = max(scores, key=scores.get)
    return best_sep if scores[best_sep] > 0 else None


def decode_bytes(data: bytes) -> str:
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="ignore")


def read_text_from_path(path: Path) -> str:
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1", errors="ignore")


def read_any(file_name: str, file_bytes: Optional[bytes] = None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Read a supported file and return a DataFrame plus log messages."""
    messages: List[str] = []
    suffix = Path(file_name).suffix.lower()
    try:
        if file_bytes is not None:
            text = decode_bytes(file_bytes)
        else:
            text = read_text_from_path(Path(file_name))
    except FileNotFoundError:
        messages.append(f"Fichier introuvable : {file_name}")
        return None, messages
    except Exception as exc:  # pragma: no cover - unexpected error handling
        messages.append(f"Impossible de lire {file_name} : {exc}")
        return None, messages

    if suffix in {".json", ""}:
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                df = pd.json_normalize(payload)
            elif isinstance(payload, dict):
                df = pd.json_normalize(payload)
            else:
                messages.append(f"Format JSON inattendu dans {file_name}")
                return None, messages
        except json.JSONDecodeError:
            messages.append(f"JSON invalide dans {file_name}")
            return None, messages
        return df, messages

    if suffix == ".jsonl":
        try:
            df = pd.read_json(io.StringIO(text), lines=True)
            return df, messages
        except ValueError as exc:
            messages.append(f"JSONL invalide dans {file_name}: {exc}")
            return None, messages

    if suffix in {".csv", ".txt"}:
        separator = detect_separator(text)
        buffer = io.StringIO(text)
        try:
            if separator:
                df = pd.read_csv(buffer, sep=separator)
            else:
                df = pd.read_csv(buffer)
        except Exception:
            rows = [line for line in text.splitlines() if line.strip()]
            df = pd.DataFrame({"raw": rows})
            messages.append(
                f"Séparateur non détecté pour {file_name}, passage en mode colonne unique."
            )
        return df, messages

    messages.append(f"Extension non supportée : {suffix}")
    return None, messages


def coerce_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().all():
        return parsed
    mask_na = parsed.isna()
    if mask_na.any():
        def _parse(value: object) -> pd.Timestamp:
            if value in (None, "", pd.NA):
                return pd.NaT
            try:
                dt = parser.parse(str(value))
            except (ValueError, TypeError):
                return pd.NaT
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return pd.Timestamp(dt)

        parsed.loc[mask_na] = series.loc[mask_na].apply(_parse)
    return parsed


def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    if df is None or df.empty:
        return pd.DataFrame(), {"recognized": [], "extras": []}

    normalized = {col: normalize_text(col) if isinstance(col, str) else col for col in df.columns}
    df = df.rename(columns=normalized)

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename_map[alias] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    if "axis" in df.columns:
        df["axis"] = df["axis"].apply(lambda x: AXIS_VALUE_ALIASES.get(normalize_text(x), x))
    if "status" in df.columns:
        df["status"] = df["status"].apply(lambda x: STATUS_VALUE_ALIASES.get(normalize_text(x), normalize_text(x)))

    for col in ["created_at", "ended_at", "in_progress_at"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col])

    if "created_at" not in df.columns and "timestamp" in df.columns:
        df["created_at"] = coerce_datetime(df["timestamp"])

    if "team" in df.columns:
        df["team"] = df["team"].fillna("Inconnu").astype(str)

    if "axis" in df.columns:
        df["axis"] = df["axis"].fillna("Non défini").astype(str)

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    recognized = [col for col in COLUMN_ALIASES if col in df.columns]
    extras = [col for col in df.columns if col not in COLUMN_ALIASES and col != "source"]

    return df, {"recognized": sorted(recognized), "extras": sorted(extras)}


def compute_cycle_time_hours(df: pd.DataFrame) -> pd.Series:
    if "created_at" not in df.columns or "ended_at" not in df.columns:
        return pd.Series([pd.NA] * len(df))
    return (df["ended_at"] - df["created_at"]).dt.total_seconds() / 3600


def compute_kpis(df: pd.DataFrame, since: datetime, until: datetime) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if df.empty:
        return pd.DataFrame(columns=["axis", "team", "metric", "value", "unit", "details"]), {}

    mask_created = pd.Series(False, index=df.index)
    mask_ended = pd.Series(False, index=df.index)
    if "created_at" in df.columns:
        mask_created = df["created_at"].between(since, until, inclusive="both")
    if "ended_at" in df.columns:
        mask_ended = df["ended_at"].between(since, until, inclusive="both")
    mask = mask_created | mask_ended
    if not mask.any():
        mask = pd.Series(True, index=df.index)

    scoped = df.loc[mask].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["axis", "team", "metric", "value", "unit", "details"]), {}

    scoped["team"] = scoped.get("team", pd.Series(dtype=str)).fillna("Non attribué")
    scoped["axis"] = scoped.get("axis", pd.Series(dtype=str)).fillna("Non défini")
    scoped["status"] = scoped.get("status", pd.Series(dtype=str)).fillna("indéfini")

    scoped["cycle_time_h"] = compute_cycle_time_hours(scoped)
    scoped["cycle_time_d"] = scoped["cycle_time_h"] / 24

    period_days = max((until - since).days, 1)
    kpi_records: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}

    for axis in DEFAULT_AXIS_ORDER:
        axis_rows = scoped[scoped["axis"].str.lower() == axis.lower()]
        if axis_rows.empty:
            continue
        summary.setdefault(axis, {})
        teams = axis_rows["team"].fillna("Non attribué").unique()
        for team in teams:
            team_rows = axis_rows[axis_rows["team"] == team]
            if team_rows.empty:
                continue
            add_axis_team_metrics(axis, team, team_rows, period_days, kpi_records, summary)
        add_axis_team_metrics(axis, "Toutes équipes", axis_rows, period_days, kpi_records, summary, aggregate_only=True)

    kpis_df = pd.DataFrame(kpi_records)
    return kpis_df, summary


def add_axis_team_metrics(
    axis: str,
    team: str,
    data: pd.DataFrame,
    period_days: int,
    kpi_records: List[Dict[str, object]],
    summary: Dict[str, Dict[str, float]],
    aggregate_only: bool = False,
) -> None:
    count = len(data)
    cycle_mean = data["cycle_time_d"].dropna().mean()
    status_counts = data["status"].value_counts(dropna=False)
    ready_ratio = status_counts.get("ready", 0) / count if count else None
    pass_count = status_counts.get("pass", 0)
    fail_count = status_counts.get("fail", 0)
    success_rate = pass_count / (pass_count + fail_count) if (pass_count + fail_count) else None
    throughput_per_week = (count / period_days) * 7 if period_days else None
    mean_value = data.get("value", pd.Series(dtype=float)).dropna().mean()
    incidents = status_counts.get("incident", 0) + status_counts.get("hotfix", 0)
    incident_ratio = incidents / count if count else None

    summary.setdefault(axis, {})

    if aggregate_only:
        axis_summary = summary[axis]
        axis_summary["count"] = float(count)
        if ready_ratio is not None:
            axis_summary["ready_ratio"] = ready_ratio
        if throughput_per_week is not None:
            axis_summary["throughput"] = throughput_per_week
        if mean_value is not None and not pd.isna(mean_value):
            axis_summary["mean_value"] = float(mean_value)
        if success_rate is not None:
            axis_summary["success_rate"] = success_rate
        if incident_ratio is not None:
            axis_summary["incident_ratio"] = incident_ratio
        if cycle_mean is not None and not pd.isna(cycle_mean):
            axis_summary["cycle_days"] = float(cycle_mean)
        return

    metrics: List[Tuple[str, Optional[float], str, str]] = []

    if axis == "Conception":
        if ready_ratio is not None:
            metrics.append(("Taux de préparation", ready_ratio * 100, "%", "Items avec statut ready"))
        if cycle_mean is not None and not pd.isna(cycle_mean):
            metrics.append(("Cycle moyen", cycle_mean, "jours", "Temps moyen création → fin"))
    elif axis == "Développement":
        if throughput_per_week is not None:
            metrics.append(("Items/semaine", throughput_per_week, "count", "Cadence moyenne"))
        if mean_value is not None and not pd.isna(mean_value):
            metrics.append(("Valeur moyenne", mean_value, "pts", "Taille moyenne de l'item"))
        if cycle_mean is not None and not pd.isna(cycle_mean):
            metrics.append(("Cycle moyen", cycle_mean, "jours", "Temps moyen création → fin"))
    elif axis == "Test":
        if success_rate is not None:
            metrics.append(("Taux de succès", success_rate * 100, "%", "Tests passés vs échoués"))
        if throughput_per_week is not None:
            metrics.append(("Volumétrie", throughput_per_week, "count", "Nombre de tests/semaine"))
    elif axis == "Release":
        if throughput_per_week is not None:
            metrics.append(("Fréquence", throughput_per_week, "count", "Livraisons par semaine"))
        if incident_ratio is not None:
            metrics.append(("Ratio incidents", incident_ratio * 100, "%", "Part des incidents/hotfix"))
        if cycle_mean is not None and not pd.isna(cycle_mean):
            metrics.append(("Cycle moyen", cycle_mean, "jours", "Temps moyen création → fin"))
    else:
        if throughput_per_week is not None:
            metrics.append(("Volumétrie", throughput_per_week, "count", "Occurrences par semaine"))

    for metric_name, value, unit, details in metrics:
        if value is None or pd.isna(value):
            continue
        kpi_records.append(
            {
                "axis": axis,
                "team": team,
                "metric": metric_name,
                "value": round(float(value), 2),
                "unit": unit,
                "details": details,
            }
        )


def score_axes(summary: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], float]:
    scores: Dict[str, float] = {}
    for axis, metrics in summary.items():
        if axis == "Conception":
            ready = metrics.get("ready_ratio") or 0
            cycle = metrics.get("cycle_days") or float("inf")
            ready_score = score_from_thresholds(ready, [
                (0.85, 5),
                (0.7, 4),
                (0.5, 3),
                (0.3, 2),
                (0.1, 1),
            ])
            cycle_score = score_from_inverse_thresholds(cycle, [
                (2, 5),
                (4, 4),
                (7, 3),
                (10, 2),
                (14, 1),
            ])
            scores[axis] = round((ready_score + cycle_score) / 2, 2)
        elif axis == "Développement":
            throughput = metrics.get("throughput") or 0
            cycle = metrics.get("cycle_days") or float("inf")
            throughput_score = score_from_thresholds(throughput, [
                (15, 5),
                (10, 4),
                (6, 3),
                (3, 2),
                (1, 1),
            ])
            cycle_score = score_from_inverse_thresholds(cycle, [
                (3, 5),
                (5, 4),
                (8, 3),
                (12, 2),
                (16, 1),
            ])
            scores[axis] = round((throughput_score + cycle_score) / 2, 2)
        elif axis == "Test":
            success = metrics.get("success_rate")
            if success is None:
                scores[axis] = 0.0
            else:
                scores[axis] = score_from_thresholds(success, [
                    (0.95, 5),
                    (0.85, 4),
                    (0.75, 3),
                    (0.6, 2),
                    (0.4, 1),
                ])
        elif axis == "Release":
            freq = metrics.get("throughput") or 0
            incident_ratio = metrics.get("incident_ratio")
            freq_score = score_from_thresholds(freq, [
                (8, 5),
                (5, 4),
                (3, 3),
                (1, 2),
                (0.5, 1),
            ])
            incident_score = 5 - score_from_thresholds(incident_ratio or 0, [
                (0.4, 5),
                (0.3, 4),
                (0.2, 3),
                (0.1, 2),
                (0.05, 1),
            ]) if incident_ratio is not None else 3
            scores[axis] = round((freq_score + incident_score) / 2, 2)
        else:
            scores[axis] = 0.0

    global_score = round(sum(scores.values()) / len(scores), 2) if scores else 0.0
    return scores, global_score


def score_from_thresholds(value: float, thresholds: List[Tuple[float, float]]) -> float:
    for threshold, score in thresholds:
        if value >= threshold:
            return float(score)
    return 0.0


def score_from_inverse_thresholds(value: float, thresholds: List[Tuple[float, float]]) -> float:
    for threshold, score in thresholds:
        if value <= threshold:
            return float(score)
    return 0.0


def make_charts(df: pd.DataFrame, since: datetime, until: datetime) -> Dict[str, pd.DataFrame]:
    charts: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return charts

    if "created_at" in df.columns:
        timeline = df.dropna(subset=["created_at"]).copy()
        if not timeline.empty:
            timeline["date"] = timeline["created_at"].dt.tz_localize(None).dt.date
            axis_trend = (
                timeline.groupby(["date", "axis"])
                .size()
                .reset_index(name="count")
                .sort_values("date")
            )
            charts["axis_trend"] = axis_trend

    if "team" in df.columns and "axis" in df.columns:
        team_counts = (
            df.groupby(["team", "axis"])
            .size()
            .reset_index(name="count")
            .sort_values(["axis", "team"])
        )
        charts["team_axis"] = team_counts

    if "status" in df.columns and "axis" in df.columns:
        status_counts = (
            df.groupby(["axis", "status"])
            .size()
            .reset_index(name="count")
            .sort_values(["axis", "count"], ascending=[True, False])
        )
        charts["status_axis"] = status_counts

    return charts


def list_local_files(folder: str) -> List[Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    files: List[Path] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def load_all_sources(
    uploaded_files: List, local_folder: str
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    frames: List[pd.DataFrame] = []
    logs: List[str] = []
    columns_info: Dict[str, List[str]] = {"recognized": [], "extras": []}

    for file in uploaded_files or []:
        data, file_logs = read_any(file.name, file.getvalue())
        logs.extend(file_logs)
        if data is None or data.empty:
            logs.append(f"Aucune donnée exploitable dans {file.name}.")
            continue
        data["source"] = file.name
        standardized, info = standardize_columns(data)
        frames.append(standardized)
        columns_info["recognized"].extend(info.get("recognized", []))
        columns_info["extras"].extend(info.get("extras", []))

    for path in list_local_files(local_folder):
        data, file_logs = read_any(str(path))
        logs.extend(file_logs)
        if data is None or data.empty:
            logs.append(f"Aucune donnée exploitable dans {path.name}.")
            continue
        data["source"] = path.name
        standardized, info = standardize_columns(data)
        frames.append(standardized)
        columns_info["recognized"].extend(info.get("recognized", []))
        columns_info["extras"].extend(info.get("extras", []))

    if not frames:
        return pd.DataFrame(), logs, columns_info

    unified = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
    columns_info["recognized"] = sorted(set(columns_info["recognized"]))
    columns_info["extras"] = sorted(set(columns_info["extras"]))
    return unified, logs, columns_info


def save_unified(df: pd.DataFrame, fmt: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("data", exist_ok=True)
    if fmt == "CSV":
        path = Path("data") / f"unified-{timestamp}.csv"
        df.to_csv(path, index=False)
        return str(path)
    if fmt == "DuckDB":
        try:
            import duckdb  # type: ignore
        except ImportError:
            raise RuntimeError("DuckDB non disponible. Installez le paquet duckdb.")
        path = Path("data") / "dashboard.duckdb"
        con = duckdb.connect(str(path))
        con.register("df", df)
        con.execute("CREATE TABLE IF NOT EXISTS maturity AS SELECT * FROM df")
        con.execute("DELETE FROM maturity")
        con.execute("INSERT INTO maturity SELECT * FROM df")
        con.close()
        return str(path)
    raise ValueError(f"Format non supporté : {fmt}")


# --- Streamlit UI ---
st.set_page_config(page_title="Maturity Grid Dashboard", layout="wide")
st.title("📊 Maturity Grid Dashboard")
st.caption("Suivi local de la maturité des équipes sur les axes clés du delivery.")

with st.sidebar:
    st.header("Sources")
    uploaded_files = st.file_uploader(
        "Importer des fichiers",
        type=["json", "jsonl", "csv", "txt"],
        accept_multiple_files=True,
        help="Les fichiers chargés restent locaux à l'application",
    )
    local_folder = st.text_input("Dossier local", value="sample_data", help="Chemin vers un dossier contenant des données")
    refresh_enabled = st.checkbox("Auto-rafraîchissement")
    refresh_interval = st.slider("Intervalle (secondes)", 30, 120, 60) if refresh_enabled else None
    if refresh_enabled and refresh_interval:
        components.html(
            f"<script>setTimeout(function(){{parent.window.location.reload();}},{refresh_interval * 1000});</script>",
            height=0,
        )

    st.divider()
    st.header("Filtres")
    period_choice = st.selectbox("Période", ["7 jours", "30 jours", "90 jours", "Personnalisée"])

    until_default = datetime.utcnow()
    if period_choice == "Personnalisée":
        since = st.date_input("Depuis", value=until_default.date() - timedelta(days=30))
        until = st.date_input("Jusqu'au", value=until_default.date())
        since_dt = datetime.combine(since, datetime.min.time(), tzinfo=timezone.utc)
        until_dt = datetime.combine(until, datetime.max.time(), tzinfo=timezone.utc)
    else:
        days = int(period_choice.split()[0])
        since_dt = (until_default - timedelta(days=days)).replace(tzinfo=timezone.utc)
        until_dt = until_default.replace(tzinfo=timezone.utc)

    save_format = st.selectbox("Format de sauvegarde", ["CSV", "DuckDB"])


data, logs, columns_info = load_all_sources(uploaded_files, local_folder)

if logs:
    with st.expander("Journal de chargement", expanded=False):
        for message in logs:
            st.write("-", message)

if data.empty:
    st.warning("Aucune donnée disponible. Chargez au moins un fichier valide.")
    st.stop()

teams = sorted(data.get("team", pd.Series(dtype=str)).dropna().unique())
axes = sorted(data.get("axis", pd.Series(dtype=str)).dropna().unique())

with st.sidebar:
    selected_teams = st.multiselect("Équipes", teams, default=teams)
    selected_axes = st.multiselect("Axes", axes, default=axes)
    st.divider()
    if st.button("Sauvegarder l'unifié", disabled=data.empty):
        try:
            saved_path = save_unified(data, save_format)
            st.success(f"Données sauvegardées dans {saved_path}")
        except Exception as exc:  # pragma: no cover - feedback utilisateur
            st.error(str(exc))

filtered = data[data["team"].isin(selected_teams) & data["axis"].isin(selected_axes)]

kpis_df, summary = compute_kpis(filtered, since_dt, until_dt)
axis_scores, global_score = score_axes(summary)
charts = make_charts(filtered, since_dt, until_dt)

st.subheader("Synthèse")
col_global, *cols_axes = st.columns(5)
col_global.metric("Score global", f"{global_score:.2f}/5", delta=None)

for axis in DEFAULT_AXIS_ORDER:
    if axis not in axis_scores:
        continue
    column = cols_axes.pop(0) if cols_axes else st.columns(1)[0]
    score = axis_scores[axis]
    delta = None
    if "axis_trend" in charts and not charts["axis_trend"].empty:
        axis_trend = charts["axis_trend"]
        axis_data = axis_trend[axis_trend["axis"] == axis]
        recent = axis_data.tail(7)["count"].sum()
        previous = axis_data.tail(14).head(7)["count"].sum()
        if previous:
            delta = f"{((recent - previous) / previous) * 100:.1f}%"
    column.metric(axis, f"{score:.2f}/5", delta=delta)
    if "axis_trend" in charts and not charts["axis_trend"].empty:
        axis_data = charts["axis_trend"][charts["axis_trend"]["axis"] == axis]
        with column:
            if not axis_data.empty:
                st.line_chart(axis_data.set_index("date")["count"], height=120)

st.caption(f"Dernier rafraîchissement : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

st.subheader("Analyse détaillée")
left_col, right_col = st.columns([2, 1])
with left_col:
    st.write("### KPI par équipe et axe")
    if not kpis_df.empty:
        st.dataframe(kpis_df, use_container_width=True)
        csv_bytes = kpis_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Exporter les KPI filtrés",
            data=csv_bytes,
            file_name="kpis.csv",
            mime="text/csv",
        )
    else:
        st.info("Aucun KPI disponible pour la période sélectionnée.")

with right_col:
    st.write("### Colonnes reconnues")
    st.write(
        ", ".join(columns_info.get("recognized", [])) or "Aucune colonne standard reconnue"
    )
    st.write("### Colonnes supplémentaires")
    st.write(
        ", ".join(columns_info.get("extras", [])) or "Aucune colonne supplémentaire"
    )

st.divider()

st.write("### Graphiques")
graph_col1, graph_col2 = st.columns(2)
with graph_col1:
    if "team_axis" in charts and not charts["team_axis"].empty:
        st.bar_chart(
            data=charts["team_axis"],
            x="team",
            y="count",
            color="axis",
            height=300,
        )
    else:
        st.info("Pas de données pour le graphique par équipe.")

with graph_col2:
    if "status_axis" in charts and not charts["status_axis"].empty:
        st.bar_chart(
            data=charts["status_axis"],
            x="axis",
            y="count",
            color="status",
            height=300,
        )
    else:
        st.info("Pas de données pour la répartition des statuts.")

st.divider()

st.write("### Table unifiée")
st.dataframe(filtered, use_container_width=True)

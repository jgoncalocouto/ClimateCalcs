import tempfile
# epw_dashboard_pro_adaptive_v2.py
# Streamlit + Plotly EPW Explorer — Pro + Adaptive (v2 tweaks for Single tab)
#
# Run: streamlit run epw_dashboard_pro_adaptive_v2.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import plotly.express as px
import plotly.graph_objects as go
from fcns import *
from plotly.subplots import make_subplots

st.set_page_config(page_title="EnergyPlus Weather File Explorer", layout="wide")


# ==============================
# Google Drive: setup & helpers
# ==============================
if "GDRIVE_FOLDER_ID" not in st.secrets:
    st.error("Missing secret: GDRIVE_FOLDER_ID in .streamlit/secrets.toml"); st.stop()
FOLDER_ID = st.secrets["GDRIVE_FOLDER_ID"]

svc_info = None
if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
    import json as _json_for_secrets
    svc_info = _json_for_secrets.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
elif "google_service_account" in st.secrets:
    svc_info = dict(st.secrets["google_service_account"])
else:
    st.error("Missing service account config. Add GOOGLE_SERVICE_ACCOUNT_JSON or [google_service_account] to secrets.toml"); st.stop()

_scopes = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

@st.cache_resource(show_spinner=False)
def _get_drive_service(_svc_info: dict):
    creds = Credentials.from_service_account_info(_svc_info, scopes=_scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

_drive = _get_drive_service(svc_info)

def _list_epw_files_drive():
    q = f"'{FOLDER_ID}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder'"
    files, page_token = [], None
    while True:
        res = _drive.files().list(
            q=q,
            fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return [f for f in files if f["name"].lower().endswith(".epw")]

@st.cache_data(show_spinner=False)
def _download_epw_bytes(file_id: str) -> bytes:
    req = _drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def _upsert_epw_to_drive(filename: str, data: bytes) -> str:
    safe_name = filename.replace("'", "\'")
    q = f"'{FOLDER_ID}' in parents and trashed=false and name = '{safe_name}'"
    res = _drive.files().list(
        q=q,
        fields="files(id, name, parents)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=10,
    ).execute()
    files = res.get("files", [])

    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/octet-stream", resumable=False)

    if files:
        file_id = files[0]["id"]
        _drive.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True,
        ).execute()
        return file_id
    else:
        file_metadata = {
            "name": filename,
            "parents": [FOLDER_ID],
        }
        created = _drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return created["id"]

def _parse_selected_from_catalog(name: str, _catalog: dict):
    item = _catalog[name]
    # materialize to temp file and parse with existing fcns.parse_one (expects a Path)
    tmp = tempfile.NamedTemporaryFile(suffix=".epw", delete=False)
    try:
        if item["source"] == "uploaded":
            tmp.write(item["bytes"])
        else:
            data = _download_epw_bytes(item["id"])
            tmp.write(data)
        tmp.flush(); tmp.close()
        from fcns import parse_one  # local import to avoid circulars at top
        return parse_one(Path(tmp.name))
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

st.markdown(
    """
    <style>
    /* Make the tab labels really big & bold */
    .stTabs [role="tablist"] button {
        padding: 1rem 1.5rem !important;      /* taller/wider tabs */
    }
    .stTabs [role="tablist"] button p,
    .stTabs [role="tablist"] button div[data-testid="stMarkdownContainer"] {
        font-size: 1.6rem !important;        /* bump font size */
        font-weight: 800 !important;         /* heavy bold */
        margin: 0 !important;
        line-height: 1.3 !important;
    }

    /* Optional: style the active tab */
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 4px solid var(--primary-color, #2E86C1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
DATA_DIR = Path("data")

# --------------- Helpers ---------------

def available_epws():
    if not DATA_DIR.exists():
        st.error(f"Folder not found: {DATA_DIR.resolve()}"); st.stop()
    files = sorted([p for p in DATA_DIR.glob("*.epw")])
    if not files:
        st.warning("No .epw files found in the 'data' folder."); st.stop()
    return files


def compute_daily_ghi(df, ghi_col):
    ghi = num(df[ghi_col]).fillna(0.0)
    return (ghi.resample("D").sum() / 1000.0).rename("kWh/m²·day").to_frame()

def emit_triplet(cols, start_idx, label, s, unit):
    """Write Mean/Min/Max metrics for series s into 3 adjacent columns."""
    if s is not None:
        with cols[start_idx + 0]: st.metric(f"{label} Mean", fmt(s.mean(), f" {unit}"))
        with cols[start_idx + 1]: st.metric(f"{label} Min",  fmt(s.min(),  f" {unit}"))
        with cols[start_idx + 2]: st.metric(f"{label} Max",  fmt(s.max(),  f" {unit}"))
    else:
        with cols[start_idx + 0]: st.metric(f"{label} Mean", "—")
        with cols[start_idx + 1]: st.metric(f"{label} Min",  "—")
        with cols[start_idx + 2]: st.metric(f"{label} Max",  "—")





# --------------- Load files (Drive + Upload) ---------------
col_up, col_refresh = st.columns([3,1])
with col_up:
    _uploaded = st.file_uploader("Upload EPW files (optional)", type=["epw"], accept_multiple_files=True)
with col_refresh:
    if st.button("🔄 Refresh Drive list"):
        st.cache_data.clear()
        st.rerun()

if "uploaded_epws" not in st.session_state:
    st.session_state.uploaded_epws = {}  # name -> bytes

if _uploaded:
    _save_to_drive = st.toggle("Save uploads to Google Drive", value=True, key="save_drive_toggle")
    for f in _uploaded:
        content = f.read()
        st.session_state.uploaded_epws[f.name] = content
        if _save_to_drive:
            try:
                _upsert_epw_to_drive(f.name, content)
                st.toast(f"Saved to Drive: {f.name}", icon="✅")
            except Exception as e:
                st.warning(f"Could not save '{f.name}' to Drive: {e}")

_drive_files = _list_epw_files_drive()
_catalog = {}
for f in _drive_files:
    _catalog[f["name"]] = {"source": "drive", "id": f["id"], "meta": f}
for name, b in st.session_state.uploaded_epws.items():
    _catalog[name] = {"source": "uploaded", "bytes": b, "meta": {"size": len(b)}}

epw_names = sorted(_catalog.keys())
if not epw_names:
    st.warning("No EPWs available yet. Upload some or place them in the Drive folder and press Refresh."); st.stop()
st.title("🌤️ EnergyPLus File Explorer")
st.caption("Analyzes EPW files existent in a local database.")

tab_single, tab_compare = st.tabs(["📁 Deep Dive analysis", "🆚 Head to Head comparisson"])

# =====================================================================================
# SINGLE FILE TAB
# =====================================================================================
with tab_single:
    
    with st.expander("Selection", expanded=True):
    
        
        selected_file = st.selectbox("Select EPW file", epw_names, index=0)
        with st.spinner("Downloading / Parsing EPW..."):
            try:
                df, location_dict, df_stats = _parse_selected_from_catalog(selected_file, _catalog)
            except Exception as e:
                st.error(f"Failed to parse EPW: {e}"); st.stop()
## Time Filtering
        min_date, max_date = df.index.min().date(), df.index.max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="single_dates")
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, date_range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        #Update dataframe to timeperiod selected
        df = df.loc[mask]
        df_stats=dataframe_stats(df)
        
        ## File & Location Details
        # File Selection
        st.markdown("**File Selection**")
        st.caption(f"File: `{selected_file}`")
        # Location details
        st.markdown("**Location**")
        if location_dict:
            loc_df = pd.DataFrame(location_dict, index=[0])
            st.dataframe(loc_df, width='stretch',hide_index=True)
        else:
            st.info("No location metadata available.")
        
    ## Download timeseries
    with st.expander("📥 Download filtered data", expanded=False):
        st.download_button("CSV",
                           df.to_csv().encode("utf-8"),
                           file_name=f"{Path(selected_file).stem}_timeseries.csv",
                           mime="text/csv")
        try:
            import io
            buf = io.BytesIO(); df.to_parquet(buf, index=True)
            st.download_button("Parquet",
                               data=buf.getvalue(),
                               file_name=f"{Path(selected_file).stem}_timeseries.parquet",
                               mime="application/octet-stream")
        except Exception:
            st.caption("Parquet export unavailable (install pyarrow if needed).")
    
    
    with st.container(border=True):
        ## KPI Pane
        st.markdown("### Summary Statistics")
        
        available_cols = list(df.columns)
        dry_col = find_col_kpi(df,"dry_bulb_temperature", "dry bulb")
        dew_col = find_col_kpi(df,"dew_point_temperature", "dew point")
        rh_col  = find_col_kpi(df,"relative_humidity","rh")
        ws_col  = find_col_kpi(df,"wind_speed", "wind speed")
        wd_col  = find_col_kpi(df,"wind_direction", "wind direction")
        dni_col = find_col_kpi(df,"direct_normal_radiation", "dni")
        ghi_col = find_col_kpi(df,"global_horizontal_radiation", "ghi")
        pr_col = find_col_kpi(df,"liquid_precipitation_quantity", "pr")
        
        pairs = [
            (("Dry Bulb Temperature",        series_or_none(df,dry_col), "°C"),
             ("Dew Point Temperature",   series_or_none(df,dew_col), "°C")),
            (("Wind Speed", series_or_none(df,ws_col),  "m/s"),
             ("Wind Direction",   series_or_none(df,wd_col),  "°")),
            (("Direct Normal Radiation", series_or_none(df,dni_col), "Wh/m²"),
             ("Global Horizontal Rad.",  series_or_none(df,ghi_col), "Wh/m²")),
        ]
        
        for (left_label, left_s, left_unit), (right_label, right_s, right_unit) in pairs:
            row = st.columns(6)
            emit_triplet(row, 0, left_label,  left_s,  left_unit)
            emit_triplet(row, 3, right_label, right_s, right_unit)


    # ===== Sync Plot (time-series + histograms)
    def _hist_bins(series, width):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return None
        lo = float(np.floor(s.min() / width) * width)
        hi = float(np.ceil(s.max() / width) * width)
        if hi <= lo: hi = lo + width
        return dict(start=lo, end=hi, size=width)
    
    with st.container(border=True):
        st.subheader("Timeseries Plot")
        
        # Build the 2x6 layout
        rows_def = [
            ("dry", "Dry bulb (°C)", dry_col,  "°C"),
            ("dew", "Dew point (°C)", dew_col,  "°C"),
            ("rh",  "Relative humidity (%)", rh_col,   "%"),
            ("ws",  "Wind speed (m/s)", ws_col,   "m/s"),
            ("ghi", "Global horizontal radiation (W/m²)", ghi_col, "W/m²"),
            ("prec","Liquid precipitation (mm)", pr_col,  "mm"),
        ]
        
        with st.expander("Select Bin width for histograms",expanded=True):
            st.markdown("#### Select Bin width for histograms")
            bw_dry = st.select_slider("Dry bulb bin width (°C)", options=[1.0, 2.0, 5.0], value=2.0, key="bw_dry")
            bw_dew = st.select_slider("Dew point bin width (°C)", options=[1.0, 2.0, 5.0], value=2.0, key="bw_dew")
            bw_rh  = st.select_slider("Humidity bin width (%)", options=[2.0, 5.0, 10.0], value=5.0, key="bw_rh")
            bw_ws  = st.select_slider("Wind speed bin width (m/s)", options=[0.1, 0.5, 1.0], value=0.5, key="bw_ws")
            bw_ghi = st.select_slider("GHI bin width (W/m²)", options=[25, 50, 100, 200], value=100, key="bw_ghi")
            bw_prc = st.select_slider("Precipitation bin width (mm)", options=[0.2, 0.5, 1.0, 2.0], value=1.0, key="bw_prc")
            binwidths = {"dry": bw_dry, "dew": bw_dew, "rh": bw_rh, "ws": bw_ws, "ghi": bw_ghi, "prec": bw_prc}
            
        with st.expander("⬇️ Download histograms",expanded=False):
            df_hists=[]
            for i, (key, title, colnm, unit) in enumerate(rows_def, start=1):
                y = num(df[colnm])
                width = binwidths[key]
                xbins = _hist_bins(y, width)
                df_hist = histogram_dataframe(y, width, unit)
                df_hist["Variable"]=title
                df_hists.append(df_hist)
                
                st.download_button(title,
                                   df_hist.to_csv().encode("utf-8"),
                                   file_name=f"{Path(selected_file).stem}_{key}_hist.csv",
                                   mime="text/csv")

        fig = make_subplots(
            rows=6, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.035,
            column_widths=[0.68, 0.32],
            subplot_titles=[t for _,t,_,_ in rows_def for __ in (0,1)]
        )
        
        ys=[]
        widths=[]
        for i, (key, title, colnm, unit) in enumerate(rows_def, start=1):
            if colnm not in df.columns:
                fig.add_trace(go.Scatter(x=[], y=[], name=title, mode="lines"), row=i, col=1)
                fig.add_trace(go.Histogram(x=[], name=f"{title} hist"), row=i, col=2)
                fig.add_annotation(xref=f"x{i} domain", yref=f"y{i}", x=0.02, y=0.5, showarrow=False,
                                   text=f"“{title}” not found")
                continue
    
            y = num(df[colnm])
            fig.add_trace(
                go.Scatter(x=df.index, y=y, mode="lines", name=title, hovertemplate="%{y:.2f} "+unit),
                row=i, col=1
            )
            fig.update_yaxes(title_text=unit, row=i, col=1)
            width = binwidths[key]
            xbins = _hist_bins(y, width)
            fig.add_trace(
                go.Histogram(x=y, xbins=xbins, name=f"{title} hist", marker_line_width=0.2, opacity=0.95),
                row=i, col=2
            )
            fig.update_xaxes(title_text=unit, row=i, col=2)
            fig.update_yaxes(title_text="Hours", row=i, col=2)
            
        
        fig.update_layout(
            height=300*6,
            hovermode="x unified",
            showlegend=False,
            margin=dict(t=60, r=10, l=60, b=40)
        )
        st.plotly_chart(fig, width="stretch")

    # ===== 2-D Temperature Bins =====
    with st.container(border=True):
        
        with st.container(border=True):
            st.markdown("## Temperature Deep-Dives")
            st.markdown("### Temperature bins — Hour-of-day heatmap (DBT)")
            step = st.number_input("Bin width (°C)", value=2.0, min_value=0.5, step=0.5)
            vals = num(df[dry_col]).dropna()
            if not vals.empty:
                bmin = float(np.floor(vals.min()))
                bmax = float(np.ceil(vals.max()))
                bins_edges = np.arange(bmin, bmax + step, step)
                pivot = hour_of_day_heatmap(df[dry_col], bins_edges)
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[str(b) for b in pivot.columns],
                    y=pivot.index,
                    colorscale="Blues",
                    reversescale=False  # 0 -> white, max -> dark
                ))
                fig.update_layout(title="Hours per bin by hour-of-day (DBT)",
                                  xaxis_title="Temperature bin (°C)",
                                  yaxis_title="Hour")
                st.plotly_chart(fig, width="stretch")
                st.download_button("⬇️ 2-D Histogram Download",
                                   pivot.to_csv().encode("utf-8"),
                                   file_name=f"{Path(selected_file).stem}_DBT_2Dhist.csv",
                                   mime="text/csv")
                
        # ===== Diurnal & Monthly (Dry bulb & Wet Bulb) =====
        with st.container(border=True):
            # Diurnal 
            # Collect data
            frames = []
            if dry_col and dry_col in df.columns:
                dry = pd.to_numeric(df[dry_col], errors="coerce").dropna()
                frames.append(pd.DataFrame({"hour": dry.index.hour, "Temperature (°C)": dry, "Type": "Dry bulb"}))
            if dew_col and dew_col in df.columns:
                dew = pd.to_numeric(df[dew_col], errors="coerce").dropna()
                frames.append(pd.DataFrame({"hour": dew.index.hour, "Temperature (°C)": dew, "Type": "Dew point"}))
            
            if frames:
                plot_df = pd.concat(frames, ignore_index=True)
            
                fig = px.box(
                    plot_df,
                    x="hour", y="Temperature (°C)",
                    color="Type",  # separates Dry vs Dew
                    points="outliers",
                    category_orders={"hour": list(range(24))}
                )
            
                fig.update_layout(
                    xaxis_title="Hour of day",
                    yaxis_title="Temperature (°C)",
                    boxmode="group",  # side-by-side boxes
                    height=450,
                    title="Diurnal distribution of temperatures by hour"
                )
            
                st.plotly_chart(fig, use_container_width=True)

            
        with st.container(border=True):
            # Monthly
            # Collect data
            frames = []
            if dry_col and dry_col in df.columns:
                dry = pd.to_numeric(df[dry_col], errors="coerce").dropna()
                frames.append(pd.DataFrame({"month": dry.index.month, "Temperature (°C)": dry, "Type": "Dry bulb"}))
            if dew_col and dew_col in df.columns:
                dew = pd.to_numeric(df[dew_col], errors="coerce").dropna()
                frames.append(pd.DataFrame({"month": dew.index.month, "Temperature (°C)": dew, "Type": "Dew point"}))
            
            if frames:
                plot_df = pd.concat(frames, ignore_index=True)
            
                fig = px.box(
                    plot_df,
                    x="month", y="Temperature (°C)",
                    color="Type",  # separates Dry vs Dew
                    points="outliers",
                    category_orders={"hour": list(range(24))}
                )
            
                fig.update_layout(
                    xaxis_title="Month of the year",
                    yaxis_title="Temperature (°C)",
                    boxmode="group",  # side-by-side boxes
                    height=450,
                    title="Monthly distribution of temperatures"
                )
            
                st.plotly_chart(fig, use_container_width=True)
            

    
    # ===== Wind rose speed-binned (kept) =====
    with st.container(border=True):
        st.markdown("### Wind direction — speed-binned rose")
        n_sectors = st.slider("Wind rose sectors", min_value=6, max_value=36, value=16, step=2, key="n_sectors_single")
        speed_max = float(num(df[ws_col]).max() or 20.0)
        default_bins = [0,2,5,8,12, max(12, round(speed_max+1))]
        bins_text = st.text_input("Speed bins (m/s, comma-separated)", value=",".join(map(str, default_bins)),
                                  help="Left-closed, right-open bins like 0,2,5,8,12,20")
        try:
            speed_bins = [float(x.strip()) for x in bins_text.split(",") if x.strip()!=""]
            if len(speed_bins) < 2: raise ValueError
        except Exception:
            speed_bins = [0,2,5,8,12,20]
        
        try:
            rose_df = wind_rose_speed_binned(df, ws_col, wd_col, speed_bins, n_sectors)
        except Exception:
            rose_df=pd.DataFrame()
        if not rose_df.empty:
            fig = px.bar_polar(rose_df, r="count", theta="sector", color="speed_bin",
                               title="Wind rose (by speed bins)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.text("Data not available to compute wind rose.")
    
    # ===== Advanced analytics =====
    with st.container(border=True):
   
        # Degree-days with explanation
        st.markdown("### Degree-days (HDD/CDD)")
        st.info("**What are degree-days?**  Heating Degree-Days (HDD) and Cooling Degree-Days (CDD) "
                "measure how much (and for how long) outdoor temperatures are below (HDD) or above (CDD) a chosen base temperature. "
                "They’re a simple way to estimate **seasonal heating/cooling demand**, compare climates, and size HVAC systems at a high level.")
        c1, c2, c3 = st.columns(3)
        with c1: base_hdd = st.number_input("HDD base (°C)", value=18.0, step=0.5)
        with c2: base_cdd = st.number_input("CDD base (°C)", value=27.0, step=0.5)
        with c3: show_monthly = st.checkbox("Show monthly totals", value=True)
    
        if dry_col:
            hdd_daily = degree_days(df[dry_col], base_hdd, "HDD")
            cdd_daily = degree_days(df[dry_col], base_cdd, "CDD")
            k1, k2 = st.columns(2)
            with k1: st.metric("Total HDD (°C·day)", f"{hdd_daily.sum():.1f}")
            with k2: st.metric("Total CDD (°C·day)", f"{cdd_daily.sum():.1f}")
            if show_monthly:
                md = pd.DataFrame({"HDD": hdd_daily, "CDD": cdd_daily})
                md["month"] = md.index.month
                monthly_dd = md.groupby("month")[["HDD","CDD"]].sum().reset_index()
                fig = go.Figure()
                fig.add_bar(x=monthly_dd["month"], y=monthly_dd["HDD"], name="HDD")
                fig.add_bar(x=monthly_dd["month"], y=monthly_dd["CDD"], name="CDD")
                fig.update_layout(barmode="group", xaxis_title="Month", yaxis_title="°C·day")
                st.plotly_chart(fig, width="stretch")
                st.download_button("⬇️ Download Data",
                                   monthly_dd.to_csv().encode("utf-8"),
                                   file_name=f"{Path(selected_file).stem}_HDDCDD_monthly.csv",
                                   mime="text/csv")
        else:
            st.warning("Dry bulb temperature not found; cannot compute degree-days.")
    
    with st.container(border=True):
        # Comfort models with explanations
        st.markdown("### Comfort hours")
        st.info("**Fixed band:** You specify acceptable indoor temperature limits (and optionally RH). "
                "The comfort % is the share of hours when conditions fall within that band."
                "**Adaptive (ASHRAE 55):** For naturally ventilated buildings, comfort temp adapts to recent outdoor climate. "
                "We compute a running-mean of daily outdoor temperature, set the comfort point `17.8 + 0.31 × T_rm`, and check if indoor temps "
                "are within ±3.5 °C (80% acceptability) or ±2.5 °C (90%).")
        mode = st.radio("Comfort model", ["Fixed band", "Adaptive (ASHRAE 55)"], horizontal=True, key="comfort_mode_single")
    
        if mode == "Fixed band":
            c1, c2, c3, c4 = st.columns(4)
            with c1: tmin = st.number_input("Comfort Tmin (°C)", value=18.0, step=0.5)
            with c2: tmax = st.number_input("Comfort Tmax (°C)", value=27.0, step=0.5)
            with c3:
                rhmin = st.number_input("Comfort RH min (%)", value=30.0, step=1.0) if rh_col else None
            with c4:
                rhmax = st.number_input("Comfort RH max (%)", value=60.0, step=1.0) if rh_col else None
    
            cmask = comfort_mask_fixed(df[dry_col] if dry_col else num([]), df[rh_col] if rh_col else None, tmin, tmax, rhmin, rhmax)
            total_hours = cmask.shape[0]; ok_hours = cmask.sum()
            pct = 100.0 * ok_hours / total_hours if total_hours else 0.0
            st.metric("Comfort hours (%)", f"{pct:.1f}%")
    
            cdf = pd.DataFrame({"ok": cmask.astype(int)}); cdf["month"] = cdf.index.month
            comfort_m = cdf.groupby("month")["ok"].mean().reset_index()
            comfort_m["ok"] *= 100.0
            fig = px.bar(comfort_m, x="month", y="ok", title="Monthly Comfort (%) — Fixed band")
            fig.update_layout(yaxis_title="% of hours within comfort band")
            st.plotly_chart(fig, width="stretch")
            comfort_m["HDD_base"]=tmin
            comfort_m["CDD_base"]=tmax
            st.download_button("⬇️ Download Data",
                               comfort_m.to_csv().encode("utf-8"),
                               file_name=f"{Path(selected_file).stem}_comfort_monthly.csv",
                               mime="text/csv")
    
        else:
            c1, c2, c3 = st.columns(3)
            with c1: accept = st.selectbox("Acceptability", ["80", "90"], index=0, help="80% band ±3.5°C, 90% band ±2.5°C")
            with c2: alpha = st.number_input("Running-mean alpha (0–1)", value=0.8, min_value=0.0, max_value=0.99, step=0.05, help="Weight for previous days; common value 0.8")
            with c3:
                choices = [c for c in [dry_col, dew_col] + available_cols if c is not None]
                outdoor_col = st.selectbox("Outdoor temp for running mean", choices, index=0)
    
            daily_out = num(df[outdoor_col]).resample("D").mean()
            band = adaptive_band_from_outdoor(daily_out, alpha=alpha, acceptability=accept)
    
            if dry_col:
                mask_ad, lo_h, hi_h = adaptive_mask_hourly(df[dry_col], band)
                pct = float(mask_ad.mean() * 100.0) if mask_ad.size else 0.0
                st.metric("Comfort hours (%)", f"{pct:.1f}%")
    
                cdf = pd.DataFrame({"ok": mask_ad.astype(int)}); cdf["month"] = cdf.index.month
                comfort_m = cdf.groupby("month")["ok"].mean().reset_index()
                comfort_m["ok"] *= 100.0
                fig = px.bar(comfort_m, x="month", y="ok", title=f"Monthly Comfort (%) — Adaptive {accept}%")
                fig.update_layout(yaxis_title="% of hours within adaptive band")
                st.plotly_chart(fig, width="stretch")
                
                comfort_m["Acceptability"]=accept
                comfort_m["RuningMean"]=alpha
                comfort_m["TemperatureColumn"]=outdoor_col
                st.download_button("⬇️ Download Data",
                                   comfort_m.to_csv().encode("utf-8"),
                                   file_name=f"{Path(selected_file).stem}_comfort_monthly.csv",
                                   mime="text/csv")
    
                st.caption("Adaptive band shown against dry-bulb temperature.")
                x = df.index; y = num(df[dry_col])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=hi_h, name="Adaptive upper", mode="lines", line=dict(width=0.5)))
                fig.add_trace(go.Scatter(x=x, y=lo_h, name="Adaptive lower", mode="lines", fill="tonexty", line=dict(width=0.5)))
                fig.add_trace(go.Scatter(x=x, y=y, name="Dry bulb", mode="lines"))
                fig.update_layout(title="Dry bulb vs Adaptive Comfort Band", xaxis_title="Time", yaxis_title="°C")
                st.plotly_chart(fig, width="stretch")

            else:
                st.warning("Dry bulb temperature not found; cannot compute adaptive comfort.")

# =====================================================================================
# COMPARISON TAB (unchanged from previous adaptive version)
# =====================================================================================
with tab_compare:
    st.subheader("Multi-file comparisson")
    with st.expander("Selection",expanded=True):
                multi_select = st.multiselect("Select EPW files", options=epw_names, default=epw_names[:2], max_selections=10, key="cmp_files")
                if len(multi_select) == 0:
                    st.info("Pick at least one EPW file to compare."); st.stop()
    
        
        parsed = []
        for name in multi_select:
            try:
                df, loc, stats = _parse_selected_from_catalog(name, _catalog)
                label = label_from_location(loc, Path(name).stem)
                parsed.append((label, name, df, loc, stats))
            except Exception as e:
                st.warning(f"Skipping {name}: {e}")
        if not parsed:
                    st.error("No valid EPWs parsed."); st.stop()
            
                # Fix colors for each file selected
                labels = [p[0] for p in parsed]
                color_map = make_color_map(labels)
        
                # Global date range
                global_min = max(p[2].index.min().date() for p in parsed)
                global_max = min(p[2].index.max().date() for p in parsed)
                date_range = st.date_input("Date range (applied to all)", value=(global_min, global_max), min_value=global_min, max_value=global_max, key="cmp_dates")
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = global_min, date_range
    
    with st.container(border=True):
        st.markdown("## Timeseries Comparison")
        base_cols = list(parsed[0][2].columns)
        sel_col = st.selectbox("Select Column to compare", base_cols, index=0, key="cmp_col")
        
        # KPIs
        with st.container(border=True):
            st.markdown("### Summary Statistics")
            kpi_rows = []
            for label, name, df, loc, stats in parsed:
                dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
                if sel_col in dff.columns:
                    t = num(dff[sel_col])
                    t_mean, t_min, t_max, t_std, t_count = t.mean(), t.min(), t.max(), t.std(), t.count()
                else:
                    t_mean = t_min = t_max = t_std = t_count = np.nan
                kpi_rows.append({"Label": label, "Mean": t_mean, "Min": t_min, "Max": t_max, "Std": t_std,"Count": t_count})
            kpi_df = pd.DataFrame(kpi_rows)
            st.dataframe(kpi_df, width="stretch")
            
        # Temperature time series
        with st.container(border=True):
            st.markdown("### Line Plot")
            fig = go.Figure()
            for label, name, df, loc, _ in parsed:
                dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
                if sel_col in dff.columns:
                    fig.add_trace(go.Scatter(x=dff.index, y=num(dff[sel_col]), mode="lines", name=label, line=dict(color=color_map[label])))
            fig.update_layout(xaxis_title="Time", yaxis_title=sel_col)
            st.plotly_chart(fig, width="stretch")

        # Temperature histogram overlay
        with st.container(border=True):
            st.markdown("### Histogram")
            bins = st.slider("Bins", min_value=10, max_value=100, value=40, step=5, key="temp_bins_cmp")
            fig = go.Figure()
            for label, name, df, loc, _ in parsed:
                dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
                if sel_col in dff.columns:
                    fig.add_trace(go.Histogram(x=num(dff[sel_col]).dropna(), nbinsx=bins, name=label, opacity=0.5, marker_color=color_map[label]))
            fig.update_layout(barmode="overlay", xaxis_title=sel_col, yaxis_title="Hours")
            st.plotly_chart(fig, width="stretch")
        
    with st.container(border=True):
        st.markdown("## Time Aggregated Comparison — Box plots")
    
        # Pick how to aggregate the x-axis buckets
        agg_choice = st.selectbox(
            "Aggregate by",
            ["Hour", "Day of week", "Month"],  # change labels if you prefer
            index=0,
            key="cmp_box_agg"
        )
    
        # Build a tidy table: one row per point, with a 'bucket' column for the x-axis
        frames = []
        for label, name, df_i, loc, _ in parsed:
            dff = df_i.loc[(df_i.index.date >= start_date) & (df_i.index.date <= end_date)]
            if sel_col in dff.columns:
                s = pd.to_numeric(dff[sel_col], errors="coerce").dropna()
                if s.empty:
                    continue
    
                if agg_choice == "Hour":
                    bucket = s.index.hour
                    bucket_name = "hour"
                    cat_order = list(range(24))  # 0..23
                    bucket_display = bucket  # integers are fine
                elif agg_choice == "Day of week":
                    # 0=Mon .. 6=Sun
                    names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    bucket = s.index.dayofweek
                    bucket_name = "day"
                    cat_order = names
                    bucket_display = pd.Categorical.from_codes(bucket, names, ordered=True)
                else:  # "Month"
                    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                    bucket = s.index.month - 1  # 0..11
                    bucket_name = "month"
                    cat_order = names
                    bucket_display = pd.Categorical.from_codes(bucket, names, ordered=True)
    
                frames.append(pd.DataFrame({
                    "bucket": bucket_display,
                    "value": s.values,
                    "Type": label,  # must match keys in color_map
                }))
    
        if not frames:
            st.info("No data available for the selected variable across the chosen files/date range.")
        else:
            plot_df = pd.concat(frames, ignore_index=True)
    
            fig = px.box(
                plot_df,
                x="bucket",
                y="value",
                color="Type",                       # one series per file
                points="outliers",
                category_orders={"bucket": cat_order},
                color_discrete_map=color_map        # keep colors coherent
            )
            # Axes and title
            y_unit = "(-)"  # replace with your unit if you have it, e.g. "°C"
            fig.update_layout(
                xaxis_title=agg_choice,
                yaxis_title=f"{sel_col} {y_unit}",
                boxmode="group",
                height=450,
                title=f"Distribution by {agg_choice} — Head-to-Head"
            )
            # Nice ticks for hour
            if agg_choice == "Hour":
                fig.update_xaxes(tickmode="linear", dtick=1)
    
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("## Monthly HDD and CDD")
        
        st.markdown("#### Select HDD and CDD calculation base")
        # If you already defined these elsewhere, remove these two inputs and reuse
        kc1, kc2 = st.columns(2)
        with kc1:
            base_hdd_kpi = st.number_input("HDD base (°C)", value=18.0, step=0.5, key="kpi_base_hdd")
        with kc2:
            base_cdd_kpi = st.number_input("CDD base (°C)", value=22.0, step=0.5, key="kpi_base_cdd")
        
        with st.container(border=True):
            st.markdown("### KPI — Total degree-days (HDD / CDD)")
        
 
        
            kpi_rows = []
            for label, name, df_i, loc, _ in parsed:
                # Filter by the same global date range
                dff = df_i.loc[(df_i.index.date >= start_date) & (df_i.index.date <= end_date)]
        
                if sel_col in dff.columns:
                    # Use your existing function: returns DAILY degree-days
                    hdd_daily = degree_days(dff[sel_col], base_hdd_kpi, kind="HDD")
                    cdd_daily = degree_days(dff[sel_col], base_cdd_kpi, kind="CDD")
                    hdd_total = float(hdd_daily.sum()) if not hdd_daily.empty else float("nan")
                    cdd_total = float(cdd_daily.sum()) if not cdd_daily.empty else float("nan")
                else:
                    hdd_total = cdd_total = float("nan")
        
                kpi_rows.append({
                    "Label": label,
                    "HDD (°C·day)": hdd_total,
                    "CDD (°C·day)": cdd_total,
                })
            kpi_df = pd.DataFrame(kpi_rows)
                
            
            # Display table
            st.dataframe(
                kpi_df.round({"HDD (°C·day)": 1, "CDD (°C·day)": 1}),
                width="stretch"
            )
            
        st.markdown("### Bar chart — Monthly view (HDD / CDD)")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        
        fig_dd = make_subplots(
            rows=1, cols=2,
            subplot_titles=("HDD by month", "CDD by month"),
            shared_yaxes=False
        )
        
        for label, name, df_i, loc, _ in parsed:
            dff = df_i.loc[(df_i.index.date >= start_date) & (df_i.index.date <= end_date)]
            if sel_col not in dff.columns:
                continue
        
            hdd_m = monthly_degree_days(dff[sel_col], base_hdd, "HDD")
            cdd_m = monthly_degree_days(dff[sel_col], base_cdd, "CDD")
        
            fig_dd.add_trace(
                go.Bar(
                    x=month_names,
                    y=hdd_m.values,
                    name=label,
                    marker_color=color_map.get(label),
                    hovertemplate="%{x}<br>HDD: %{y:.1f} °C·day<extra></extra>",
                ),
                row=1, col=1
            )
            fig_dd.add_trace(
                go.Bar(
                    x=month_names,
                    y=cdd_m.values,
                    name=label,
                    marker_color=color_map.get(label),
                    hovertemplate="%{x}<br>CDD: %{y:.1f} °C·day<extra></extra>",
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig_dd.update_layout(
            barmode="group",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        
        
    # ======================= COMFORT (Compare, monthly bars) =======================
    with st.container(border=True):
        st.markdown("## Comfort comparison — monthly")
    
        # --- Auto-detect columns from the first dataframe ---
        def find_col(df, *needles):
            cols = list(df.columns); low = [c.lower() for c in cols]
            for i, c in enumerate(low):
                if all(n in c for n in needles):
                    return cols[i]
            return None
    
        base_df = parsed[0][2]
        dry_col = (find_col(base_df, "dry", "bulb")
                   or find_col(base_df, "drybulb")
                   or find_col(base_df, "dbt")
                   or "dry_bulb_temperature")
        dew_col = (find_col(base_df, "dew", "point")
                   or "dew_point_temperature")
        rh_col  = (find_col(base_df, "relative", "humidity")
                   or find_col(base_df, "rh")
                   or "relative_humidity")
    
        # Choose which temperature series to evaluate comfort against
        temp_options = []
        if dry_col in base_df.columns: temp_options.append(("Dry bulb (°C)", dry_col))
        if dew_col in base_df.columns: temp_options.append(("Dew point (°C)", dew_col))
        if not temp_options:
            temp_options = [("Dry bulb (°C)", dry_col)]
    
        label_to_col = {lbl: col for lbl, col in temp_options}
        sel_temp_label = st.selectbox(
            "Temperature series for comfort evaluation",
            options=[lbl for lbl, _ in temp_options],
            index=0,
            key="cmp_comfort_temp_sel"
        )
        temp_col = label_to_col[sel_temp_label]
    
        mode = st.radio("Model", ["Fixed band", "Adaptive (ASHRAE 55)"],
                        horizontal=True, key="cmp_comfort_mode_monthly")
    
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        import pandas as pd
    
        # Helper: monthly % comfortable from a boolean hourly Series
        def monthly_percent(mask: pd.Series) -> pd.Series:
            if mask is None or mask.empty:
                return pd.Series([np.nan]*12, index=range(1,13))
            mask = mask.dropna()
            by_m = mask.groupby(mask.index.month).mean() * 100.0
            return by_m.reindex(range(1,13), fill_value=np.nan)
    
        # Build one figure with monthly bars (one trace per location)
        fig = go.Figure()
    
        if mode == "Fixed band":
            c1, c2, c3, c4 = st.columns(4)
            with c1: tmin = st.number_input("Comfort Tmin (°C)", value=18.0, step=0.5, key="cmp_cb_tmin_m")
            with c2: tmax = st.number_input("Comfort Tmax (°C)", value=28.0, step=0.5, key="cmp_cb_tmax_m")
            with c3: rhmin = st.number_input("RH min (%)", value=20.0, step=1.0, key="cmp_cb_rhmin_m")
            with c4: rhmax = st.number_input("RH max (%)", value=80.0, step=1.0, key="cmp_cb_rhmax_m")
    
            for label, name, dfi, loc, _ in parsed:
                dff = dfi.loc[(dfi.index.date >= start_date) & (dfi.index.date <= end_date)]
                if temp_col not in dff.columns:
                    continue
                T  = dff[temp_col]
                RH = dff[rh_col] if rh_col in dff.columns else None
                mask = comfort_mask_fixed(T, RH, tmin, tmax, rhmin=rhmin, rhmax=rhmax)
                m_pct = monthly_percent(mask)
                fig.add_bar(
                    x=month_names, y=m_pct.values, name=label,
                    marker_color=color_map.get(label, None)
                )
    
            fig.update_layout(
                barmode="group",
                title=f"Monthly comfort (%) — Fixed band on {sel_temp_label}",
                yaxis_title="Comfort hours (%)",
                xaxis_title="Month",
                yaxis_range=[0, 100],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    
        else:
            c1, c2, c3 = st.columns(3)
            with c1: accept = st.selectbox("Acceptability", ["80", "90"], index=0, key="cmp_adp_acc_m")
            with c2: alpha  = st.number_input("Running-mean α (0–0.99)", value=0.8, min_value=0.0, max_value=0.99, step=0.05, key="cmp_adp_alpha_m")
    
            # Which outdoor series feeds the running mean
            out_opts = []
            if dry_col in base_df.columns: out_opts.append(("Outdoor: Dry bulb (°C)", dry_col))
            if dew_col in base_df.columns: out_opts.append(("Outdoor: Dew point (°C)", dew_col))
            if not out_opts: out_opts = [("Outdoor: Dry bulb (°C)", dry_col)]
            out_map = {lbl: col for lbl, col in out_opts}
            with c3:
                sel_out_label = st.selectbox("Outdoor series for running mean", [lbl for lbl, _ in out_opts],
                                             index=0, key="cmp_adp_outcol_m")
            out_col = out_map[sel_out_label]
    
            for label, name, dfi, loc, _ in parsed:
                dff = dfi.loc[(dfi.index.date >= start_date) & (dfi.index.date <= end_date)]
                if temp_col not in dff.columns:
                    continue
    
                # Daily mean of chosen outdoor series (fallback to temp_col)
                if out_col in dff.columns:
                    daily_out = pd.to_numeric(dff[out_col], errors="coerce").resample("D").mean()
                else:
                    daily_out = pd.to_numeric(dff[temp_col], errors="coerce").resample("D").mean()
    
                band = adaptive_band_from_outdoor(daily_out, alpha=alpha, acceptability=accept)
                ok_mask, _, _ = adaptive_mask_hourly(dff[temp_col], band)  # boolean per hour
                m_pct = monthly_percent(ok_mask)
                fig.add_bar(
                    x=month_names, y=m_pct.values, name=label,
                    marker_color=color_map.get(label, None)
                )
    
            fig.update_layout(
                barmode="group",
                title=f"Monthly comfort (%) — Adaptive {accept}% on {sel_temp_label}",
                yaxis_title="Comfort hours (%)",
                xaxis_title="Month",
                yaxis_range=[0, 100],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ===== Global comfort KPI (annual) =====
        with st.container(border=True):
            st.markdown("### Global comfort KPI (annual)")
        
            rows = []
        
            for label, name, dfi, loc, _ in parsed:
                # restrict to the same global date window
                dff = dfi.loc[(dfi.index.date >= start_date) & (dfi.index.date <= end_date)]
        
                # skip if temp column is missing
                if temp_col not in dff.columns:
                    rows.append({"Label": label, "Comfort %": float("nan")})
                    continue
        
                if mode == "Fixed band":
                    # Uses your helper (RH optional)
                    T  = dff[temp_col]
                    RH = dff[rh_col] if rh_col in dff.columns else None
                    ok = comfort_mask_fixed(T, RH, tmin, tmax, rhmin=rhmin, rhmax=rhmax)
        
                else:  # Adaptive (ASHRAE 55)
                    # Build daily band from the chosen outdoor series (fallback to temp_col)
                    if out_col in dff.columns:
                        daily_out = pd.to_numeric(dff[out_col], errors="coerce").resample("D").mean()
                    else:
                        daily_out = pd.to_numeric(dff[temp_col], errors="coerce").resample("D").mean()
        
                    band = adaptive_band_from_outdoor(daily_mean_outdoor=daily_out, alpha=alpha, acceptability=accept)
                    ok, _, _ = adaptive_mask_hourly(dff[temp_col], band)  # boolean per hour
        
                pct = float(ok.mean() * 100.0) if ok.size else float("nan")
                rows.append({"Label": label, "Comfort %": pct})
        
            kpi_global_df = pd.DataFrame(rows).sort_values("Comfort %", ascending=False, na_position="last")
            st.dataframe(kpi_global_df.round({"Comfort %": 1}), width="stretch")


    

st.caption("Built with Streamlit + Ladybug + Plotly — Pro + Adaptive v2")

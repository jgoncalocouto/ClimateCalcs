# epw_dashboard_pro_adaptive_v2.py
# Streamlit + Plotly EPW Explorer — Pro + Adaptive (v2 tweaks for Single tab)
#
# Run: streamlit run epw_dashboard_pro_adaptive_v2.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from fcns import *

st.set_page_config(page_title="EnergyPlus Weather File Explorer", layout="wide")
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






# --------------- Load files ---------------
epw_paths = available_epws()
epw_names = [p.name for p in epw_paths]

st.title("🌤️ EnergyPLus File Explorer")
st.caption("Analyzes EPW files existent in a local database.")

tab_single, tab_compare = st.tabs(["📁 Deep Dive analysis", "🆚 Head to Head comparisson"])

# =====================================================================================
# SINGLE FILE TAB
# =====================================================================================
with tab_single:
    st.subheader("Deep Dive Analysis")
    with st.expander("Selection", expanded=True):
    
        selected_file = st.selectbox("Select EPW file", epw_names, index=0)
        epw_path = DATA_DIR / selected_file
        
        with st.spinner("Parsing EPW..."):
            try:
                df, location_dict, df_stats = parse_one(epw_path)
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
    st.markdown("### Download timeseries (filtered range)")
    with st.expander("📥 Download data", expanded=False):
        st.download_button("Download filtered timeseries (CSV)",
                           df.to_csv().encode("utf-8"),
                           file_name=f"{Path(selected_file).stem}_timeseries.csv",
                           mime="text/csv")
        try:
            import io
            buf = io.BytesIO(); df.to_parquet(buf, index=True)
            st.download_button("Download filtered timeseries (Parquet)",
                               data=buf.getvalue(),
                               file_name=f"{Path(selected_file).stem}_timeseries.parquet",
                               mime="application/octet-stream")
        except Exception:
            st.caption("Parquet export unavailable (install pyarrow if needed).")
    
    
    ## KPIs for Dry & Wet bulb
    st.markdown("### Key statistics (filtered range)")
    
    available_cols = list(df.columns)
    dry_col = "dry_bulb_temperature" if "dry_bulb_temperature" in available_cols else None
    dew_col = "dew_point_temperature" if "dew_point_temperature" in available_cols else None
    rh_col = next((c for c in available_cols if "relative_humidity" in c), None)
    ghi_col = next((c for c in available_cols if "global_horizontal_radiation" in c), None)
    ws_col  = next((c for c in available_cols if "wind_speed" in c), None)
    wd_col  = next((c for c in available_cols if "wind_direction" in c), None)
    In_col  = next((c for c in available_cols if "direct_normal_radiation" in c), None)
    Ig_col  = next((c for c in available_cols if "global_horizontal_radiation" in c), None)
       
    kpi_cols = st.columns(6)
    if dry_col:
        v = num(df[dry_col])
        with kpi_cols[0]: st.metric("DBT Mean", fmt(v.mean(), " °C"))
        with kpi_cols[1]: st.metric("DBT Min", fmt(v.min(), " °C"))
        with kpi_cols[2]: st.metric("DBT Max", fmt(v.max(), " °C"))
    else:
        with kpi_cols[0]: st.metric("DBT Mean", "—")
        with kpi_cols[1]: st.metric("DBT Min", "—")
        with kpi_cols[2]: st.metric("DBT Max", "—")

    if dew_col and dew_col in df.columns:
        w = num(df[dew_col])
        with kpi_cols[3]: st.metric("DewPoint Mean", fmt(w.mean(), " °C"))
        with kpi_cols[4]: st.metric("DewPoint Min", fmt(w.min(), " °C"))
        with kpi_cols[5]: st.metric("DewPoint Max", fmt(w.max(), " °C"))
    else:
        with kpi_cols[3]: st.metric("DewPoint Mean", "—")
        with kpi_cols[4]: st.metric("DewPoint Min", "—")
        with kpi_cols[5]: st.metric("DewPoint Max", "—")
        
    kpi_cols = st.columns(6)
    if ws_col:
        v = num(df[ws_col])
        with kpi_cols[0]: st.metric("Wind Speed Mean", fmt(v.mean(), " m/s"))
        with kpi_cols[1]: st.metric("Wind Speed Min", fmt(v.min(), " m/s"))
        with kpi_cols[2]: st.metric("Wind Speed Max", fmt(v.max(), " m/s"))
    else:
        with kpi_cols[0]: st.metric("Wind Speed Mean", "—")
        with kpi_cols[1]: st.metric("Wind Speed Min", "—")
        with kpi_cols[2]: st.metric("Wind Speed Max", "—")

    if wd_col and wd_col in df.columns:
        w = num(df[wd_col])
        with kpi_cols[3]: st.metric("Wind Direction Mean", fmt(w.mean(), " °"))
        with kpi_cols[4]: st.metric("Wind Direction Min", fmt(w.min(), " °"))
        with kpi_cols[5]: st.metric("Wind Direction Max", fmt(w.max(), " °"))
    else:
        with kpi_cols[3]: st.metric("Wind Direction Mean", "—")
        with kpi_cols[4]: st.metric("Wind Direction Min", "—")
        with kpi_cols[5]: st.metric("Wind Direction Max", "—")
        
    kpi_cols = st.columns(6)
    if In_col:
        v = num(df[In_col])
        with kpi_cols[0]: st.metric("Direct Normal Radiation Mean", fmt(v.mean(), " Wh/m^2"))
        with kpi_cols[1]: st.metric("Direct Normal Radiation Min", fmt(v.min(), " Wh/m^2"))
        with kpi_cols[2]: st.metric("Direct Normal Radiation Max", fmt(v.max(), " Wh/m^2"))
    else:
        with kpi_cols[0]: st.metric("Direct Normal Radiation Mean", "—")
        with kpi_cols[1]: st.metric("Direct Normal Radiation Min", "—")
        with kpi_cols[2]: st.metric("Direct Normal Radiation Max", "—")

    if Ig_col and Ig_col in df.columns:
        w = num(df[Ig_col])
        with kpi_cols[3]: st.metric("Global Horizontal Radiation Mean", fmt(w.mean(), " Wh/m^2"))
        with kpi_cols[4]: st.metric("Global Horizontal Radiation Min", fmt(w.min(), " Wh/m^2"))
        with kpi_cols[5]: st.metric("Global Horizontal Radiation Max", fmt(w.max(), " Wh/m^2"))
    else:
        with kpi_cols[3]: st.metric("Global Horizontal Radiation Mean", "—")
        with kpi_cols[4]: st.metric("Global Horizontal Radiation Min", "—")
        with kpi_cols[5]: st.metric("Global Horizontal Radiation Max", "—")


    # ===== Synced 2 cols × 6 rows figure (time-series + histograms) =====
    # Robust column detection for required signals
    def find_col(df, *needles):
        cols = list(df.columns)
        low = [c.lower() for c in cols]
        for i, c in enumerate(low):
            if all(n in c for n in needles):
                return cols[i]
        return None

    # Prefer already detected columns; fall back to fuzzy finder
    dry_c  = dry_col or find_col(df, "dry", "bulb") or find_col(df, "dbt") or "dry_bulb_temperature"
    dew_c  = dew_col or find_col(df, "dew", "point") or "dew_point_temperature"
    rh_c   = rh_col  or find_col(df, "relative", "humidity") or "relative_humidity"
    ws_c   = ws_col  or find_col(df, "wind", "speed") or "wind_speed"
    ghi_c  = ghi_col or find_col(df, "global", "horizontal", "radiation") or find_col(df, "ghi") or "global_horizontal_radiation"
    pr_c   = find_col(df, "liquid", "precipitation") or find_col(df, "precipitation") or "liquid_precipitation_quantity"

    # UI for per-row histogram bin widths
    c1, c2 = st.columns(2)
    with c1:
        bw_dry = st.select_slider("Dry bulb bin width (°C)", options=[1.0, 2.0, 5.0], value=2.0, key="bw_dry")
        bw_dew = st.select_slider("Dew point bin width (°C)", options=[1.0, 2.0, 5.0], value=2.0, key="bw_dew")
        bw_rh  = st.select_slider("Humidity bin width (%)", options=[2.0, 5.0, 10.0], value=5.0, key="bw_rh")
    with c2:
        bw_ws  = st.select_slider("Wind speed bin width (m/s)", options=[0.1, 0.5, 1.0], value=0.5, key="bw_ws")
        bw_ghi = st.select_slider("GHI bin width (W/m²)", options=[25, 50, 100, 200], value=100, key="bw_ghi")
        bw_prc = st.select_slider("Precipitation bin width (mm)", options=[0.2, 0.5, 1.0, 2.0], value=1.0, key="bw_prc")

    binwidths = {"dry": bw_dry, "dew": bw_dew, "rh": bw_rh, "ws": bw_ws, "ghi": bw_ghi, "prec": bw_prc}

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    def _hist_bins(series, width):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return None
        lo = float(np.floor(s.min() / width) * width)
        hi = float(np.ceil(s.max() / width) * width)
        if hi <= lo: hi = lo + width
        return dict(start=lo, end=hi, size=width)

    # Build the 2x6 layout
    rows_def = [
        ("dry", "Dry bulb (°C)", dry_c,  "°C"),
        ("dew", "Dew point (°C)", dew_c,  "°C"),
        ("rh",  "Relative humidity (%)", rh_c,   "%"),
        ("ws",  "Wind speed (m/s)", ws_c,   "m/s"),
        ("ghi", "Global horizontal radiation (W/m²)", ghi_c, "W/m²"),
        ("prec","Liquid precipitation (mm)", pr_c,  "mm"),
    ]

    fig = make_subplots(
        rows=6, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.035,
        column_widths=[0.68, 0.32],
        subplot_titles=[t for _,t,_,_ in rows_def for __ in (0,1)]
    )

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

    st.markdown("### Synchronized dashboard")
    st.plotly_chart(fig, use_container_width=True)

    # ===== Diurnal & Monthly (use Dry bulb) =====
    if dry_col:
        st.markdown("### Diurnal temperature profile (Dry bulb)")
        diurnal_df = diurnal_stats(df[dry_col])
        fig = go.Figure()
        fig.add_scatter(x=diurnal_df["hour"], y=diurnal_df["mean"], 
                        name="Mean", mode="lines+markers")
        fig.add_scatter(x=diurnal_df["hour"], y=diurnal_df["min"], 
                        name="Min", mode="lines+markers")
        fig.add_scatter(x=diurnal_df["hour"], y=diurnal_df["max"], 
                        name="Max", mode="lines+markers")
        fig.update_layout(
            title="Daily temperature (DBT)",
            xaxis_title="Hour (hr)", 
            yaxis_title="Temperature (°C)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("### Monthly climatology (Dry bulb)")
        monthly = monthly_stats(df[dry_col])
        fig = go.Figure()
        fig.add_scatter(x=monthly["month"], y=monthly["mean"], 
                        name="Mean", mode="lines+markers")
        fig.add_scatter(x=monthly["month"], y=monthly["min"], 
                        name="Min", mode="lines+markers")
        fig.add_scatter(x=monthly["month"], y=monthly["max"], 
                        name="Max", mode="lines+markers")
        fig.update_layout(
            title="Monthly temperature (DBT)",
            xaxis_title="Month", 
            yaxis_title="Temperature (°C)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
        # ===== Wind rose speed-binned (kept) =====
        if ws_col and wd_col:
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
            rose_df = wind_rose_speed_binned(df, ws_col, wd_col, speed_bins, n_sectors)
            if not rose_df.empty:
                fig = px.bar_polar(rose_df, r="count", theta="sector", color="speed_bin",
                                   title="Wind rose (by speed bins)")
                st.plotly_chart(fig, use_container_width=True)
    
        # ===== Advanced analytics =====
        st.markdown("## Advanced analytics")
    
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
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dry bulb temperature not found; cannot compute degree-days.")
    
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
            st.plotly_chart(fig, use_container_width=True)
    
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
                st.plotly_chart(fig, use_container_width=True)
    
                st.caption("Adaptive band shown against dry-bulb temperature.")
                x = df.index; y = num(df[dry_col])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=hi_h, name="Adaptive upper", mode="lines", line=dict(width=0.5)))
                fig.add_trace(go.Scatter(x=x, y=lo_h, name="Adaptive lower", mode="lines", fill="tonexty", line=dict(width=0.5)))
                fig.add_trace(go.Scatter(x=x, y=y, name="Dry bulb", mode="lines"))
                fig.update_layout(title="Dry bulb vs Adaptive Comfort Band", xaxis_title="Time", yaxis_title="°C")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dry bulb temperature not found; cannot compute adaptive comfort.")
    
        # Temperature bins heatmap (Dry bulb) with inverted greyscale (0=white, max=dark)
        if dry_col:
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
                st.plotly_chart(fig, use_container_width=True)

# =====================================================================================
# COMPARISON TAB (unchanged from previous adaptive version)
# =====================================================================================
with tab_compare:
    st.subheader("Compare EPWs (up to 5)")
    multi_select = st.multiselect("Select EPW files", options=epw_names, default=epw_names[:2], max_selections=5, key="cmp_files")
    if len(multi_select) == 0:
        st.info("Pick at least one EPW file to compare."); st.stop()

    parsed = []
    for name in multi_select:
        path = DATA_DIR / name
        try:
            df, loc, stats = parse_one(path)
            label = label_from_location(loc, Path(name).stem)
            parsed.append((label, name, df, loc, stats))
        except Exception as e:
            st.warning(f"Skipping {name}: {e}")

    if not parsed:
        st.error("No valid EPWs parsed."); st.stop()

    labels = [p[0] for p in parsed]
    color_map = make_color_map(labels)

    base_cols = list(parsed[0][2].columns)
    default_temp = "dry_bulb_temperature" if "dry_bulb_temperature" in base_cols else base_cols[0]
    temp_candidates = [c for c in base_cols if "temp" in c or c == default_temp] or base_cols

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_temp_c = st.selectbox("Temperature column (°C)", temp_candidates, index=0, key="cmp_temp")
    with c2:
        sel_rh_c = st.selectbox("Relative humidity column (%)", [c for c in base_cols if "relative_humidity" in c] or ["(none)"], index=0, key="cmp_rh")
    with c3:
        sel_ghi_c = st.selectbox("Global horiz. radiation", [c for c in base_cols if "global_horizontal_radiation" in c] or ["(none)"], index=0, key="cmp_ghi")

    c4, c5 = st.columns(2)
    with c4:
        sel_ws_c = st.selectbox("Wind speed column (m/s)", [c for c in base_cols if "wind_speed" in c] or ["(none)"], index=0, key="cmp_ws")
    with c5:
        sel_wd_c = st.selectbox("Wind direction column (deg)", [c for c in base_cols if "wind_direction" in c] or ["(none)"], index=0, key="cmp_wd")

    # Comfort model selection for comparison
    st.markdown("#### Comfort model for comparison")
    mode_cmp = st.radio("Model", ["Fixed band", "Adaptive (ASHRAE 55)"], horizontal=True, key="comfort_mode_cmp")
    if mode_cmp == "Fixed band":
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: tmin_c = st.number_input("Comfort Tmin (°C)", value=20.0, step=0.5, key="cmp_tmin")
        with cc2: tmax_c = st.number_input("Comfort Tmax (°C)", value=26.0, step=0.5, key="cmp_tmax")
        with cc3: rhmin_c = st.number_input("Comfort RH min (%)", value=30.0, step=1.0, key="cmp_rhmin")
        with cc4: rhmax_c = st.number_input("Comfort RH max (%)", value=60.0, step=1.0, key="cmp_rhmax")
    else:
        cc1, cc2, cc3 = st.columns(3)
        with cc1: accept_c = st.selectbox("Acceptability", ["80", "90"], index=0, key="cmp_accept")
        with cc2: alpha_c = st.number_input("Running-mean alpha (0–1)", value=0.8, min_value=0.0, max_value=0.99, step=0.05, key="cmp_alpha")
        with cc3: outdoor_col_c = st.selectbox("Outdoor temp for running mean", [sel_temp_c] + [c for c in base_cols if c != sel_temp_c], index=0, key="cmp_out_col")

    # Global date range
    global_min = max(p[2].index.min().date() for p in parsed)
    global_max = min(p[2].index.max().date() for p in parsed)
    date_range = st.date_input("Date range (applied to all)", value=(global_min, global_max), min_value=global_min, max_value=global_max, key="cmp_dates")
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = global_min, date_range

    # KPIs
    st.markdown("### KPIs per file (filtered range)")
    kpi_rows = []
    for label, name, df, loc, stats in parsed:
        dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        if sel_temp_c in dff.columns:
            t = num(dff[sel_temp_c])
            t_mean, t_min, t_max = t.mean(), t.min(), t.max()
        else:
            t_mean = t_min = t_max = np.nan
        if sel_ghi_c in dff.columns and sel_ghi_c != "(none)":
            daily = compute_daily_ghi(dff, sel_ghi_c)
            ghi_avg = daily["kWh/m²·day"].mean()
        else:
            ghi_avg = np.nan
        kpi_rows.append({"Label": label, "Mean temp (°C)": t_mean, "Min temp (°C)": t_min, "Max temp (°C)": t_max, "Avg daily GHI (kWh/m²·day)": ghi_avg})
    kpi_df = pd.DataFrame(kpi_rows)
    st.dataframe(kpi_df.style.format({
        "Mean temp (°C)": "{:.2f}", "Min temp (°C)": "{:.2f}", "Max temp (°C)": "{:.2f}", "Avg daily GHI (kWh/m²·day)": "{:.2f}",
    }), use_container_width=True)

    # Temperature time series
    st.markdown("### Temperature time series")
    fig = go.Figure()
    for label, name, df, loc, _ in parsed:
        dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        if sel_temp_c in dff.columns:
            fig.add_trace(go.Scatter(x=dff.index, y=num(dff[sel_temp_c]), mode="lines", name=label, line=dict(color=color_map[label])))
    fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)

    # Temperature histogram overlay
    st.markdown("### Temperature histogram (overlay)")
    bins = st.slider("Bins", min_value=10, max_value=100, value=40, step=5, key="temp_bins_cmp")
    fig = go.Figure()
    for label, name, df, loc, _ in parsed:
        dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        if sel_temp_c in dff.columns:
            fig.add_trace(go.Histogram(x=num(dff[sel_temp_c]).dropna(), nbinsx=bins, name=label, opacity=0.5, marker_color=color_map[label]))
    fig.update_layout(barmode="overlay", xaxis_title="Temperature (°C)", yaxis_title="Hours")
    st.plotly_chart(fig, use_container_width=True)

    # Comfort comparison
    st.markdown("### Comfort comparison")
    if mode_cmp == "Fixed band":
        rows = []
        for label, name, df, loc, _ in parsed:
            dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
            rh_s = dff[sel_rh_c] if (sel_rh_c in dff.columns and sel_rh_c != "(none)") else None
            cm = comfort_mask_fixed(dff[sel_temp_c], rh_s, tmin_c, tmax_c, rhmin_c if rh_s is not None else None, rhmax_c if rh_s is not None else None)
            pct = (100.0 * cm.mean()) if cm.size else 0.0
            rows.append({"Label": label, "Comfort %": pct})
        cmp_df = pd.DataFrame(rows)
        fig = px.bar(cmp_df, x="Label", y="Comfort %", color="Label", color_discrete_map=color_map, title="Comfort hours (%) by file — Fixed band")
        fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)
    else:
        rows = []
        for label, name, df, loc, _ in parsed:
            dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
            out_col = outdoor_col_c if outdoor_col_c in dff.columns else sel_temp_c
            daily_out = num(dff[out_col]).resample("D").mean()
            band = adaptive_band_from_outdoor(daily_out, alpha=alpha_c, acceptability=accept_c)
            cm, _, _ = adaptive_mask_hourly(dff[sel_temp_c], band)
            pct = (100.0 * cm.mean()) if cm.size else 0.0
            rows.append({"Label": label, "Comfort %": pct})
        cmp_df = pd.DataFrame(rows)
        fig = px.bar(cmp_df, x="Label", y="Comfort %", color="Label", color_discrete_map=color_map, title=f"Comfort hours (%) by file — Adaptive {accept_c}%")
        fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)

    # Degree-days comparison (unchanged)
    st.markdown("### Degree-days comparison")
    c1, c2 = st.columns(2)
    with c1: base_hdd_c = st.number_input("HDD base (°C)", value=18.0, step=0.5, key="cmp_hdd_base")
    with c2: base_cdd_c = st.number_input("CDD base (°C)", value=22.0, step=0.5, key="cmp_cdd_base")
    rows = []
    for label, name, df, loc, _ in parsed:
        dff = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        if sel_temp_c in dff.columns:
            hdd = degree_days(dff[sel_temp_c], base_hdd_c, "HDD").sum()
            cdd = degree_days(dff[sel_temp_c], base_cdd_c, "CDD").sum()
        else:
            hdd = cdd = np.nan
        rows.append({"Label": label, "HDD": hdd, "CDD": cdd})
    dd_df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(dd_df, x="Label", y="HDD", color="Label", color_discrete_map=color_map, title="Total HDD (°C·day)")
        fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(dd_df, x="Label", y="CDD", color="Label", color_discrete_map=color_map, title="Total CDD (°C·day)")
        fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)
    
    st.caption("End here")

st.caption("Built with Streamlit + Ladybug + Plotly — Pro + Adaptive v2")

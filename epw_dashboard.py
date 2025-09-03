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
from fcns import *
from plotly.subplots import make_subplots

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




# --------------- Load files ---------------
epw_paths = available_epws()
epw_names = [p.name for p in epw_paths]

st.title("🌤️ EnergyPLus File Explorer")
st.caption("Analyzes EPW files existent in a local database.")
st.subheader("Select analysis type")

tab_single, tab_compare = st.tabs(["📁 Deep Dive analysis", "🆚 Head to Head comparisson"])

# =====================================================================================
# SINGLE FILE TAB
# =====================================================================================
with tab_single:
    
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

    # ===== Diurnal & Monthly (use Dry bulb) =====
    with st.container(border=True):
        # Temperature bins heatmap (Dry bulb) with inverted greyscale (0=white, max=dark)
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

        with st.container(border=True):
            
            # Compute stats
            diurnal_dry = diurnal_stats(df[dry_col])
            diurnal_dew = diurnal_stats(df[dew_col])
            # Create 1 row × 2 columns layout with shared x-axis
            fig = make_subplots(rows=1, cols=2, shared_xaxes=True,subplot_titles=("Dry bulb temperature", "Dew point temperature"))
            # --- Dry bulb traces (left) ---
            fig.add_trace(go.Scatter(x=diurnal_dry["hour"], y=diurnal_dry["mean"], name="Mean (Dry)", mode="lines+markers"),row=1, col=1)
            fig.add_trace(go.Scatter(x=diurnal_dry["hour"], y=diurnal_dry["min"], name="Min (Dry)", mode="lines+markers"),row=1, col=1)
            fig.add_trace(go.Scatter(x=diurnal_dry["hour"], y=diurnal_dry["max"], name="Max (Dry)", mode="lines+markers"),row=1, col=1)
            # --- Dew point traces (right) ---
            fig.add_trace(go.Scatter(x=diurnal_dew["hour"], y=diurnal_dew["mean"], name="Mean (Dew)", mode="lines+markers"),row=1, col=2)
            fig.add_trace(go.Scatter(x=diurnal_dew["hour"], y=diurnal_dew["min"], name="Min (Dew)", mode="lines+markers"),row=1, col=2)
            fig.add_trace(go.Scatter(x=diurnal_dew["hour"], y=diurnal_dew["max"], name="Max (Dew)", mode="lines+markers"),row=1, col=2)
            # Layout adjustments
            fig.update_xaxes(title_text="Hour", row=1, col=1)
            fig.update_xaxes(title_text="Hour", row=1, col=2)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
            fig.update_layout(height=400,width=1000,hovermode="x unified",title="Diurnal profiles (Dry bulb vs Dew point)")
            st.plotly_chart(fig, width="stretch")
            
        with st.container(border=True):
            
            # Compute stats
            monthly_dry = monthly_stats(df[dry_col])
            monthly_dew = monthly_stats(df[dew_col])
            # Create 1 row × 2 columns layout with shared x-axis
            fig = make_subplots(rows=1, cols=2, shared_xaxes=True,subplot_titles=("Dry bulb temperature", "Dew point temperature"))
            # --- Dry bulb traces (left) ---
            fig.add_trace(go.Scatter(x=monthly_dry["month"], y=monthly_dry["mean"], name="Mean (Dry)", mode="lines+markers"),row=1, col=1)
            fig.add_trace(go.Scatter(x=monthly_dry["month"], y=monthly_dry["min"], name="Min (Dry)", mode="lines+markers"),row=1, col=1)
            fig.add_trace(go.Scatter(x=monthly_dry["month"], y=monthly_dry["max"], name="Max (Dry)", mode="lines+markers"),row=1, col=1)
            # --- Dew point traces (right) ---
            fig.add_trace(go.Scatter(x=monthly_dew["month"], y=monthly_dew["mean"], name="Mean (Dew)", mode="lines+markers"),row=1, col=2)
            fig.add_trace(go.Scatter(x=monthly_dew["month"], y=monthly_dew["min"], name="Min (Dew)", mode="lines+markers"),row=1, col=2)
            fig.add_trace(go.Scatter(x=monthly_dew["month"], y=monthly_dew["max"], name="Max (Dew)", mode="lines+markers"),row=1, col=2)
            # Layout adjustments
            fig.update_xaxes(title_text="Month", row=1, col=1)
            fig.update_xaxes(title_text="Month", row=1, col=2)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
            fig.update_layout(height=400,width=1000,hovermode="x unified",title="Monthly values (Dry bulb vs Dew point)")
            st.plotly_chart(fig, width="stretch")
            

    
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
            path = DATA_DIR / name
            try:
                df, loc, stats = parse_one(path)
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
            st.markdown("### Summary Statistics",)
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


st.caption("Built with Streamlit + Ladybug + Plotly — Pro + Adaptive v2")

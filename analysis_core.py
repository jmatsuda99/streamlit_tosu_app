
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from pathlib import Path
from datetime import timedelta

def set_jp_font():
    try:
        from matplotlib.font_manager import findSystemFonts, FontProperties
        candidates = ["Noto Sans CJK JP", "NotoSansCJKJP", "NotoSansCJK", "NotoSansJP",
                      "Source Han Sans", "SourceHanSans", "IPAGothic", "VL Gothic", "TakaoGothic"]
        fonts = findSystemFonts(fontext='ttf') + findSystemFonts(fontext='otf')
        chosen = None
        for f in fonts:
            low = f.lower().replace(" ", "")
            if any(c.replace(" ", "").lower() in low for c in candidates):
                chosen = FontProperties(fname=f)
                break
        if chosen:
            matplotlib.rcParams['font.family'] = chosen.get_name()
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="ignore", utc=False, infer_datetime_format=True)
            if is_datetime64_any_dtype(parsed):
                df[col] = parsed
        if is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s) > 0 and (s.between(20000, 60000).mean() > 0.7):
                try:
                    df[col] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df[col].round(0), unit="D")
                except Exception:
                    pass
    return df

def first_datetime_col(df: pd.DataFrame):
    for c in df.columns:
        if is_datetime64_any_dtype(df[c]):
            return c
    return None

def pick_usage_col(df: pd.DataFrame):
    cands = [c for c in df.columns if "使用電力量" in str(c)]
    if cands:
        return cands[0]
    for c in df.columns:
        if is_numeric_dtype(df[c]):
            return c
    return None

def ensure_30min_kW(df: pd.DataFrame, time_col: str, y_col: str) -> pd.DataFrame:
    tmp = df[[time_col, y_col]].dropna().copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col]).set_index(time_col).sort_index()
    ts = tmp[[y_col]].resample("30T").sum()
    ts[y_col] = ts[y_col] * 2.0  # kWh/30分 → kW
    return ts

def plot_timeseries(df_ts: pd.DataFrame, y_col: str, title: str):
    set_jp_font()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df_ts.index, df_ts[y_col])
    ax.set_title(title)
    ax.set_xlabel("時刻")
    ax.set_ylabel("需要電力 [kW]")
    fig.tight_layout()
    return fig

def plot_day(df_ts: pd.DataFrame, y_col: str, day_str: str):
    day_df = df_ts.loc[day_str].dropna()
    return plot_timeseries(day_df.to_frame(), y_col, f"{day_str} の30分推移［kW］")

def peak_day(df_ts: pd.DataFrame, y_col: str):
    daily_peak = df_ts[y_col].resample("D").max()
    day = daily_peak.idxmax().date()
    return str(day)

# --- Forecast models ---
def forecast_lr(ts: pd.DataFrame, y_col: str, target_day: str):
    from sklearn.linear_model import LinearRegression
    df_m = ts.copy()
    df_m["slot"] = ((df_m.index.hour*60 + df_m.index.minute)//30).astype(int)
    last_day = df_m.index.date.max()
    start = last_day - timedelta(days=7)
    train = df_m[(df_m.index.date >= start) & (df_m.index.date <= last_day)]
    X, y = train[["slot"]], train[y_col]
    model = LinearRegression().fit(X, y)
    pred_index = pd.date_range(start=pd.Timestamp(target_day), periods=48, freq="30T")
    Xp = pd.DataFrame({"slot": np.arange(48)}, index=pred_index)
    y_pred = model.predict(Xp[["slot"]])
    pred = pd.Series(y_pred, index=pred_index, name="予測[kW]")
    return pred

def forecast_weekday_slot(ts: pd.DataFrame, y_col: str, target_day: str):
    df_m = ts.copy()
    df_m["slot"] = ((df_m.index.hour*60 + df_m.index.minute)//30).astype(int)
    df_m["weekday"] = df_m.index.weekday
    last_day = df_m.index.date.max()
    start = last_day - timedelta(days=30)
    train = df_m[df_m.index.date >= start]
    pivot = train.groupby(["weekday","slot"])[y_col].mean().unstack(0)
    wd = pd.Timestamp(target_day).weekday()
    pred_vals = pivot[wd].values
    pred_index = pd.date_range(start=pd.Timestamp(target_day), periods=48, freq="30T")
    pred = pd.Series(pred_vals, index=pred_index, name="予測[kW]")
    return pred

def forecast_ml(ts: pd.DataFrame, y_col: str, target_day: str):
    df = ts[[y_col]].copy()
    df["slot"] = ((df.index.hour*60 + df.index.minute)//30).astype(int)
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["lag_1d"] = df[y_col].shift(48)
    df["lag_2d"] = df[y_col].shift(96)
    df["lag_1w"] = df[y_col].shift(48*7)
    df["roll_mean_1d"] = df[y_col].rolling(48).mean()
    df["roll_mean_2d"] = df[y_col].rolling(96).mean()
    df["roll_mean_1w"] = df[y_col].rolling(48*7).mean()
    df = df.dropna()
    feat = ["slot","weekday","month","lag_1d","lag_2d","lag_1w","roll_mean_1d","roll_mean_2d","roll_mean_1w"]
    split = int(len(df)*0.8)
    Xtr, ytr = df.iloc[:split][feat], df.iloc[:split][ycol]
    Xva, yva = df.iloc[split:][feat], df.iloc[split:][ycol]
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9, random_state=42)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        model_name = "LightGBM"
        importance = model.feature_importances_.tolist()
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=42).fit(Xtr, ytr)
        model_name = "GradientBoosting (fallback)"
        importance = None

    ser = ts[y_col]
    lag_1d = ser.iloc[-48:].values
    lag_2d = ser.iloc[-96:-48].values if len(ser) >= 96 else lag_1d
    lag_1w = ser.iloc[-48*7:].values[-48:] if len(ser) >= 48*7 else lag_1d
    roll1 = ser.rolling(48).mean().iloc[-48:].values
    roll2 = ser.rolling(96).mean().iloc[-48:].values if len(ser) >= 96 else roll1
    rollw = ser.rolling(48*7).mean().iloc[-48:].values if len(ser) >= 48*7 else roll1

    pred_index = pd.date_range(start=pd.Timestamp(target_day), periods=48, freq="30T")
    Xp = pd.DataFrame({
        "slot": np.arange(48),
        "weekday": pd.Timestamp(target_day).weekday(),
        "month": pd.Timestamp(target_day).month,
        "lag_1d": lag_1d, "lag_2d": lag_2d, "lag_1w": lag_1w,
        "roll_mean_1d": roll1, "roll_mean_2d": roll2, "roll_mean_1w": rollw
    }, index=pred_index)
    y_pred = model.predict(Xp[feat])
    pred = pd.Series(y_pred, index=pred_index, name=f"予測[kW] {model_name}")
    return pred, model_name, importance

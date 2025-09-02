
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import datetime as _dt
import matplotlib.pyplot as plt

from analysis_core import (
    set_jp_font, try_parse_dates, first_datetime_col, pick_usage_col,
    ensure_30min_kW, plot_timeseries, plot_day, peak_day,
    forecast_lr, forecast_weekday_slot, forecast_ml
)
from db import init_db, add_file, list_files, get_path_by_id

st.set_page_config(page_title="鳥栖 需要分析ツール", layout="wide")
st.title("鳥栖 需要分析ツール（GitHub + Streamlit）")
st.caption("ファイル選択 → DB保持 → 解析。6〜8の学習予測は解析時に選択可能。")

# Sidebar
with st.sidebar:
    st.header("入力ファイル")
    uploaded = st.file_uploader("Excel（.xlsx）をアップロード", type=["xlsx"])
    if uploaded is not None:
        content = uploaded.read()
        path = add_file(uploaded.name, content)
        st.success(f"保存しました：{path}")

    st.divider()
    st.subheader("データベースから選択")
    init_db()
    rows = list_files()
    if len(rows) == 0:
        st.info("DBにファイルがありません。左でアップロードしてください。")
        st.stop()

    labels = {f"[{r[0]}] {r[1]} ({r[3]})": r[0] for r in rows}
    sel_label = st.selectbox("解析対象ファイル", list(labels.keys()))
    file_id = labels[sel_label]
    stored_path = get_path_by_id(file_id)

    st.divider()
    st.header("解析オプション")
    run_step6 = st.checkbox("6) 単純LR予測", value=False)
    run_step7 = st.checkbox("7) 曜日×時刻平均予測", value=False)
    run_step8 = st.checkbox("8) ML予測（LightGBM/GBDT）", value=False)
    target_day = st.date_input("予測ターゲット日", _dt.date.today())

# Main
st.subheader("データの読み込み")
xf = pd.ExcelFile(stored_path)
st.write(f"**シート一覧**: {xf.sheet_names}")
sheet = st.selectbox("解析するシート", xf.sheet_names, index=0)
df = xf.parse(sheet_name=sheet)
df = df.dropna(how="all").dropna(axis=1, how="all")
df = try_parse_dates(df)
st.write("先頭プレビュー", df.head())

tcol_auto = first_datetime_col(df) or df.columns[0]
ycol_auto = pick_usage_col(df) or df.columns[1]
c1, c2 = st.columns(2)
with c1:
    time_col = st.selectbox("時刻列", df.columns.tolist(), index=df.columns.get_loc(tcol_auto))
with c2:
    y_col = st.selectbox("使用電力量(ロス後) 列", df.columns.tolist(), index=df.columns.get_loc(ycol_auto))

ts = ensure_30min_kW(df, time_col, y_col)

st.subheader("基本の可視化")
fig_all = plot_timeseries(ts, y_col, "使用電力量(ロス後)の30分推移［kW換算］")
st.pyplot(fig_all)

# 任意日
st.markdown("**任意日（日内）**")
any_day_default = _dt.date.fromtimestamp(ts.index.min().timestamp())
any_day = st.date_input("表示する日付", any_day_default, key="any_day_input")
fig_day = plot_day(ts, y_col, str(any_day))
st.pyplot(fig_day)

# ピーク日
pday = peak_day(ts, y_col)
st.info(f"最大全ピーク日: **{pday}**")
fig_peak = plot_day(ts, y_col, pday)
st.pyplot(fig_peak)

# 予測（選択実行）
st.subheader("予測（選択したもののみ実行）")
if st.button("予測を実行", type="primary"):
    set_jp_font()  # no-op but keeps unicode minus correct

    if run_step6:
        pred6 = forecast_lr(ts, y_col, str(target_day))
        fig6, ax6 = plt.subplots(figsize=(12, 5))
        ax6.plot(pred6.index, pred6.values)
        ax6.set_title("6) 単純LR 予測")
        ax6.set_xlabel("時刻")
        ax6.set_ylabel("需要電力 [kW]")
        fig6.tight_layout()
        st.pyplot(fig6)
        st.download_button(
            label="CSVダウンロード（6）",
            data=pred6.to_csv().encode("utf-8-sig"),
            file_name=f"forecast_lr_{target_day}.csv",
            mime="text/csv"
        )

    if run_step7:
        pred7 = forecast_weekday_slot(ts, y_col, str(target_day))
        fig7, ax7 = plt.subplots(figsize=(12, 5))
        ax7.plot(pred7.index, pred7.values)
        ax7.set_title("7) 曜日×時刻平均 予測")
        ax7.set_xlabel("時刻")
        ax7.set_ylabel("需要電力 [kW]")
        fig7.tight_layout()
        st.pyplot(fig7)
        st.download_button(
            label="CSVダウンロード（7）",
            data=pred7.to_csv().encode("utf-8-sig"),
            file_name=f"forecast_wdslot_{target_day}.csv",
            mime="text/csv"
        )

    if run_step8:
        pred8, name8, imp = forecast_ml(ts, y_col, str(target_day))
        fig8, ax8 = plt.subplots(figsize=(12, 5))
        ax8.plot(pred8.index, pred8.values)
        ax8.set_title(f"8) ML（{name8}） 予測")
        ax8.set_xlabel("時刻")
        ax8.set_ylabel("需要電力 [kW]")
        fig8.tight_layout()
        st.pyplot(fig8)

        if imp is not None:
            feat_names = ["slot","weekday","month","lag_1d","lag_2d","lag_1w",
                          "roll_mean_1d","roll_mean_2d","roll_mean_1w"]
            imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)

        st.download_button(
            label="CSVダウンロード（8）",
            data=pred8.to_csv().encode("utf-8-sig"),
            file_name=f"forecast_ml_{target_day}.csv",
            mime="text/csv"
        )

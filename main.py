import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io, os, platform, subprocess, sys

# ─────────────────────────────────────────────────────────────
# 한글 폰트 설정 (환경에 따라 자동 탐색)
# ─────────────────────────────────────────────────────────────
def setup_korean_font():
    candidates = [
        "NotoSansCJKkr-Regular.ttf",
        "NotoSansKR-Regular.ttf",
        "NanumGothic.ttf",
        "NanumBarunGothic.ttf",
        "AppleGothic.ttf",
        "Seoul Hangang Pod/Regular.ttc",
    ]
    found = None
    for cand in candidates:
        for root, _, files in os.walk("/"):
            if cand in files:
                found = os.path.join(root, cand)
                break
        if found:
            break
    # Linux 패키지 폰트 경로도 확인
    if not found:
        for cand in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                     "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
                     "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"]:
            if os.path.exists(cand):
                found = cand
                break
    if not found:
        try:
            result = subprocess.run(["fc-list", ":lang=ko"], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if ".ttf" in line or ".otf" in line:
                    parts = line.split(":")
                    if len(parts) >= 1 and os.path.isfile(parts[0].strip()):
                        found = parts[0].strip()
                        break
        except Exception:
            pass
    if found:
        fm.fontMgr.addfont(found)
        prop = fm.FontProperties(fname=found)
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        return prop.get_name()
    return None

font_name = setup_korean_font()
if font_name:
    st.caption(f"💬 사용 중인 한글 폰트: {font_name}")
else:
    st.warning("⚠ 한글 폰트를 찾지 못했습니다. 그래프 라벨이 깨질 수 있습니다.")

# ─────────────────────────────────────────────────────────────
# 통계 용어 짧은 설명 (고등학생용)
# ─────────────────────────────────────────────────────────────
STAT_HINTS = {
    "평균 (mean)": "모든 값을 더해 개수로 나눈 '대표값'입니다.",
    "표준편차 (std)": "값이 평균 주변에 얼마나 퍼져 있는지를 나타냅니다. 클수록 개체 간 차이가 큼.",
    "상관계수 (r)": "두 변수가 함께 움직이는 정도를 −1 ~ +1로 나타냅니다. +1에 가까울수록 함께 증가.",
    "중앙값 (median)": "값을 크기순으로 늘어놓았을 때 정확히 중간에 있는 값.",
    "최대/최소": "데이터 중 가장 큰 값과 가장 작은 값.",
    "개체 수 (N)": "측정한 개체(식물)의 총 개수.",
}

# ─────────────────────────────────────────────────────────────
# 사이드바: 데이터 업로드
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")
st.markdown("*인천아라고등학교 극지식물 프로젝트 — 업로드한 CSV를 바로 시각화*")

st.sidebar.header("📂 데이터 업로드")
env_file  = st.sidebar.file_uploader("환경 데이터 (CSV)", type=["csv"])
grow_file = st.sidebar.file_uploader("생육 결과 데이터 (CSV)", type=["csv"])

# ─────────────────────────────────────────────────────────────
# 인코딩 자동 감지 + 읽기
# ─────────────────────────────────────────────────────────────
def read_csv_auto(file):
    if file is None:
        return None
    raw = file.read()
    file.seek(0)
    # BOM 확인 → utf-8-sig
    if raw[:3] == b"\xef\xbb\xbf":
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    # cp949 시도
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="cp949")
    except UnicodeDecodeError:
        pass
    # utf-8 시도
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except UnicodeDecodeError:
        pass
    # 전부 실패하면 latin-1로 읽고 컬럼명은 유지
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1")

env_df  = read_csv_auto(env_file)
grow_df = read_csv_auto(grow_file)

# ─────────────────────────────────────────────────────────────
# 열 이름 표준화 (학교마다 다른 이름 → 내부 맵핑)
# ─────────────────────────────────────────────────────────────
ENV_COL_MAP = {          # 여러 가능한 이름 → 내부 표준 키
    "time":       ["time", "Time", "일시", "날짜", "date", "datetime", "timestamp"],
    "temperature":["temperature", "temp", "기온", "온도", "temperature(°C)", "temperature_c"],
    "humidity":   ["humidity", "hum", "습도", "humidity(%)", "humidity_percent"],
    "ec":         ["ec", "EC", "EC(mS/cm)", "ec_ms_cm", "전기전도도"],
    "ph":         ["ph", "pH", "pH값", "ph_value", "수소이온농도"],
    "water_temp": ["water_temp", "watertemperature", "수온", "water_temp(°C)"],
    "co2":        ["co2", "CO2", "co2(ppm)", "co2_ppm", "이산화탄소"],
}

GROW_COL_MAP = {
    "개체번호":  ["개체번호", "개체", "id", "ID", "plant_id", "번호"],
    "잎 수(장)":["잎 수(장)", "잎수", "leaf_count", "잎 수", "leaf"],
    "지상부 길이(mm)":["지상부 길이(mm)", "지상부길이", "shoot_length", "지상부 길이", "shoot_len"],
    "지하부길이(mm)":["지하부길이(mm)", "지하부 길이(mm)", "지하부길이", "root_length", "지하부 길이"],
    "생중량(g)":["생중량(g)", "생중량", "fresh_weight", "생 무게", "fw"],
    "지상부질량(g)":["지상부질량(g)", "지상부 질량(g)", "지상부질량", "shoot_mass", "지상부 질량"],
    "지하부질량(g)":["지하부질량(g)", "지하부 질량(g)", "지하부질량", "root_mass", "지하부 질량"],
}

def map_columns(df, col_map):
    if df is None:
        return None
    cols = list(df.columns)
    new_cols = {}
    for std_key, aliases in col_map.items():
        for alias in aliases:
            if alias in cols:
                new_cols[alias] = std_key
                break
    renamed = df.rename(columns=new_cols)
    # 매핑되지 않은 열은 원래 이름 유지
    return renamed, new_cols

env_mapped  = None
env_rename  = {}
grow_mapped = None
grow_rename = {}

if env_df is not None:
    env_mapped, env_rename = map_columns(env_df, ENV_COL_MAP)
if grow_df is not None:
    grow_mapped, grow_rename = map_columns(grow_df, GROW_COL_MAP)

# ─────────────────────────────────────────────────────────────
# 데이터 미리보기
# ─────────────────────────────────────────────────────────────
st.sidebar.subheader("📋 업로드된 파일 미리보기")

if env_mapped is not None:
    st.sidebar.markdown(f"**환경 데이터** — {env_mapped.shape[0]}행 × {env_mapped.shape[1]}열")
    st.sidebar.dataframe(env_mapped.head(3), use_container_width=True)
    env_cols = list(env_mapped.columns)
    st.sidebar.caption(f"실제 열: {env_cols}")
else:
    st.sidebar.info("환경 CSV를 업로드하세요.")

if grow_mapped is not None:
    st.sidebar.markdown(f"**생육 결과 데이터** — {grow_mapped.shape[0]}행 × {grow_mapped.shape[1]}열")
    st.sidebar.dataframe(grow_mapped.head(3), use_container_width=True)
    grow_cols = list(grow_mapped.columns)
    st.sidebar.caption(f"실제 열: {grow_cols}")
else:
    st.sidebar.info("생육 결과 CSV를 업로드하세요.")

# ─────────────────────────────────────────────────────────────
# 환경 데이터 시간 파싱 (포맷 다양함)
# ─────────────────────────────────────────────────────────────
def parse_datetime_col(series):
    if series is None or len(series) == 0:
        return pd.Series(dtype="datetime64[ns]")
    s = series.astype(str).str.strip()
    # 포맷 A: '20260715_0' (날짜_시각번호)
    parsed = pd.to_datetime(s, format="%Y%m%d_%H", errors="coerce")
    if parsed.isna().all():
        # 포맷 B: '2026-07-15 06:00:00'
        parsed = pd.to_datetime(s, errors="coerce")
    if parsed.isna().all():
        # 기타 일반 형식
        parsed = pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
    return parsed

env_time_col = None
if env_mapped is not None and "time" in env_mapped.columns:
    env_time_col = parse_datetime_col(env_mapped["time"])
    env_mapped = env_mapped.assign(_dt=env_time_col)

# ─────────────────────────────────────────────────────────────
# 생육 데이터 분석 (통계)
# ─────────────────────────────────────────────────────────────
grow_stats = None
if grow_mapped is not None and "개체번호" in grow_mapped.columns:
    num_cols = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)",
                "생중량(g)", "지상부질량(g)", "지하부질량(g)"]
    present = [c for c in num_cols if c in grow_mapped.columns]
    grow_stats = grow_mapped[present].describe().T
    # 표준편차 추가
    if "std" not in grow_stats.columns:
        grow_stats["std"] = grow_mapped[present].std()
    grow_stats = grow_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    # 컬럼명 한국어화
    grow_stats.columns = ["개체 수(N)", "평균", "표준편차", "최소", "제1사분위(Q1)", "중앙값", "제3사분위(Q3)", "최대"]

# ─────────────────────────────────────────────────────────────
# 메인 화면
# ─────────────────────────────────────────────────────────────
if env_mapped is None and grow_mapped is None:
    st.info("📁 사이드바에서 환경 데이터와 생육 결과 CSV를 각각 업로드해주세요.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡 환경 데이터 요약",
    "📊 생육 결과 통계",
    "🔗 상관 분석",
    "📈 시계열 보기",
])

# ──────────────────────── Tab 1: 환경 데이터 요약 ────────────────────────
with tab1:
    st.header("🌡 환경 데이터 요약")
    if env_mapped is None:
        st.warning("환경 데이터가 없습니다.")
    else:
        env_present = [k for k in ENV_COL_MAP
                       if any(a in env_mapped.columns for a in ENV_COL_MAP[k])]
        if env_time_col is not None and not env_time_col.isna().all():
            first_ts = env_time_col.min()
            last_ts  = env_time_col.max()
            st.markdown(f"- **측정 기간**: {first_ts.strftime('%Y-%m-%d %H:%M')} ~ {last_ts.strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"- **총 측정 횟수**: {len(env_time_col)}회")
            st.markdown(f"- **시간당 평균 측정 간격**: 약 {round(len(env_time_col) / max(1,(last_ts-first_ts).total_seconds()/3600),1)}회/시간")

        metric_cols = st.columns(4)
        for i, key in enumerate(["temperature", "humidity", "ec", "ph"]):
            real = [a for a in ENV_COL_MAP[key] if a in env_mapped.columns]
            if real:
                col = env_mapped[real[0]]
                valid = pd.to_numeric(col, errors="coerce").dropna()
                if len(valid) > 0:
                    with metric_cols[i % 4]:
                        st.metric(
                            label=f"{key.upper()}  _({STAT_HINTS.get('평균 (mean)','')})" if i==0 else key.upper(),
                            value=round(valid.mean(), 2),
                            delta=None,
                        )
                        st.caption(STAT_HINTS.get("평균 (mean)", ""))

        st.subheader("📉 환경 변수 분포 (히스토그램)")
        env_plot_cols = [k for k in ["temperature","humidity","ec","ph","water_temp","co2"]
                         if any(a in env_mapped.columns for a in ENV_COL_MAP[k])]
        if env_plot_cols:
            fig, axes = plt.subplots(2, 3, figsize=(14, 6))
            axes = axes.flatten()
            for j, key in enumerate(env_plot_cols):
                real = [a for a in ENV_COL_MAP[key] if a in env_mapped.columns][0]
                vals = pd.to_numeric(env_mapped[real], errors="coerce").dropna()
                axes[j].hist(vals, bins=30, color="#4C72B0", alpha=0.7, edgecolor="white")
                axes[j].set_title(f"{key} 분포  (n={len(vals)})")
                axes[j].set_xlabel(key)
                axes[j].set_ylabel("빈도")
                axes[j].axvline(vals.mean(), color="red", linestyle="--",
                                label=f"평균={vals.mean():.2f}")
                axes[j].legend(fontsize=8)
            for ax in axes[len(env_plot_cols):]:
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("환경 데이터에서 수치형 열을 찾지 못했습니다.")

        st.subheader("📋 환경 데이터 전체 보기")
        st.dataframe(env_mapped, use_container_width=True, height=300)

# ──────────────────────── Tab 2: 생육 결과 통계 ────────────────────────
with tab2:
    st.header("📊 생육 결과 통계")
    if grow_mapped is None:
        st.warning("생육 결과 데이터가 없습니다.")
    else:
        st.markdown("아래 표는 각 생육 지표별로 **개체들의 통계 요약**입니다.")
        if grow_stats is not None:
            st.dataframe(grow_stats, use_container_width=True, height=260)
            st.caption("※ 평균 = 대표값 / 표준편차 = 개체 간 차이 정도 / 중앙값 = 중간값")

        st.subheader("🌿 개체별 생육 지표 (막대 그래프)")
        grow_num = [c for c in ["잎 수(장)","지상부 길이(mm)","지하부길이(mm)",
                                 "생중량(g)","지상부질량(g)","지하부질량(g)"]
                     if c in grow_mapped.columns]
        if grow_num:
            for col_name in grow_num:
                vals = pd.to_numeric(grow_mapped[col_name], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                st.markdown(f"### {col_name}")
                st.caption(STAT_HINTS.get("평균 (mean)", "") +
                           "  ·  " + STAT_HINTS.get("표준편차 (std)", ""))
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ids = grow_mapped["개체번호"].astype(str).tolist()[:len(vals)]
                bars = ax.bar(ids, vals.values, color="#54A24B", alpha=0.8,
                              edgecolor="white", linewidth=0.5)
                ax.axhline(vals.mean(), color="red", linestyle="--",
                           label=f"평균 = {vals.mean():.2f}")
                ax.axhline(vals.median(), color="orange", linestyle=":",
                           label=f"중앙값 = {vals.median():.2f}")
                ax.set_xlabel("개체번호")
                ax.set_ylabel(col_name)
                ax.set_title(f"{col_name} — 개체별 분포  (n={len(vals)})")
                ax.legend(fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("📋 생육 결과 전체 데이터")
        st.dataframe(grow_mapped, use_container_width=True, height=300)

# ──────────────────────── Tab 3: 상관 분석 ────────────────────────
with tab3:
    st.header("🔗 상관 분석")
    st.markdown(
        "**[상관계수 (r)](https://ko.wikipedia.org/wiki/상관계수)** 는 "
        "두 변수가 함께 움직이는 정도를 −1에서 +1 사이의 값으로 나타냅니다. "
        "+1에 가까울수록 함께 증가, −1에 가까울수록 하나는 증가, 다른 하나는 감소합니다."
    )
    if grow_mapped is None:
        st.warning("생육 결과 데이터가 없습니다.")
    else:
        grow_num = [c for c in ["잎 수(장)","지상부 길이(mm)","지하부길이(mm)",
                                 "생중량(g)","지상부질량(g)","지하부질량(g)"]
                     if c in grow_mapped.columns]
        if len(grow_num) >= 2:
            corr = grow_mapped[grow_num].corr(numeric_only=True).round(3)

            # 히트맵
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(corr.index, fontsize=9)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    val = corr.iloc[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if abs(val) > 0.5 else "black", fontsize=9)
            fig.colorbar(im, ax=ax, label="상관계수 (r)")
            ax.set_title("생육 지표 간 상관계수 히트맵")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("📋 상관계수 표")
            st.dataframe(corr, use_container_width=True, height=220)
            st.caption("※ 대각선(자기 자신)은 1.000입니다. "
                       "예: 생중량(g) ↔ 지상부질량(g) r=0.98 → 매우 강한 양의 상관")

            # 환경-생육 간 상관 (환경이 있을 때만)
            if env_mapped is not None:
                env_num = [k for k in ["temperature","humidity","ec","ph","water_temp","co2"]
                           if any(a in env_mapped.columns for a in ENV_COL_MAP[k])]
                if env_num and grow_num:
                    st.subheader("🌡 환경 × 생육 상관")
                    st.markdown(
                        "환경 데이터의 각 시점 평균을 구한 뒤 생육 지표와 상관관계를 봅니다. "
                        "환경 측정이 생육 결과 수확 시점과 정확히 일치하지 않을 수 있어 "
                        "참고값으로 활용하세요."
                    )
                    env_vals = {}
                    for key in env_num:
                        real = [a for a in ENV_COL_MAP[key] if a in env_mapped.columns][0]
                        env_vals[key] = pd.to_numeric(env_mapped[real], errors="coerce")
                    env_means = pd.DataFrame(env_vals).mean()
                    env_df_corr = pd.DataFrame({"환경_평균": env_means})
                    # 환경 평균과 생육 평균 간 상관계수
                    combined = env_df_corr.copy()
                    for col_name in grow_num:
                        combined[col_name] = grow_mapped[col_name].mean()
                    if len(combined) > 1:
                        corr2 = combined.T.corr().round(2)
                        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                        im2 = ax2.imshow(corr2.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
                        ax2.set_xticks(range(len(corr2.columns)))
                        ax2.set_yticks(range(len(corr2.index)))
                        ax2.set_xticklabels(corr2.columns, fontsize=9)
                        ax2.set_yticklabels(corr2.index, fontsize=9)
                        for i in range(len(corr2.index)):
                            for j in range(len(corr2.columns)):
                                v = corr2.iloc[i, j]
                                ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                                         color="white" if abs(v) > 0.5 else "black", fontsize=8)
                        fig2.colorbar(im2, ax=ax2, label="r")
                        ax2.set_title("환경 평균 vs 생육 평균 (개체 평균 기준)")
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
        else:
            st.info("비교할 생육 지표가 2개 이상 필요합니다.")

# ──────────────────────── Tab 4: 시계열 보기 ────────────────────────
with tab4:
    st.header("📈 환경 시계열 보기")
    if env_mapped is None or env_time_col is None or env_time_col.isna().all():
        st.warning("시간 축이 있는 환경 데이터가 필요합니다.")
    else:
        env_plot_keys = [k for k in ["temperature","humidity","ec","ph","water_temp","co2"]
                         if any(a in env_mapped.columns for a in ENV_COL_MAP[k])]
        if not env_plot_keys:
            st.info("시계열로 볼 환경 수치형 열이 없습니다.")
        else:
            sel = st.multiselect("표시할 환경 변수 선택",
                                 env_plot_keys, default=env_plot_keys[:2])
            if sel:
                fig, ax = plt.subplots(figsize=(14, 4))
                colors = ["#E45756", "#54A24B", "#4C72B0", "#8C8C8C", "#CCB974", "#6699CC"]
                for i, key in enumerate(sel):
                    real = [a for a in ENV_COL_MAP[key] if a in env_mapped.columns][0]
                    vals = pd.to_numeric(env_mapped[real], errors="coerce")
                    mask = env_time_col.notna() & vals.notna()
                    ax.plot(env_time_col[mask], vals[mask], color=colors[i % len(colors)],
                            label=key, linewidth=1)
                ax.set_xlabel("시간")
                ax.set_ylabel("측정값")
                ax.set_title("환경 변수 시계열")
                ax.legend(fontsize=9, ncol=3)
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("※ 빨간 점선 구간은 센서가 측정하지 못한 빈 기간이 있을 수 있습니다.")

            # 결측 구간 확인
            st.subheader("⏳ 결측(측정 누락) 구간 확인")
            if "temperature" in env_mapped.columns:
                temp_real = [a for a in ENV_COL_MAP["temperature"] if a in env_mapped.columns][0]
                temp_vals = pd.to_numeric(env_mapped[temp_real], errors="coerce")
                gap = env_time_col.isna() | temp_vals.isna()
                gap_count = gap.sum()
                total = len(env_time_col)
                st.markdown(f"- **측정 있음**: {total - gap_count}회  ({round(100*(total-gap_count)/total,1)}%)")
                st.markdown(f"- **결측 / 미측정**: {gap_count}회  ({round(100*gap_count/total,1)}%)")

st.sidebar.markdown("---")
st.sidebar.caption("📌 팁")
st.sidebar.caption("1. csv 인코딩이 달라도 utf-8-sig / cp949 / utf-8 자동 감지")
st.sidebar.caption("2. 열 이름이 달라도 매핑 테이블로 자동 대응")
st.sidebar.caption("3. 한글 폰트 자동 탐색 (matplotlib + streamlit)")

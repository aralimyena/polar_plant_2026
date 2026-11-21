import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
import re

# ============================================================
# 한글 폰트 설정 (Streamlit Cloud에서도 깨지지 않도록)
# ============================================================
def setup_korean_font():
    """Noto Sans CJK KR 폰트를 찾아 matplotlib 기본 폰트로 설정"""
    font_paths = fm.findSystemFonts()
    target_font = None

    # Noto Sans CJK KR 우선, 없으면 다른 CJK 폰트
    for fp in font_paths:
        if "NotoSansCJK" in fp and "Regular" in fp:
            try:
                prop = fm.FontProperties(fname=fp)
                if "KR" in fp or "Korean" in fp or "CJK" in fp:
                    target_font = prop
                    break
            except:
                continue

    if target_font is None:
        # 두 번째 시도: 모든 CJK 폰트
        for fp in font_paths:
            if "CJK" in fp:
                try:
                    target_font = fm.FontProperties(fname=fp)
                    break
                except:
                    continue

    if target_font is not None:
        plt.rcParams["font.family"] = target_font.get_name()
        plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
        return True
    return False

font_setup_ok = setup_korean_font()

# ============================================================
# 열 이름 매핑 (학교마다 다른 열 이름 대응)
# ============================================================

# 환경 데이터 열 매핑: 다양한 형식 → 표준 이름
ENV_COLUMN_MAPPINGS = {
    "time": ["time", "categories", "날짜", "일시", "time stamp", "timestamp"],
    "temperature": ["temperature", "온도", "temp", "기온", "Temperature", "TEMP"],
    "humidity": ["humidity", "습도", "hum", "Humidity", "HUM"],
    "ec": ["ec", "EC", "전기전도도", "EC(ms/cm)", "ec_ms_cm"],
    "ph": ["ph", "PH", "수소이온농도", "pH"],
    "water_temp": ["water_temp", "수온", "water temperature", "WT", "water_temp_c"],
    "co2": ["co2", "CO2", "이산화탄소", "co2_ppm"],
    "비고": ["비고", "note", "memo", "remarks", "참고"],
}

# 생육 데이터 열 매핑
GROWTH_COLUMN_MAPPINGS = {
    "개체번호": ["개체번호", "개체 번호", "번호", "id", "plant_id", "개체 id", "ID"],
    "잎 수": ["잎 수(장)", "잎수", "잎 수", "leaf_count", "leaves", "엽수", "잎 개수"],
    "지상부 길이": ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이", "shoot_length", "shoot_len"],
    "지하부 길이": ["지하부 길이(mm)", "지하부길이(mm)", "지하부 길이", "지하부길이", "root_length", "root_len"],
    "생중량": ["생중량(g)", "생중량", "생 중량", "fresh_weight", "생체중량", "fw"],
    "지상부질량": ["지상부질량(g)", "지상부 질량(g)", "지상부질량", "shoot_mass", "shoot_weight", "지상부 무게"],
    "지하부질량": ["지하부질량(g)", "지하부 질량(g)", "지하부질량", "root_mass", "root_weight", "지하부 무게"],
}

def normalize_environmental_columns(df):
    """환경 데이터 열 이름을 표준화하여 반환"""
    renamed = {}
    found_columns = {}

    for standard_name, aliases in ENV_COLUMN_MAPPINGS.items():
        actual_col = None
        for alias in aliases:
            for col in df.columns:
                if col.lower().strip() == alias.lower().strip():
                    actual_col = col
                    break
            if actual_col:
                break
        if actual_col and actual_col not in renamed:
            renamed[actual_col] = standard_name
            found_columns[standard_name] = actual_col

    df_renamed = df.rename(columns=renamed)
    return df_renamed, found_columns

def normalize_growth_columns(df):
    """생육 데이터 열 이름을 표준화하여 반환"""
    renamed = {}

    for standard_name, aliases in GROWTH_COLUMN_MAPPINGS.items():
        actual_col = None
        for alias in aliases:
            for col in df.columns:
                if col.lower().strip() == alias.lower().strip():
                    actual_col = col
                    break
            if actual_col:
                break
        if actual_col and actual_col not in renamed:
            renamed[actual_col] = standard_name

    df_renamed = df.rename(columns=renamed)
    return df_renamed, list(renamed.keys())

# ============================================================
# 인코딩 자동 감지
# ============================================================
def detect_encoding(file_bytes):
    """파일 인코딩을 감지하여 적절한 디코딩 시도"""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "iso-8859-1"]

    for enc in encodings:
        try:
            decoded = file_bytes.decode(enc)
            # 한글 문자가 정상적으로 디코딩되었는지 확인
            if any(0xac00 <= ord(c) <= 0xd7af for c in decoded[:1000]):
                return enc
            # UTF-8 BOM 확인
            if decoded.startswith("\ufeff"):
                return "utf-8-sig"
        except (UnicodeDecodeError, UnicodeError):
            continue

    return "utf-8-sig"

def load_csv_safe(file):
    """CSV 파일을 안전하게 로드 (인코딩 자동 감지 + 다양한 열 이름 처리)"""
    file_bytes = file.read()
    file.seek(0)

    encoding = detect_encoding(file_bytes)

    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
        except:
            df = pd.read_csv(BytesIO(file_bytes), encoding="cp949")

    df = df.replace("", np.nan)
    return df, encoding

# ============================================================
# 시간 데이터 파싱
# ============================================================
def parse_time_column(df, time_col_name):
    """시간 열 값을 datetime으로 변환 (다양한 형식 지원)"""
    if time_col_name is None or time_col_name not in df.columns:
        return df

    series = df[time_col_name].astype(str)

    # 형식 1: YYYY-MM-DD HH:MM:SS
    try:
        df["datetime"] = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S")
        return df
    except:
        pass

    # 형식 2: YYYYMMDD_N (날짜_시각 인덱스)
    pattern = r"^(\d{8})_(\d+)$"
    matched = series.str.match(pattern)
    if matched.any():
        def parse_date_time(val):
            match = re.match(pattern, str(val))
            if match:
                date_str, time_idx = match.groups()
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                idx = int(time_idx)
                hour = idx // 60
                minute = idx % 60
                return pd.Timestamp(year=year, month=month, day=day, 
                                   hour=hour, minute=minute)
            return pd.NaT

        df["datetime"] = series.apply(parse_date_time)
        return df

    # 형식 3: 기타 날짜 형식
    try:
        df["datetime"] = pd.to_datetime(series, errors="coerce")
    except:
        df["datetime"] = pd.NaT

    return df

# ============================================================
# 통계 설명 (고등학생용)
# ============================================================
STAT_EXPLANATIONS = {
    "평균": "평균(mean): 모든 값을 더해 개수로 나눈 값. 데이터의 대표값을 나타냅니다.",
    "표준편차": "표준편차(std): 데이터가 평균에서 얼마나 흩어져 있는지 나타내는 값. 클수록 데이터가 넓게 퍼져 있습니다.",
    "분산": "분산(variance): 표준편차를 제곱한 값. 데이터가 평균에서 떨어진 거리의 제곱 평균.",
    "상관관계": "상관관계(correlation): 두 변수가 함께 움직이는 정도. +1에 가까우면 같이 증가, -1에 가까우면 반대 방향, 0에 가까우면 관계 없음.",
    "중앙값": "중앙값(median): 데이터를 크기순으로 정렬했을 때 정가운데 있는 값. 극단값의 영향을 덜 받습니다.",
    "최솟값": "최솟값(min): 데이터 중 가장 작은 값.",
    "최댓값": "최댓값(max): 데이터 중 가장 큰 값.",
}

# ============================================================
# Streamlit 앱
# ============================================================
st.set_page_config(
    page_title="극지식물 생육 분석 대시보드",
    page_icon="\U0001f331",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("\U0001f331 극지식물 생육 차이 분석 대시보드")
st.caption("인천아라고등학교 극지식물 프로젝트 - 환경 데이터와 개체별 생육 결과 통합 분석")

# 사이드바 - 데이터 업로드
st.sidebar.header("\U0001f4c1 데이터 업로드")
st.sidebar.info("""
**사용 방법**
1. 환경 데이터 CSV와 생육 결과 데이터 CSV를 각각 업로드
2. 인코딩(utf-8-sig/cp949)과 열 이름이 달라도 자동 처리
3. 한글 폰트 깨짐 없이 Streamlit Cloud에서도 실행
""")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type=["csv"])
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type=["csv"])

# 예시 데이터 안내
st.sidebar.divider()
st.sidebar.subheader("\U0001f4cb 열 이름 예시")
st.sidebar.code("""
# 환경 데이터 (형식 A - 우리 학교)
time, temperature, humidity, ec, ph, water_temp, co2
2026-07-13 17:00:00, 21.9, 53.7, 0.84, 7.05, 23.3, 1186

# 환경 데이터 (형식 B - 다른 학교 가능)
categories, 온도, 습도, EC, PH

# 생육 데이터
개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g), 지상부질량(g), 지하부질량(g)
1, 61, 48, 170, 11.31, 7.31, 4.00
""", language="python")

if env_file is None and growth_file is None:
    st.warning("\U0001f4d0 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")
    st.stop()

# ============================================================
# 데이터 로드 및 처리
# ============================================================
env_df = None
growth_df = None
env_encoding = None
growth_encoding = None

if env_file is not None:
    with st.spinner("환경 데이터 로딩 중..."):
        env_df, env_encoding = load_csv_safe(env_file)
        env_df, env_found = normalize_environmental_columns(env_df)

        time_col = env_found.get("time")
        if time_col:
            env_df = parse_time_column(env_df, time_col)

        st.success(f"\u2705 환경 데이터 로드 완료 (인코딩: {env_encoding})")
        st.info(f"감지된 환경 열: {list(env_found.keys())}")

if growth_file is not None:
    with st.spinner("생육 결과 데이터 로딩 중..."):
        growth_df, growth_encoding = load_csv_safe(growth_file)
        growth_df, growth_original = normalize_growth_columns(growth_df)

        st.success(f"\u2705 생육 데이터 로드 완료 (인코딩: {growth_encoding})")
        st.info(f"감지된 생육 열: {list(growth_df.columns)}")

# ============================================================
# 데이터 미리보기
# ============================================================
if env_df is not None:
    st.subheader("\U0001f321\u200d\u25b3 환경 데이터")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 측정 횟수", f"{len(env_df):,}회")
    with col2:
        if "datetime" in env_df.columns:
            time_range = f"{env_df['datetime'].min()} ~ {env_df['datetime'].max()}"
            st.metric("측정 기간", time_range)

    st.caption("* 학교마다 열 이름이 달라도 자동으로 표준화하여 처리합니다.")
    st.dataframe(env_df, use_container_width=True, height=300)

    missing = env_df.isnull().sum()
    if missing.any():
        st.warning(f"* 일부 환경 측정값에 결측이 있습니다: {dict(missing[missing > 0])}")

if growth_df is not None:
    st.subheader("\U0001f33f 개체별 생육 결과 데이터")
    st.metric("총 개체 수", f"{len(growth_df):,}개체")
    st.dataframe(growth_df, use_container_width=True, height=300)

# ============================================================
# 분석 탭
# ============================================================
if env_df is not None or growth_df is not None:
    st.divider()
    st.subheader("\U0001f4ca 분석 결과")

    tab1, tab2, tab3 = st.tabs(["\U0001f321\u200d\u25b3 환경 데이터 분석", "\U0001f33f 생육 데이터 분석", "\U0001f517 환경-생육 상관관계"])

    with tab1:
        if env_df is None:
            st.warning("환경 데이터를 업로드해주세요.")
        else:
            env_vars = [col for col in ["temperature", "humidity", "ec", "ph", "water_temp", "co2"] 
                       if col in env_df.columns]

            if env_vars:
                st.subheader("\U0001f4c8 환경 변수별 기초 통계")
                st.caption("각 통계값의 의미: 아래 버튼을 눌러 확인하세요.")

                stats_rows = []
                for var in env_vars:
                    data = env_df[var].dropna()
                    stats_rows.append({
                        "변수": var,
                        "평균": round(data.mean(), 2),
                        "표준편차": round(data.std(), 2),
                        "중앙값": round(data.median(), 2),
                        "최솟값": round(data.min(), 2),
                        "최댓값": round(data.max(), 2),
                        "측정 개수": len(data)
                    })
                stats_df = pd.DataFrame(stats_rows)
                st.dataframe(stats_df, use_container_width=True)

                with st.expander("\U0001f4d6 통계 용어 설명 (고등학생용)"):
                    for term, explanation in STAT_EXPLANATIONS.items():
                        st.write(f"**{term}**: {explanation}")

                st.subheader("\U0001f4c9 시간에 따른 환경 변화")

                chart_cols = st.multiselect(
                    "차트에 표시할 환경 변수 선택",
                    env_vars,
                    default=env_vars[:2] if len(env_vars) >= 2 else env_vars
                )

                if chart_cols and "datetime" in env_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for col in chart_cols:
                        ax.plot(env_df["datetime"], env_df[col], label=col, alpha=0.8, linewidth=1)
                    ax.set_xlabel("시간")
                    ax.set_ylabel("값")
                    ax.set_title("시간에 따른 환경 변수 변화")
                    ax.legend(loc="best")
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                elif chart_cols:
                    st.warning("시간 정보가 없어 시계열 차트를 그릴 수 없습니다.")

                st.subheader("\U0001f4ca 환경 변수 분포")
                dist_var = st.selectbox("분포를 볼 변수 선택", env_vars)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                data = env_df[dist_var].dropna()
                axes[0].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
                axes[0].axvline(data.mean(), color="red", linestyle="--", label=f"평균: {data.mean():.2f}")
                axes[0].axvline(data.median(), color="green", linestyle="--", label=f"중앙값: {data.median():.2f}")
                axes[0].set_xlabel(dist_var)
                axes[0].set_ylabel("빈도")
                axes[0].set_title(f"{dist_var} 분포")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                bp = axes[1].boxplot([data], labels=[dist_var], patch_artist=True)
                bp["boxes"][0].set_facecolor("lightblue")
                axes[1].set_ylabel(dist_var)
                axes[1].set_title(f"{dist_var} 박스 플롯")
                axes[1].grid(True, alpha=0.3, axis="y")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("환경 데이터에 분석할 변수가 없습니다.")

    with tab2:
        if growth_df is None:
            st.warning("생육 데이터를 업로드해주세요.")
        else:
            growth_vars = [col for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부질량", "지하부질량"] 
                          if col in growth_df.columns]

            if growth_vars:
                st.subheader("\U0001f4c8 개체별 생육 통계")

                g_stats_rows = []
                for var in growth_vars:
                    data = growth_df[var].dropna()
                    g_stats_rows.append({
                        "변수": var,
                        "평균": round(data.mean(), 2),
                        "표준편차": round(data.std(), 2),
                        "중앙값": round(data.median(), 2),
                        "최솟값": round(data.min(), 2),
                        "최댓값": round(data.max(), 2),
                    })
                g_stats_df = pd.DataFrame(g_stats_rows)
                st.dataframe(g_stats_df, use_container_width=True)

                with st.expander("\U0001f4d6 통계 용어 설명 (고등학생용)"):
                    for term, explanation in STAT_EXPLANATIONS.items():
                        st.write(f"**{term}**: {explanation}")

                st.subheader("\U0001f4ca 개체별 생육 비교")

                x_var = st.selectbox("X축: 비교 기준", growth_vars, index=0)
                y_var = st.selectbox("Y축: 비교 대상", growth_vars, index=1 if len(growth_vars) > 1 else 0)

                fig, ax = plt.subplots(figsize=(10, 5))
                for var in growth_vars:
                    if var == x_var:
                        continue
                    ax.plot(growth_df["개체번호"], growth_df[var], marker="o", label=var, alpha=0.8)
                ax.set_xlabel("개체 번호")
                ax.set_ylabel(y_var)
                ax.set_title("개체별 생육 특성 비교")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

                st.subheader("\U0001f4ca 개체별 생육 성분 막대 그래프")
                bar_vars = [v for v in growth_vars if v in ["지상부질량", "지하부질량", "생중량"]]

                if bar_vars:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    x = np.arange(len(growth_df))
                    width = 0.25
                    colors = {"지상부질량": "#2ecc71", "지하부질량": "#3498db", "생중량": "#e74c3c"}

                    for i, var in enumerate(bar_vars):
                        if var in growth_df.columns:
                            ax.bar(x + i * width, growth_df[var], width, 
                                   label=var, color=colors.get(var, "gray"), alpha=0.8)

                    ax.set_xlabel("개체 번호")
                    ax.set_ylabel("질량 (g)")
                    ax.set_title("개체별 지상부/지하부/전체 질량 비교")
                    ax.set_xticks(x + width)
                    ax.set_xticklabels(growth_df["개체번호"])
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis="y")
                    st.pyplot(fig)
                    plt.close()

                st.subheader("\U0001f4c9 생육 변수 분포")
                dist_var = st.selectbox("분포를 볼 변수", growth_vars)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                data = growth_df[dist_var].dropna()
                axes[0].hist(data, bins=max(10, len(data)//3), color="coral", edgecolor="white", alpha=0.8)
                axes[0].axvline(data.mean(), color="red", linestyle="--", label=f"평균: {data.mean():.2f}")
                axes[0].set_xlabel(dist_var)
                axes[0].set_ylabel("개체 수")
                axes[0].set_title(f"{dist_var} 분포")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                bp = axes[1].boxplot([data], labels=[dist_var], patch_artist=True)
                bp["boxes"][0].set_facecolor("lightcoral")
                axes[1].set_ylabel(dist_var)
                axes[1].set_title(f"{dist_var} 박스 플롯")
                axes[1].grid(True, alpha=0.3, axis="y")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("생육 데이터에 분석할 변수가 없습니다.")

    with tab3:
        st.subheader("\U0001f517 환경 조건과 생육의 관계 분석")
        st.caption("""
**상관관계란?** 두 값이 함께 움직이는 정도를 나타냅니다.
- **양수(+) 상관**: 한 값이 증가할 때 다른 값도 증가 (예: 온도↑ → 생장↑)
- **음수(-) 상관**: 한 값이 증가할 때 다른 값은 감소
- **0에 가까움**: 두 값 사이에 뚜렷한 관계가 없음
""")

        if env_df is not None and growth_df is not None:
            env_vars = [col for col in ["temperature", "humidity", "ec", "ph", "water_temp", "co2"] 
                       if col in env_df.columns]
            growth_vars = [col for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부질량", "지하부질량"] 
                          if col in growth_df.columns]

            if env_vars and growth_vars:
                if "datetime" in env_df.columns:
                    env_time_range = (env_df["datetime"].min(), env_df["datetime"].max())
                    st.info(f"\U0001f535 환경 데이터 기간: {env_time_range[0]} ~ {env_time_range[1]}")
                else:
                    st.warning("\u26a0\ufe0f 환경 데이터에 시간 정보가 없어 생육 데이터와 직접 매칭이 어렵습니다.")

                st.subheader("\U0001f4ca 환경-생육 상관계수 행렬")
                st.warning("""
**참고**: 이 분석은 환경 데이터 전체 평균과 각 개체의 생육 결과를 단순 비교한 것입니다.
정확한 분석을 위해서는 각 개체가 자란 기간의 환경 데이터를 매칭해야 합니다.
""")

                corr_data = {}
                for g_var in growth_vars:
                    if g_var not in growth_df.columns:
                        continue
                    g_values = growth_df[g_var].values
                    corr_data[g_var] = {}
                    for e_var in env_vars:
                        if len(env_df) >= len(growth_df):
                            step = len(env_df) // len(growth_df)
                            if step > 0:
                                env_avg = [env_df[e_var].iloc[i*step:(i+1)*step].mean() 
                                         for i in range(len(growth_df))]
                                env_avg = env_avg[:len(growth_df)]
                            else:
                                env_avg = [env_df[e_var].mean()] * len(growth_df)
                        else:
                            env_avg = [env_df[e_var].mean()] * len(growth_df)

                        if len(env_avg) == len(g_values) and np.std(env_avg) > 0:
                            corr = np.corrcoef(env_avg, g_values)[0, 1]
                            corr_data[g_var][e_var] = corr
                        else:
                            corr_data[g_var][e_var] = np.nan

                corr_df = pd.DataFrame(corr_data).T.round(3)
                corr_df = corr_df[env_vars]

                if not corr_df.empty:
                    st.dataframe(corr_df, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                    ax.set_xticks(range(len(corr_df.columns)))
                    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(corr_df.index)))
                    ax.set_yticklabels(corr_df.index)
                    ax.set_title("환경-생육 상관관계 히트맵")

                    for i in range(len(corr_df.index)):
                        for j in range(len(corr_df.columns)):
                            val = corr_df.values[i, j]
                            if not np.isnan(val):
                                color = "white" if abs(val) > 0.5 else "black"
                                ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                                       color=color, fontsize=10)

                    plt.colorbar(im, ax=ax, label="상관계수")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.subheader("\U0001f4a1 주요 상관관계 해석")
                    strong_corr = []
                    for g_var in corr_df.index:
                        for e_var in corr_df.columns:
                            val = corr_df.loc[g_var, e_var]
                            if not np.isnan(val) and abs(val) > 0.3:
                                direction = "함께 증가" if val > 0 else "반대 방향"
                                strong_corr.append(f"**{g_var}** <-> **{e_var}**: r = {val:.2f} ({direction})")

                    if strong_corr:
                        for item in strong_corr:
                            st.write(f"- {item}")
                    else:
                        st.write("상관계수가 0.3을 넘는 관계가 발견되지 않았습니다.")
                else:
                    st.warning("상관계수를 계산할 수 없습니다.")
            else:
                if not env_vars:
                    st.warning("환경 데이터가 없어 상관관계를 분석할 수 없습니다.")
                if not growth_vars:
                    st.warning("생육 데이터가 없어 상관관계를 분석할 수 없습니다.")
        else:
            st.warning("환경 데이터와 생육 데이터를 모두 업로드해야 상관관계를 분석할 수 있습니다.")

# ============================================================
# 데이터 내보내기
# ============================================================
if env_df is not None or growth_df is not None:
    st.divider()
    st.subheader("\U0001f4be 데이터 내보내기")

    col1, col2 = st.columns(2)

    with col1:
        if env_df is not None:
            csv = env_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="\U0001f4e1 환경 데이터 다운로드 (utf-8-sig)",
                data=csv,
                file_name="processed_environment.csv",
                mime="text/csv"
            )

    with col2:
        if growth_df is not None:
            csv = growth_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="\U0001f4e1 생육 데이터 다운로드 (utf-8-sig)",
                data=csv,
                file_name="processed_growth.csv",
                mime="text/csv"
            )

# ============================================================
# 푸터
# ============================================================
st.divider()
st.caption("""
**참고 사항**
- 이 대시보드는 utf-8-sig/cp949 인코딩과 다양한 열 이름 형식을 자동 처리합니다
- 한글 폰트는 Noto Sans CJK KR을 사용하여 Streamlit Cloud에서도 깨지지 않습니다
- 통계 용어 설명은 고등학생이 이해하기 쉽도록 작성되었습니다
- 환경 데이터와 생육 데이터의 시간 정보가 다를 경우 상관관계 분석에 제한이 있을 수 있습니다
""")

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

# ───────────────── 기본 설정 ─────────────────
st.set_page_config(page_title="그린란드고추냉이 생육 분석", page_icon="🌱", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
HI_C, LO_C, MID_C = "#2E8B57", "#E07A5F", "#B0B7C3"
RATIO = "뿌리 질량 대비 길이"
HI, LO, MID = "상위 25%", "하위 25%", "중간"
UNITS = {"잎 수": "장", "지상부 길이": "mm", "지하부 길이": "mm", "생중량": "g",
         "지상부 질량": "g", "지하부 질량": "g", RATIO: "mm/g"}
NICK = {"지하부 길이": "뿌리 길이", "지하부 질량": "뿌리 질량"}
ENV_NAMES = {"temperature": "기온", "humidity": "습도", "ph": "pH", "ec": "EC",
             "water_temp": "수온", "co2": "CO₂"}


# ───────────────── 열 이름 표준화 ─────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


# ───────────────── 데이터 읽기 (한 번만) ─────────────────
@st.cache_data(show_spinner=False)
def load_csv(data):
    raw = pd.DataFrame()
    for enc in ("utf-8-sig", "cp949"):
        try:
            raw = pd.read_csv(io.BytesIO(data), encoding=enc)
            break
        except Exception:
            pass
    try:
        return std_cols(raw)
    except Exception:
        return raw


def school_name(file):
    stem = file.name.rsplit(".", 1)[0]
    return stem.split("_")[0].strip() or stem


def id_str(v):
    # 개체번호를 56.0이 아닌 56, A1처럼 원래 모양으로 표시
    if pd.isna(v):
        return "-"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def is_num(df, c):
    return c in df.columns and pd.api.types.is_numeric_dtype(df[c])


def num_cols(df):
    return [c for c in df.columns if c not in ("time", "개체번호") and is_num(df, c)]


def label(c):
    u = UNITS.get(c, "")
    return f"{NICK.get(c, c)}({u})" if u else NICK.get(c, c)


def clean(s):
    return pd.to_numeric(s, errors="coerce").dropna()


def prepare_growth(df):
    g = df.copy()
    if "개체번호" in g.columns:
        g["개체"] = g["개체번호"].map(id_str)
    else:
        g["개체"] = [str(i + 1) for i in range(len(g))]
    if is_num(g, "지하부 길이") and is_num(g, "지하부 질량"):
        mass = g["지하부 질량"].where(g["지하부 질량"] > 0)
        g[RATIO] = g["지하부 길이"] / mass
    g["생육 그룹"] = MID
    if is_num(g, "생중량"):
        fw = g["생중량"].dropna()
        if len(fw) >= 4:
            q1, q3 = fw.quantile(0.25), fw.quantile(0.75)
            g.loc[g["생중량"] >= q3, "생육 그룹"] = HI
            g.loc[g["생중량"] <= q1, "생육 그룹"] = LO
    return g


def welch(a, b):
    a, b = clean(a), clean(b)
    if len(a) < 2 or len(b) < 2:
        return None
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return None if pd.isna(p) else float(p)


def style(fig, title, height=420):
    fig.update_layout(title=title, height=height, font=dict(family=FONT),
                      margin=dict(l=40, r=20, t=70, b=40),
                      legend=dict(orientation="h", y=-0.15))
    return fig


# ───────────────── 그래프 함수 (재사용) ─────────────────
def individual_bar(g, col):
    d = g[["개체", col, "생육 그룹"]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col]).sort_values(col, ascending=False)
    if len(d) < 2:
        return None
    colors = [HI_C if x == HI else LO_C if x == LO else MID_C for x in d["생육 그룹"]]
    fig = go.Figure(go.Bar(x=d["개체"], y=d[col], marker_color=colors,
                           hovertemplate="개체 %{x}<br>%{y:.2f}<extra></extra>"))
    fig.add_hline(y=d[col].mean(), line_dash="dot", line_color="#555",
                  annotation_text=f"평균 {d[col].mean():.2f}")
    fig.update_xaxes(type="category", title="개체번호")
    fig.update_yaxes(title=label(col))
    return style(fig, f"개체별 {label(col)} — 초록: 생중량 상위 25% · 주황: 생중량 하위 25%")


def group_bar(hi, lo, cols, title):
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=[label(c) for c in cols])
    for i, c in enumerate(cols, start=1):
        for name, part, color in ((HI, hi, HI_C), (LO, lo, LO_C)):
            v = clean(part[c])
            if v.empty:
                continue
            sd = float(v.std()) if len(v) > 1 else 0.0
            fig.add_trace(go.Bar(x=[name], y=[v.mean()], name=name, marker_color=color,
                                 error_y=dict(type="data", array=[sd]),
                                 text=[f"{v.mean():.2f}"], textposition="inside",
                                 showlegend=(i == 1)), row=1, col=i)
    return style(fig, title)


def scatter_trend(g, x, y, title):
    d = g[[x, y, "개체", "생육 그룹"]].copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=[x, y])
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None, None, None
    fig = go.Figure()
    for grp, color in ((HI, HI_C), (MID, MID_C), (LO, LO_C)):
        part = d[d["생육 그룹"] == grp]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(x=part[x], y=part[y], mode="markers", name=grp, text=part["개체"],
                                 marker=dict(color=color, size=11, line=dict(width=1, color="white")),
                                 hovertemplate="개체 %{text}<br>x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>"))
    r, p = stats.pearsonr(d[x], d[y])
    slope, inter = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + inter, mode="lines", name="추세선",
                             line=dict(color="#333333", dash="dash")))
    fig.update_xaxes(title=label(x))
    fig.update_yaxes(title=label(y))
    return style(fig, title), float(r), float(p)


def compare_text(c, hi, lo):
    a, b = clean(hi[c]), clean(lo[c])
    if a.empty or b.empty:
        return None
    p = welch(hi[c], lo[c])
    u = UNITS.get(c, "")
    if p is None:
        sig = "검정 불가(개체 수 부족)"
    elif p < 0.05:
        sig = f"뚜렷한 차이 (p = {p:.4f})"
    else:
        sig = f"통계적으로 뚜렷한 차이 없음 (p = {p:.3f})"
    way = "더 작음" if b.mean() < a.mean() else "더 큼"
    return f"- **{NICK.get(c, c)}**: 하위 그룹 {b.mean():.2f}{u} · 상위 그룹 {a.mean():.2f}{u} → 하위 그룹이 {way}, {sig}"


def env_section(env):
    cols = [c for c in ENV_NAMES if is_num(env, c)]
    if not cols:
        st.info("환경 CSV에서 온도·습도·pH·EC 같은 숫자 열을 찾지 못했어요.")
        return
    rows = []
    for c in cols:
        v = env[c].dropna()
        if len(v) == 0:
            continue
        rows.append({"항목": ENV_NAMES[c], "평균": round(v.mean(), 2),
                     "표준편차": round(v.std(), 2) if len(v) > 1 else None,
                     "최소": round(v.min(), 2), "최대": round(v.max(), 2), "측정 수": len(v)})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("📘 평균: 값을 모두 더해 개수로 나눈 값 · 표준편차: 값들이 평균에서 얼마나 퍼져 있는지(클수록 환경이 많이 흔들림)")
    if "time" in env.columns and env["time"].notna().sum() >= 2:
        d = env.dropna(subset=["time"]).sort_values("time").set_index("time")[cols]
        d = d[~d.index.duplicated()]
        diffs = d.index.to_series().diff()
        step = diffs.median()
        gaps = 0
        if pd.notna(step) and step > pd.Timedelta(0) and (d.index[-1] - d.index[0]) / step < 200000:
            gaps = int((diffs > step * 2).sum())
            d = d.resample(step).mean()
        fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=[ENV_NAMES[c] for c in cols])
        for i, c in enumerate(cols, start=1):
            fig.add_trace(go.Scatter(x=d.index, y=d[c], mode="lines", name=ENV_NAMES[c],
                                     showlegend=False), row=i, col=1)
        st.plotly_chart(style(fig, "시간에 따른 재배 환경 변화", height=max(320, 170 * len(cols))), width="stretch")
        st.caption(f"측정 기간: {d.index.min():%Y-%m-%d %H:%M} ~ {d.index.max():%Y-%m-%d %H:%M}"
                   + (f" · 센서가 측정하지 못한 빈 구간 {gaps}곳(그래프가 끊긴 부분)" if gaps else ""))
    else:
        st.info("시간 열을 읽지 못해 환경 변화 그래프는 건너뛰었어요.")


# ───────────────── 사이드바 & 데이터 로딩 ─────────────────
st.sidebar.header("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df, growth, school = pd.DataFrame(), pd.DataFrame(), ""
with st.spinner("데이터를 불러오는 중이에요..."):
    if env_file is not None:
        env_df = load_csv(env_file.getvalue())
        school = school_name(env_file)
    if growth_file is not None:
        growth = load_csv(growth_file.getvalue())
        school = school_name(growth_file)
        try:
            growth = prepare_growth(growth)
        except Exception:
            pass
has_growth = (not growth.empty) and "생육 그룹" in growth.columns and len(num_cols(growth)) > 0

st.sidebar.header("📏 생육 측정 항목 선택")
items = num_cols(growth) if has_growth else []
if items:
    bar_item = st.sidebar.selectbox("개체별 막대그래프 항목", items,
                                    index=items.index("생중량") if "생중량" in items else 0, format_func=label)
    cmp_items = st.sidebar.multiselect("그룹 비교 표 항목", items, format_func=label,
                                       default=[c for c in ["잎 수", "지하부 길이", "지하부 질량", RATIO] if c in items])
else:
    bar_item, cmp_items = None, []
    st.sidebar.caption("생육 CSV를 올리면 항목을 고를 수 있어요.")

st.title("그린란드고추냉이의 생육 수준에 따른 뿌리 특성 비교")
if school:
    st.caption(f"🏫 {school} 스마트팜 데이터")
if env_file is None and growth_file is None:
    st.info("👈 왼쪽 사이드바에서 환경 데이터 CSV와 생육 결과 데이터 CSV를 올려 주세요. 한 파일만 올려도 해당 부분은 동작해요.")

tab1, tab2, tab3 = st.tabs(["① 탐구의 배경", "② 생육 그룹별 특징 비교", "③ 가설 검증과 최종 결론"])

# ───────────────── 탭 1 ─────────────────
with tab1:
    try:
        st.header("탐구의 배경 — 양분과 그린란드고추냉이의 생육")
        st.subheader("① 그린란드고추냉이와 양분의 관계")
        st.markdown("- 그린란드고추냉이는 북극해 주변 해안에 서식하는 식물이다.\n"
                    "- 기존 스발바르 야외 연구에서는 토양 양분이 풍부한 바닷새 번식지 주변의 개체가 더 크게 자라는 경향이 확인되었다.\n"
                    "- 이를 통해 양분 조건이 그린란드고추냉이의 생육과 관련될 수 있음을 알 수 있다.")

        st.subheader("② 우리 학교 스마트팜에서도 생육 차이가 나타났다")
        if not has_growth:
            st.info("생육 결과 CSV를 올리면 개체별 생육 차이를 볼 수 있어요.")
        else:
            if is_num(growth, "생중량") and growth["생중량"].dropna().shape[0] >= 2:
                fw = growth["생중량"].dropna()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("탐구 개체 수", f"{len(fw)}개체")
                c2.metric("가장 무거운 개체", f"{fw.max():.2f} g", f"개체 {growth.loc[fw.idxmax(), '개체']}", delta_color="off")
                c3.metric("가장 가벼운 개체", f"{fw.min():.2f} g", f"개체 {growth.loc[fw.idxmin(), '개체']}", delta_color="off")
                c4.metric("최대 ÷ 최소", f"약 {fw.max() / fw.min():.1f}배" if fw.min() > 0 else "-")
                st.markdown(f"같은 스마트팜에서 같은 기간 재배했지만 개체별 생중량은 **{fw.min():.2f} ~ {fw.max():.2f} g**으로 차이가 났다.")
            else:
                st.info("생중량(또는 지상부·지하부 질량) 열이 없어 생중량 요약은 건너뛰었어요.")
            fig = individual_bar(growth, bar_item) if bar_item else None
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
                st.caption("색은 항상 생중량 기준 그룹이에요. 사이드바에서 다른 항목을 고르면 같은 개체의 다른 측정값을 비교할 수 있어요.")
            else:
                st.info("선택한 항목에 숫자 값이 2개 이상 없어 그래프를 건너뛰었어요.")

        st.subheader("③ 뿌리에 주목해 가설을 세웠다")
        st.markdown("- 뿌리는 물과 무기 양분을 흡수하는 기관이다.\n"
                    "- 식물은 광합성으로 만든 유기물을 잎·줄기·뿌리에 나누어 사용한다.\n"
                    "- 덜 자란 개체는 뿌리에 축적할 수 있는 유기물이 적어 뿌리가 가늘고 길게 자랐을 것이라고 예상했다.\n\n"
                    "> **가설: 덜 자란 개체일수록 뿌리 질량 대비 길이(뿌리 길이 ÷ 뿌리 질량, mm/g)가 클 것이다.**\n\n"
                    "**탐구 목적**: 같은 환경에서 자란 개체들의 생육 차이가 뿌리의 어떤 특성과 함께 나타나는지 확인한다.")

        st.subheader("🌡️ 재배 환경 요약 (같은 스마트팜 환경)")
        if env_df.empty:
            st.info("환경 데이터 CSV를 올리면 재배 기간의 환경 변화를 볼 수 있어요.")
        else:
            env_section(env_df)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ───────────────── 탭 2 ─────────────────
with tab2:
    try:
        st.header("생육 그룹별 특징 비교 — 덜 자란 개체는 무엇이 달랐을까?")
        if not has_growth or not is_num(growth, "생중량"):
            st.info("생육 결과 CSV(생중량 또는 지상부·지하부 질량 포함)를 올려 주세요.")
        else:
            hi, lo = growth[growth["생육 그룹"] == HI], growth[growth["생육 그룹"] == LO]
            if len(hi) < 2 or len(lo) < 2:
                st.info("그룹을 나눌 개체 수가 부족해요(각 그룹 2개체 이상 필요).")
            else:
                st.caption("📘 상위 25%: 생중량이 큰 순서로 위쪽 4분의 1(제3사분위수 이상) · 하위 25%: 아래쪽 4분의 1(제1사분위수 이하)")
                c1, c2 = st.columns(2)
                c1.metric("상위 그룹 (잘 자란 개체)", f"{len(hi)}개체", f"생중량 ≥ {hi['생중량'].min():.2f} g", delta_color="off")
                c2.metric("하위 그룹 (덜 자란 개체)", f"{len(lo)}개체", f"생중량 ≤ {lo['생중량'].max():.2f} g", delta_color="off")

                st.subheader("① 생중량 상위 25%와 하위 25% 비교")
                rows = []
                for c in cmp_items:
                    a, b = clean(hi[c]), clean(lo[c])
                    if a.empty or b.empty:
                        continue
                    p = welch(hi[c], lo[c])
                    rows.append({"생육 항목": label(c), "상위 그룹 평균": round(a.mean(), 2),
                                 "하위 그룹 평균": round(b.mean(), 2), "차이(상위-하위)": round(a.mean() - b.mean(), 2),
                                 "p값": None if p is None else round(p, 4),
                                 "판단": "-" if p is None else ("뚜렷한 차이" if p < 0.05 else "뚜렷한 차이 없음")})
                if rows:
                    table = pd.DataFrame(rows)
                    st.dataframe(table, width="stretch", hide_index=True)
                    st.download_button("📥 그룹 비교 표 내려받기 (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                                       file_name=f"{school}_그룹비교.csv", mime="text/csv")
                    st.caption("📘 p값(웰치 t검정): 두 그룹의 평균 차이가 우연히 생겼을 가능성. 보통 0.05보다 작으면 '뚜렷한 차이'로 본다. "
                               "그룹당 개체 수가 적으면 결과가 흔들릴 수 있다.")
                else:
                    st.info("사이드바에서 비교할 항목을 골라 주세요.")

                st.subheader("② 주요 분석 결과")
                lines = [compare_text(c, hi, lo) for c in ["잎 수", "지하부 길이", "지하부 질량"] if is_num(growth, c)]
                lines = [x for x in lines if x]
                if lines:
                    st.markdown("\n".join(lines))
                    p_leaf, p_len, p_mass = (welch(hi[c], lo[c]) if is_num(growth, c) else None
                                             for c in ["잎 수", "지하부 길이", "지하부 질량"])
                    if p_len is not None and p_len >= 0.05 and ((p_leaf is not None and p_leaf < 0.05) or (p_mass is not None and p_mass < 0.05)):
                        st.success("따라서 생육 차이는 뿌리 길이보다 잎 수와 뿌리 질량에서 뚜렷하게 나타났다.")
                else:
                    st.info("잎 수·뿌리 길이·뿌리 질량 열을 찾지 못했어요.")

                if is_num(growth, "잎 수"):
                    st.plotly_chart(group_bar(hi, lo, ["잎 수"], "상위·하위 그룹의 잎 수 비교 (막대 위 선: 표준편차)"), width="stretch")
                else:
                    st.info("잎 수 열이 없어 잎 수 그래프는 건너뛰었어요.")
                root_cols = [c for c in ["지하부 길이", "지하부 질량"] if is_num(growth, c)]
                if root_cols:
                    st.plotly_chart(group_bar(hi, lo, root_cols, "상위·하위 그룹의 뿌리 길이와 뿌리 질량 비교"), width="stretch")
                else:
                    st.info("뿌리(지하부) 열이 없어 뿌리 그래프는 건너뛰었어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ───────────────── 탭 3 ─────────────────
with tab3:
    try:
        st.header("가설 검증과 최종 결론 — 뿌리의 양분 흡수와 생육 차이")
        if not has_growth or RATIO not in growth.columns or not is_num(growth, "생중량"):
            st.info("생육 결과 CSV에 생중량(또는 지상부·지하부 질량), 지하부 길이, 지하부 질량이 있어야 해요.")
        else:
            hi, lo = growth[growth["생육 그룹"] == HI], growth[growth["생육 그룹"] == LO]
            st.subheader("① 가설 검증")
            st.caption("📘 뿌리 질량 대비 길이 = 뿌리 길이(mm) ÷ 뿌리 질량(g). 값이 클수록 같은 질량에 비해 뿌리가 길다(가늘고 길다).")
            hr, lr = clean(hi[RATIO]), clean(lo[RATIO])
            if len(hr) >= 2 and len(lr) >= 2:
                c1, c2, c3 = st.columns(3)
                c1.metric("상위 그룹 뿌리 질량 대비 길이", f"{hr.mean():.2f} mm/g")
                c2.metric("하위 그룹 뿌리 질량 대비 길이", f"{lr.mean():.2f} mm/g")
                c3.metric("하위 ÷ 상위", f"약 {lr.mean() / hr.mean():.1f}배" if hr.mean() > 0 else "-")
                st.plotly_chart(group_bar(hi, lo, [RATIO], "상위·하위 그룹의 뿌리 질량 대비 길이 비교"), width="stretch")
            else:
                st.info("그룹별 값이 2개 미만이라 그룹 비교는 건너뛰었어요.")

            fig, r, p = scatter_trend(growth, "생중량", RATIO, "생중량과 뿌리 질량 대비 길이의 관계")
            if fig is None:
                st.info("산점도를 그릴 값이 부족해요(3개체 이상 필요).")
            else:
                st.plotly_chart(fig, width="stretch")
                st.metric("상관계수 r", f"{r:.3f}", f"p = {p:.4f}", delta_color="off")
                st.caption("📘 상관계수 r: -1~1 사이 값. 0에 가까우면 관계가 약하고, 음수(-)면 한쪽이 클수록 다른 쪽이 작은 관계. "
                           "상관은 '함께 변한다'는 뜻이지 원인과 결과를 뜻하지는 않는다.")
                if len(hr) >= 1 and len(lr) >= 1 and lr.mean() > hr.mean() and r < 0:
                    st.success(f"하위 그룹의 값이 더 크고 생중량과 음의 상관관계(r = {r:.3f})를 보였으므로, 가설은 수치상 지지되었다.")
                else:
                    st.info("이 데이터에서는 가설이 수치상 지지되지 않았다.")
                with st.expander("🔍 비판적 점검: 생중량 안에 뿌리 질량이 들어 있다면?"):
                    st.markdown("생중량 = 지상부 질량 + 지하부 질량이고, 뿌리 질량 대비 길이의 분모도 지하부 질량이다. "
                                "따라서 생중량과 이 지표 사이의 음의 상관은 일부가 계산 구조 때문에 생길 수 있다. "
                                "뿌리 질량이 들어가지 않은 **지상부 질량**으로도 같은 경향이 나오는지 확인해 보자.")
                    fig2, r2, p2 = scatter_trend(growth, "지상부 질량", RATIO, "지상부 질량과 뿌리 질량 대비 길이의 관계") if is_num(growth, "지상부 질량") else (None, None, None)
                    if fig2 is not None:
                        st.plotly_chart(fig2, width="stretch")
                        st.markdown(f"지상부 질량 기준 상관계수 r = **{r2:.3f}** (p = {p2:.4f})")
                    else:
                        st.info("지상부 질량 열이 없어 이 점검은 건너뛰었어요.")

            st.subheader("② 처음 예상과 달랐던 결과")
            p_len = welch(hi["지하부 길이"], lo["지하부 길이"])
            p_mass = welch(hi["지하부 질량"], lo["지하부 질량"])
            if p_len is not None and p_mass is not None:
                len_txt = (f"두 그룹의 뿌리 길이는 유의한 차이가 없었고(p = {p_len:.3f})" if p_len >= 0.05
                           else f"두 그룹의 뿌리 길이에 차이가 있었고(p = {p_len:.4f})")
                mass_txt = (f"뿌리 질량은 하위 그룹에서 더 적었다(p = {p_mass:.4f})." if clean(lo["지하부 질량"]).mean() < clean(hi["지하부 질량"]).mean()
                            else f"뿌리 질량은 하위 그룹에서 더 적지 않았다(p = {p_mass:.4f}).")
                st.markdown(f"- 덜 자란 개체의 뿌리가 더 길게 자랐을 것이라는 예상은 {'확인되지 않았다' if p_len >= 0.05 else '추가 확인이 필요하다'}.\n- {len_txt}, {mass_txt}")
                if p_len >= 0.05 and p_mass < 0.05:
                    st.markdown("- 즉, 덜 자란 개체의 뿌리 질량 대비 길이가 큰 이유는 **뿌리가 더 길어서가 아니라 뿌리 질량이 적었기 때문**이다.")
            else:
                st.info("그룹별 뿌리 값이 부족해 차이 검정을 건너뛰었어요.")

            st.subheader("③ 양분과 생육의 관계에 대한 해석")
            st.markdown("- 뿌리는 양분을 흡수하는 기관이므로 뿌리의 발달 상태는 식물의 생육과 관련될 수 있다.\n"
                        "- 이번 연구에서는 덜 자란 그룹의 뿌리 질량이 더 적다는 사실을 확인했다.\n"
                        "- 그러나 개체별 양분 흡수량을 직접 측정하지 않았으므로, 뿌리 질량 차이가 실제 양분 흡수량의 차이로 이어졌다고 단정할 수는 없다.")

            st.subheader("④ 최종 결론 및 후속 탐구")
            st.markdown("- 덜 자란 그린란드고추냉이는 잎 수와 뿌리 질량이 상대적으로 적었다.\n"
                        "- 뿌리 질량 대비 길이는 높았지만, 실제 뿌리 길이에는 유의한 차이가 없었다.\n"
                        "- 앞으로는 양액의 양분 농도와 개체별 양분 흡수량, 재배 위치에 따른 환경 차이를 함께 조사할 필요가 있다.")
            st.caption("※ 위 결론 문장은 이 탭의 수치·검정 결과와 함께 읽어야 한다. 다른 학교 데이터라면 결과가 달라질 수 있다.")

            out_cols = ["개체"] + num_cols(growth) + ["생육 그룹"]
            out = growth[[c for c in out_cols if c in growth.columns]].copy()
            if RATIO in out.columns:
                out[RATIO] = out[RATIO].round(2)
            st.download_button("📥 개체별 분석 결과 내려받기 (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{school}_개체별_분석결과.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

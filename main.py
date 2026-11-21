# polar-plant-dashboard/main.py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from scipy import stats
import streamlit as st

# ─────────────────────────────────────────────
# 1. 한글 폰트 설정 (Streamlit Cloud 대응)
# ─────────────────────────────────────────────
def setup_korean_font():
    """한글 폰트를 찾아 matplotlib에 등록한다."""
    candidates = [
        'NanumGothic', 'NanumBarunGothic', 'NanumSquare',
        'AppleGothic', 'MalgunGothic', 'Noto Sans CJK KR',
    ]
    for name in candidates:
        try:
            rc('font', family=name)
            rc('axes', unicode_minus=False)
            # 실제 활성화되어 있는지 확인
            fig = plt.figure()
            fig.text(0.5, 0.5, '한글', fontfamily=name)
            plt.close(fig)
            return
        except Exception:
            continue
    # 시스템 폰트 중 한글 포함 이름으로 폴백
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Nano', 'Apple', 'Malgun', 'Noto', 'CJK', 'Gothic']):
            try:
                rc('font', family=f.name)
                rc('axes', unicode_minus=False)
                return
            except Exception:
                continue
    rc('font', family='sans-serif')
    rc('axes', unicode_minus=False)

setup_korean_font()

# ─────────────────────────────────────────────
# 2. 통계 용어 설명 (고등학생용)
# ─────────────────────────────────────────────
TERM_NOTES = {
    '평균': '모든 값을 더해 개수로 나눈 값 — 자료의 대표값',
    '표준편차': '값이 평균에서 얼마나 흩어져 있는지 나타내는 수치 — 클수록 들쑥날쭉',
    '분산': '표준편차를 제곱한 값 — 편차 제곱의 평균',
    '중앙값': '자료를 크기순으로 나열했을 때 정가운데 값 — 극단값 영향을 덜 받음',
    '상관계수(r)': '두 변수가 함께 움직이는 정도 (−1 ~ +1) — ±1에 가까울수록 강한 관계',
    'p-value': '우연으로도 그 정도 결과가 나올 확률 — 작을수록 의미 있는 차이/관계',
}

# ─────────────────────────────────────────────
# 3. CSV 읽기 — 인코딩 유연 처리
# ─────────────────────────────────────────────
def read_csv_flexible(file_bytes):
    """utf-8-sig → cp949 → utf-8 순으로 시도하여 CSV를 읽는다."""
    for enc in ['utf-8-sig', 'cp949', 'utf-8']:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("CSV 인코딩을 인식하지 못했습니다. utf-8-sig 또는 cp949로 저장된 파일인지 확인하세요.")

# ─────────────────────────────────────────────
# 4. 환경 데이터 열 이름 매핑
# ─────────────────────────────────────────────
def map_env_columns(df):
    """
    학교마다 다른 환경 데이터 열 이름을 표준 키로 매핑한다.
    반환값: dict — key=표준 라벨, value=실제 열 이름 (또는 None)
    """
    # 공백 제거한 열 이름 → 원래 열 이름 매핑
    col_lookup = {c.strip(): c for c in df.columns}

    def find_col(*keys):
        for k in keys:
            if k in col_lookup:
                return col_lookup[k]
        return None

    return {
        'time_col':      find_col('time', '시간', 'date', '일시', '타임스탬프'),
        'temp_col':      find_col('temperature', '온도', 'temp'),
        'humidity_col':  find_col('humidity', '습도', 'rh'),
        'ec_col':        find_col('ec', 'EC', '전기전도도'),
        'ph_col':        find_col('ph', 'PH', '수소이온농도', '산도'),
        'water_temp_col':find_col('water_temp', '수온', 'water temperature'),
        'co2_col':       find_col('co2', 'CO2', '이산화탄소'),
    }

# ─────────────────────────────────────────────
# 5. 생육 결과 데이터 열 이름 매핑
# ─────────────────────────────────────────────
def map_growth_columns(df):
    """
    학교마다 다른 생육 결과 데이터 열 이름을 표준 키로 매핑한다.
    """
    col_lookup = {c.strip(): c for c in df.columns}

    def find_col(*keys):
        for k in keys:
            if k in col_lookup:
                return col_lookup[k]
        return None

    return {
        'id_col':         find_col('개체번호', '개체번호 ', 'plant_id', 'id', '번호'),
        'leaf_col':       find_col('잎 수(장)', '잎수', '잎 수', 'leaf_count', '잎'),
        'above_len_col':  find_col('지상부 길이(mm)', '지상부길이(mm)', '지상부 길이',
                                   '지상부길이', 'shoot_length',
                                   '지상부길이 (mm)', '지상부 길이 (mm)'),
        'below_len_col':  find_col('지하부길이(mm)', '지하부 길이(mm)', '지하부길이',
                                   '지하부 길이', 'root_length',
                                   '지하부길이 (mm)', '지하부 길이 (mm)'),
        'fresh_col':      find_col('생중량(g)', '생중량', 'fresh_weight',
                                   '생체중량(g)', '생체중량'),
        'shoot_mass_col': find_col('지상부질량(g)', '지상부질량', '지상부 질량(g)',
                                   'shoot_mass', '지상부질량 (g)', '지상부 질량 (g)'),
        'root_mass_col':  find_col('지하부질량(g)', '지하부질량', '지하부 질량(g)',
                                   'root_mass', '지하부질량 (g)', '지하부 질량 (g)'),
    }

# ─────────────────────────────────────────────
# 6. 환경 데이터 후처리
# ─────────────────────────────────────────────
def process_env_data(df, mapped):
    """시간 열을 datetime으로 변환하고 시간순 정렬한다."""
    time_col = mapped['time_col']
    if time_col is None:
        return df, None

    df = df.copy()
    try:
        df['_time'] = pd.to_datetime(df[time_col], errors='coerce')
    except Exception:
        # '20260715_0' 형식 폴백
        df['_time'] = pd.to_datetime(
            df[time_col].astype(str).str.replace('_', ' ', regex=False),
            format='%Y%m%d %H', errors='coerce'
        )

    df = df.sort_values('_time').reset_index(drop=True)
    return df, '_time'

# ─────────────────────────────────────────────
# 7. 생육 데이터 후처리
# ─────────────────────────────────────────────
def process_growth_data(df_raw, mapped):
    """
    생육 데이터를 정리한다.
    - 모든 수치형 열을 숫자형으로 강제 변환
    - 생중량(fresh_col)이 없으면 지상부질량 + 지하부질량으로 계산
    - 개체번호를 문자열로 통일
    """
    df = df_raw.copy()
    d = mapped

    num_cols = []

    for col_key, attr in [
        ('leaf_col', 'leaf_col'),
        ('above_len_col', 'above_len_col'),
        ('below_len_col', 'below_len_col'),
        ('fresh_col', 'fresh_col'),
        ('shoot_mass_col', 'shoot_mass_col'),
        ('root_mass_col', 'root_mass_col'),
    ]:
        col = d.get(attr)
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            num_cols.append(col)

    # 생중량 결측 처리: 지상부질량 + 지하부질량
    if (d['fresh_col'] is None or d['fresh_col'] not in df.columns) \
       and d['shoot_mass_col'] and d['root_mass_col'] \
       and d['shoot_mass_col'] in df.columns and d['root_mass_col'] in df.columns:
        df['생중량(g)'] = df[d['shoot_mass_col']] + df[d['root_mass_col']]
        d['fresh_col'] = '생중량(g)'
        if '생중량(g)' not in num_cols:
            num_cols.append('생중량(g)')

    # 개체번호를 문자열로
    if d['id_col'] and d['id_col'] in df.columns:
        df['_id_str'] = df[d['id_col']].astype(str)

    # 결측 제거로 분석용 데이터프레임
    analysis_cols = [c for c in num_cols if c in df.columns]
    df_clean = df.dropna(subset=analysis_cols).reset_index(drop=True) if analysis_cols else df.copy()

    return df, df_clean, d, num_cols

# ─────────────────────────────────────────────
# 8. Streamlit 대시보드
# ─────────────────────────────────────────────
st.set_page_config(page_title='극지식물 생육 차이 분석 대시보드', layout='wide')
st.title('🌿 극지식물 개체별 생육 차이 분석 대시보드')
st.caption('CSV를 업로드하면 학교마다 다른 열 이름·인코딩을 자동으로 처리하여 분석합니다.')

# ── 사이드바 ──
st.sidebar.header('📁 데이터 업로드')
env_file = st.sidebar.file_uploader(
    '환경 데이터 (CSV)',
    type=['csv'],
    help='time, temperature, humidity 등이 포함된 파일 (utf-8-sig / cp949 지원)',
)
growth_file = st.sidebar.file_uploader(
    '생육 결과 데이터 (CSV)',
    type=['csv'],
    help='개체번호, 잎 수, 길이, 질량 등이 포함된 파일',
)

# ── 데이터 로드 ──
env_df_raw = None
growth_df_raw = None
env_mapped = None
growth_mapped = None
env_processed = None
env_time_col = None
growth_clean = None
growth_d = None
growth_num_cols = None

if env_file is not None:
    try:
        env_bytes = env_file.read()
        env_df_raw = read_csv_flexible(env_bytes)
        env_mapped = map_env_columns(env_df_raw)
        env_processed, env_time_col = process_env_data(env_df_raw, env_mapped)
        st.sidebar.success(f'✅ 환경 데이터 로드 — {len(env_df_raw)}행 × {len(env_df_raw.columns)}열')
        st.sidebar.caption(f'열: {", ".join(env_df_raw.columns)}')
    except Exception as e:
        st.sidebar.error(f'환경 데이터 읽기 오류: {e}')

if growth_file is not None:
    try:
        growth_bytes = growth_file.read()
        growth_df_raw = read_csv_flexible(growth_bytes)
        growth_mapped = map_growth_columns(growth_df_raw)
        _, growth_clean, growth_d, growth_num_cols = process_growth_data(
            growth_df_raw, growth_mapped
        )
        st.sidebar.success(f'✅ 생육 데이터 로드 — {len(growth_df_raw)}개체 × {len(growth_df_raw.columns)}열')
        st.sidebar.caption(f'열: {", ".join(growth_df_raw.columns)}')
    except Exception as e:
        st.sidebar.error(f'생육 데이터 읽기 오류: {e}')

if env_file is None and growth_file is None:
    st.info(
        '📌 왼쪽 사이드바에서 CSV 파일 2개를 업로드하세요.\n\n'
        '예시 구조:\n'
        '- 환경: time, temperature, humidity, ec, ph, water_temp, co2\n'
        '- 생육: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), '
        '생중량(g), 지상부질량(g), 지하부질량(g)'
    )
    st.stop()

# ══════════════════════════════════════════════════════
#  SECTION 1 — 환경 데이터 요약
# ══════════════════════════════════════════════════════
if env_processed is not None and env_mapped is not None:
    st.header('🌡️ 환경 데이터 요약')

    # (표시라벨, 실제열이름) 쌍을 만든다 — DataFrame 인덱싱에는 실제열이름만 사용
    env_col_pairs = []
    if env_mapped['temp_col']:
        env_col_pairs.append(('온도', env_mapped['temp_col']))
    if env_mapped['humidity_col']:
        env_col_pairs.append(('습도', env_mapped['humidity_col']))
    if env_mapped['ec_col']:
        env_col_pairs.append(('EC', env_mapped['ec_col']))
    if env_mapped['ph_col']:
        env_col_pairs.append(('pH', env_mapped['ph_col']))
    if env_mapped['water_temp_col']:
        env_col_pairs.append(('수온', env_mapped['water_temp_col']))
    if env_mapped['co2_col']:
        env_col_pairs.append(('CO₂', env_mapped['co2_col']))

    if env_col_pairs:
        st.caption('※ 센서 미측정 구간은 결측값(NaN)으로 표시됩니다.')

        # ★ 수정: 실제 열 이름 리스트만 추출하여 DataFrame 인덱싱
        env_col_names = [col_name for _, col_name in env_col_pairs]
        st.dataframe(env_processed[env_col_names].describe().T, use_container_width=True)

        # 환경 시계열 차트
        st.subheader('환경 변수 시계열')
        fig, axes = plt.subplots(
            len(env_col_pairs), 1,
            figsize=(10, 2.5 * len(env_col_pairs)),
            sharex=True,
        )
        if len(env_col_pairs) == 1:
            axes = [axes]

        for ax, (label, col_name) in zip(axes, env_col_pairs):
            valid = env_processed[col_name].dropna()
            if env_time_col and len(valid) > 0:
                t_valid = env_processed.loc[valid.index, env_time_col]
                ax.plot(t_valid, valid.values, linewidth=0.7, color='#2a7f8c')
            else:
                ax.plot(valid.values, linewidth=0.7, color='#2a7f8c')
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel('시간', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════
#  SECTION 2 — 생육 결과: 개체별 비교
# ══════════════════════════════════════════════════════
if growth_clean is not None and growth_d is not None:
    st.header('🌱 생육 결과 — 개체별 비교')
    st.caption(f'사용 중인 열: {", ".join(growth_df_raw.columns)}')

    # 지표 라벨 ↔ 실제 열 이름 매핑
    metric_map = []  # (표시라벨, 단위, 실제열이름)
    for col_key, label, unit in [
        ('leaf_col',        '잎 수',        '장'),
        ('above_len_col',   '지상부 길이',  'mm'),
        ('below_len_col',   '지하부 길이',  'mm'),
        ('fresh_col',       '생중량',       'g'),
        ('shoot_mass_col',  '지상부 질량',  'g'),
        ('root_mass_col',   '지하부 질량',  'g'),
    ]:
        col = growth_d.get(col_key)
        if col and col in growth_clean.columns:
            metric_map.append((label, unit, col))

    # 요약 통계 테이블
    if metric_map:
        st.subheader('📊 개체별 생육 지표 요약 (평균 ± 표준편차)')
        summary_rows = []
        for label, unit, col in metric_map:
            series = growth_clean[col].dropna()
            if len(series) == 0:
                continue
            summary_rows.append({
                '지표': label,
                '단위': unit,
                '개체 수': len(series),
                '평균': f'{series.mean():.2f}',
                '표준편차': f'{series.std(ddof=0):.2f}',
                '중앙값': f'{series.median():.2f}',
                '최소': f'{series.min():.2f}',
                '최대': f'{series.max():.2f}',
            })

        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                '💡 표준편차: 값이 평균에서 얼마나 흩어져 있는지 나타내는 수치 — '
                '클수록 개체 간 차이가 큼 (분산 = 표준편차²)'
            )

    # 개체별 막대 차트
    if metric_map:
        st.subheader('개별 개체 비교 막대 차트')
        metric_labels = [m[0] for m in metric_map]
        selected_label = st.selectbox('비교할 지표 선택', metric_labels)

        selected_col = next((m[2] for m in metric_map if m[0] == selected_label), None)
        selected_unit = next((m[1] for m in metric_map if m[0] == selected_label), '')

        if selected_col:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ids = growth_clean['_id_str'].values
            vals = growth_clean[selected_col].values
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ids)))
            ax.bar(ids, vals, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('개체번호', fontsize=10)
            ax.set_ylabel(f'{selected_label} ({selected_unit})', fontsize=10)
            mean_val = np.nanmean(vals)
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
                       label=f'평균: {mean_val:.2f}')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(labelsize=8)
            plt.xticks(rotation=45, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # 개체별 데이터 테이블
    if growth_num_cols:
        st.subheader('📋 개체별 전체 데이터')
        display_cols = ['_id_str'] + [
            c for c in growth_num_cols if c in growth_clean.columns
        ]
        renamed = growth_clean[display_cols].copy()
        if '_id_str' in renamed.columns:
            renamed = renamed.rename(columns={'_id_str': '개체번호'})
        st.dataframe(renamed, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
#  SECTION 3 — 상관 분석
# ══════════════════════════════════════════════════════
if growth_clean is not None and growth_num_cols is not None:
    st.header('🔗 생육 지표 간 상관 분석')
    st.caption(
        '💡 상관계수(r): 두 변수가 함께 움직이는 정도 (−1 ~ +1). '
        '+1에 가까울수록 함께 증가, −1에 가까울수록 반대로 움직임, '
        '0에 가까우면 관계 약함.'
    )

    numeric_data = growth_clean[growth_num_cols].dropna()
    if len(numeric_data) >= 3 and len(numeric_data.columns) >= 2:
        corr_matrix = numeric_data.corr(method='pearson')

        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(
            corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto'
        )
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(
            corr_matrix.columns, rotation=45, ha='right', fontsize=8
        )
        ax.set_yticklabels(corr_matrix.index, fontsize=8)
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                ax.text(
                    j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if abs(val) > 0.5 else 'black',
                )
        fig.colorbar(im, ax=ax, shrink=0.85, label='상관계수 r')
        ax.set_title('Pearson 상관계수 행렬', fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # p-value 테이블
        st.subheader('상관계수 및 p-value')
        p_rows = []
        for i in range(len(corr_matrix.index)):
            for j in range(i + 1, len(corr_matrix.columns)):
                x = numeric_data[corr_matrix.index[i]]
                y = numeric_data[corr_matrix.columns[j]]
                r_val, p_val = stats.pearsonr(x, y)
                if r_val > 0.7:
                    interp = '강한 양의 상관'
                elif r_val > 0.3:
                    interp = '약한 양의 상관'
                elif abs(r_val) < 0.3:
                    interp = '상관 거의 없음'
                elif r_val > -0.7:
                    interp = '약한 음의 상관'
                else:
                    interp = '강한 음의 상관'
                p_rows.append({
                    '지표 A': corr_matrix.index[i],
                    '지표 B': corr_matrix.columns[j],
                    'r': f'{r_val:.3f}',
                    'p-value': f'{p_val:.4f}',
                    '해석': interp,
                })

        if p_rows:
            st.dataframe(
                pd.DataFrame(p_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                '💡 p-value: 이 정도 상관이 우연으로도 나올 확률. '
                '보통 0.05보다 작으면 의미 있는 관계로 본다.'
            )

# ══════════════════════════════════════════════════════
#  SECTION 4 — 환경-생육 비교 요약
# ══════════════════════════════════════════════════════
if env_processed is not None and growth_clean is not None and env_time_col is not None:
    st.header('🌡️ 생육 vs 환경: 지표별 평균')

    st.caption(
        '전체 관측 기간의 환경 평균과 개체별 생육 지표 평균을 나란히 비교합니다.'
    )

    col1, col2 = st.columns(2)

    env_summary = [
        ('평균 온도', env_mapped['temp_col'],   '℃'),
        ('평균 습도', env_mapped['humidity_col'], '%'),
        ('평균 EC',   env_mapped['ec_col'],      'dS/m'),
        ('평균 pH',   env_mapped['ph_col'],      ''),
        ('평균 CO₂',  env_mapped['co2_col'],     'ppm'),
    ]
    env_summary = [(lbl, c, u) for lbl, c, u in env_summary if c and c in env_processed.columns]

    with col1:
        st.markdown('**전체 기간 환경 평균**')
        for lbl, col, unit in env_summary:
            m = env_processed[col].mean()
            s = env_processed[col].std(ddof=0)
            st.write(f'- **{lbl}**: {m:.2f} ± {s:.2f} {unit}')

    with col2:
        st.markdown('**개체별 생육 지표 평균**')
        for _, _, col in metric_map if metric_map else []:
            if col in growth_clean.columns:
                m = growth_clean[col].mean()
                s = growth_clean[col].std(ddof=0)
                label, unit, _ = next(m for m in metric_map if m[2] == col)
                st.write(f'- **{label}**: {m:.2f} ± {s:.2f} {unit}')

# ── 하단 정보 ──
st.divider()
st.caption(
    '**처리 특징**\n'
    '- 인코딩: utf-8-sig → cp949 → utf-8 자동 폴백\n'
    '- 열 이름: 영문·한글·단위·띄어쓰기 포함 등 여러 변형을 유연 매핑\n'
    '- 한글 폰트: matplotlib에서 Nanum·Apple·Malgun·Noto 계열 자동 감지\n'
    '- 생중량(g) 열이 없으면 지상부질량 + 지하부질량으로 자동 계산\n'
    '- 통계 용어 설명은 지표 표·캡션에 짧게 표시'
)


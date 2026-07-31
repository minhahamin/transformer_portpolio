import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

from lib import review_analysis as ra

st.set_page_config(page_title="영화 리뷰 분석", page_icon="🎬", layout="wide")
st.title("🎬 실습1 — 한국어 영화 리뷰 분석 및 시각화")
st.caption("Okt 형태소 분석 + 불용어 필터링으로 리뷰 코퍼스의 핵심 단어를 추출합니다.")

with st.sidebar:
    st.header("설정")
    mode = st.radio("불용어 필터", ["기본", "개선"], index=1)
    top_n = st.slider("표시할 상위 단어 수", 10, 60, 30)
    min_freq = st.slider("최소 빈도 (개선 필터만 적용)", 1, 10, 5, disabled=(mode == "기본"))

mode_key = "basic" if mode == "기본" else "improved"

try:
    counter = ra.analyze(mode_key, min_freq=min_freq if mode_key == "improved" else 1)
except Exception as e:
    st.error(
        "형태소 분석기(Okt)를 불러오지 못했습니다. Streamlit Cloud에 배포한 경우 "
        "`packages.txt`에 `default-jdk`가 포함되어 있는지 확인하세요.\n\n"
        f"오류: {e}"
    )
    st.stop()

if not counter:
    st.warning("추출된 단어가 없습니다. 최소 빈도를 낮춰보세요.")
    st.stop()

top_items = counter.most_common(top_n)
words, freqs = zip(*top_items)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📊 상위 단어 빈도")
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    ax.barh(words[::-1], freqs[::-1], color="#4C72B0")
    ax.set_xlabel("빈도")
    font_path = ra.get_korean_font_path()
    if font_path:
        import matplotlib.font_manager as fm
        prop = fm.FontProperties(fname=font_path)
        ax.set_yticklabels(words[::-1], fontproperties=prop)
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("☁️ 워드클라우드")
    font_path = ra.get_korean_font_path()
    wc = WordCloud(
        font_path=font_path,
        width=600,
        height=500,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(dict(top_items))
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.imshow(wc, interpolation="bilinear")
    ax2.axis("off")
    st.pyplot(fig2, use_container_width=True)

st.subheader("📋 통계 요약")
c1, c2, c3 = st.columns(3)
c1.metric("고유 단어 수", f"{len(counter):,}")
c2.metric("전체 토큰 수", f"{sum(counter.values()):,}")
c3.metric("최고 빈도 단어", f"{words[0]} ({freqs[0]:,}회)")

with st.expander("상위 단어 전체 목록 보기"):
    st.dataframe(
        {"단어": list(words), "빈도": list(freqs)},
        use_container_width=True,
        hide_index=True,
    )

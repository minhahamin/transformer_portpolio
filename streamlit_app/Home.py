import streamlit as st

st.set_page_config(
    page_title="Transformer 실습 포트폴리오",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Transformer 실습 포트폴리오")
st.caption("AI Human 강의 · Part II. Transformer — 실습 1 · 2 · 6 · 9 통합 데모")

st.markdown(
    """
왼쪽 사이드바에서 4개 실습 중 하나를 선택하세요.

| 실습 | 내용 | 핵심 기술 |
|---|---|---|
| 🎬 실습1 | 한국어 영화 리뷰 분석 | Okt 형태소 분석, BarPlot/WordCloud |
| 🌐 실습2 | Seq2Seq 번역 | BiGRU Encoder, Luong Attention, Beam Search |
| 👗 실습6 | CLIP 패션 스타일 검색 | CLIP 텍스트-이미지 임베딩, 코사인 유사도 |
| 🖼️ 실습9 | 이미지 캡셔닝 | BLIP, 한국어 번역 |

---

**참고 (배포판 vs 원본 노트북 차이)**

무료 호스팅(Streamlit Community Cloud)의 메모리 제약 때문에 아래 두 가지를 배포판에서만 가볍게 교체했습니다.
각 실습 폴더의 원본 `.ipynb`는 수정되지 않았습니다.

- CLIP 로드 방식: `git+openai/CLIP` → `transformers.CLIPModel`(`openai/clip-vit-base-patch16`, 동일 가중치)
- 실습9 번역 모델: `nllb-200-distilled-600M`(2.4GB) → `Helsinki-NLP/opus-mt-tc-big-en-ko`(약 800MB, 번역 품질은 다소 낮아질 수 있음)

무거운 모델(BLIP, 번역 모델, CLIP)은 해당 페이지를 처음 열 때만 다운로드/로드됩니다.
"""
)

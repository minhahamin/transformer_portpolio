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

각 실습 폴더의 원본 `.ipynb`는 수정되지 않았습니다. 배포판 코드에서만 아래처럼 교체했습니다.

- CLIP 로드 방식: `git+openai/CLIP` → `transformers.CLIPModel`(`openai/clip-vit-base-patch16`, 동일 가중치, 배포 신뢰성 목적)
- 실습2 번역 모델: 매번 즉석에서 3 epoch만 학습하면 품질이 너무 낮아, 로컬에서 60 epoch 미리 학습한 가중치를 로드하도록 변경
- 실습9 번역 모델: 메모리를 아끼려고 더 가벼운 모델(opus-mt-tc-big-en-ko, ~800MB)을 시도했으나 번역 출력이 심각하게 깨져서(중국어 혼입) 원본과 동일한 `nllb-200-distilled-600M`(2.4GB)으로 되돌림 — 품질을 우선한 선택

무거운 모델(BLIP, 번역 모델, CLIP)은 해당 페이지를 처음 열 때만 다운로드/로드되고, CLIP ↔ BLIP+번역 페이지를 전환하면 이전 모델은 메모리에서 비웁니다. 그래도 무료 호스팅 RAM 한도를 넘으면 앱이 재시작될 수 있습니다.
"""
)

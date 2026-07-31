import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from lib import seq2seq_model as s2s

st.set_page_config(page_title="Seq2Seq 번역", page_icon="🌐", layout="wide")
st.title("🌐 실습2 — Seq2Seq 기반 번역 AI 모델링")
st.caption("BiGRU Encoder + Luong Attention + GRU Decoder, 한국어 → 영어 번역 (자체 학습 720문장)")

bundle = s2s.train_bundle()

st.info(
    f"학습 완료 — vocab 크기: 한국어 {len(bundle.vocab_ko):,} / 영어 {len(bundle.vocab_en):,} · "
    f"최종 loss: {bundle.losses[-1]:.4f} (3 epoch)",
    icon="✅",
)

with st.expander("Epoch별 학습 Loss"):
    st.line_chart({"loss": bundle.losses})

st.subheader("🇰🇷 → 🇺🇸 번역해보기")
st.caption("⚠️ 학습 데이터가 720문장뿐인 작은 토이 모델입니다. 학습 데이터에 없는 단어/구조는 번역이 부정확할 수 있습니다.")

sentence = st.text_input(
    "한국어 문장 입력", value="", placeholder="예: 안녕하세요, 오늘 날씨가 정말 좋네요"
)
decoding = st.radio("디코딩 방식", ["Greedy", "Beam Search"], horizontal=True)
beam_width = st.slider("Beam width", 2, 5, 3, disabled=(decoding == "Greedy"))

if st.button("번역하기", type="primary", disabled=not sentence.strip()):
    with st.spinner("번역 중..."):
        if decoding == "Greedy":
            result = bundle.translate(sentence)
        else:
            result = bundle.beam_search_translate(sentence, beam_width=beam_width)
    st.success(result if result else "(빈 번역 결과 — 학습 데이터에 없는 단어일 수 있습니다)")

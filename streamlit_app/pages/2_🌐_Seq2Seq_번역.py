import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import streamlit as st

from lib import review_analysis as ra
from lib import seq2seq_model as s2s

st.set_page_config(page_title="Seq2Seq 번역", page_icon="🌐", layout="wide")
st.title("🌐 실습2 — Seq2Seq 기반 번역 AI 모델링")
st.caption("BiGRU Encoder + Luong Attention + GRU Decoder, 한국어 → 영어 번역 (자체 학습 720문장)")

bundle = s2s.train_bundle()

st.info(
    f"학습 완료 — vocab 크기: 한국어 {len(bundle.vocab_ko):,} / 영어 {len(bundle.vocab_en):,} · "
    f"최종 loss: {bundle.losses[-1]:.4f} ({len(bundle.losses)} epoch)",
    icon="✅",
)

with st.expander("Epoch별 학습 Loss"):
    st.line_chart({"loss": bundle.losses})

st.subheader("🇰🇷 → 🇺🇸 번역해보기")
st.caption(
    "⚠️ 학습 데이터가 720문장뿐인 작은 토이 모델입니다. 720문장을 거의 암기하듯 학습해서, "
    "학습 문장과 같거나 아주 비슷한 표현은 정확히 번역되지만 완전히 새로운 조합은 그럴듯한 "
    "오역을 지어낼 수 있습니다. 디코더 확신도가 낮으면(평균 토큰 확률 85% 미만) 오역 대신 "
    "'학습되지 않은 표현'이라고 알려줍니다."
)

sentence = st.text_input(
    "한국어 문장 입력", value="어머니는 그들을 스케치하고 있다."
)
decoding = st.radio("디코딩 방식", ["Greedy", "Beam Search"], horizontal=True)
beam_width = st.slider("Beam width", 2, 5, 3, disabled=(decoding == "Greedy"))
show_attention = st.checkbox(
    "🔍 Attention 시각화 보기 (Greedy만 지원)", value=True, disabled=(decoding != "Greedy")
)

if st.button("번역하기", type="primary", disabled=not sentence.strip()):
    src_tokens = result_tokens = attn = None
    with st.spinner("번역 중..."):
        if decoding == "Greedy":
            if show_attention:
                result, src_tokens, result_tokens, attn = bundle.translate_with_attention(sentence)
            else:
                result = bundle.translate(sentence)
        else:
            result = bundle.beam_search_translate(sentence, beam_width=beam_width)

    if result == s2s.UNTRAINED_MESSAGE:
        st.warning(result, icon="🤷")
    else:
        st.success(result if result else "(빈 번역 결과)")

        if attn:
            st.subheader("🔥 Attention 히트맵")
            st.caption("디코더가 각 영어 단어를 생성할 때 어떤 한국어 형태소에 주목했는지 보여줍니다.")
            font_path = ra.get_korean_font_path()
            font_prop = None
            if font_path:
                import matplotlib.font_manager as fm
                font_prop = fm.FontProperties(fname=font_path)

            fig, ax = plt.subplots(figsize=(max(4, len(src_tokens) * 0.7), max(3, len(result_tokens) * 0.6)))
            im = ax.imshow(attn, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontproperties=font_prop)
            ax.set_yticks(range(len(result_tokens)))
            ax.set_yticklabels(result_tokens)
            ax.set_xlabel("입력 (한국어 형태소)", fontproperties=font_prop)
            ax.set_ylabel("출력 (영어 단어)")
            fig.colorbar(im, ax=ax, label="Attention weight")
            st.pyplot(fig, width="content")

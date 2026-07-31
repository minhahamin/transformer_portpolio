import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import streamlit as st

from lib import captioning as cap
from lib import memory_manager

st.set_page_config(page_title="이미지 캡셔닝", page_icon="🖼️", layout="wide")
memory_manager.activate("caption")

st.title("🖼️ 실습9 — BLIP Image Captioning")
st.caption("이미지를 업로드하면 BLIP이 영어 캡션을 생성하고, 원하면 한국어로 번역합니다. 여러 장을 한 번에 올려 배치로 처리할 수 있습니다.")

st.caption(
    "ℹ️ 번역 모델은 원본 노트북과 동일한 NLLB-600M입니다. 더 가벼운 대안(opus-mt-tc-big-en-ko, "
    "~800MB)을 시도했으나 번역 품질이 심각하게 낮아(중국어 단어/문장부호가 섞여 나옴) 품질을 우선해 되돌렸습니다. "
    "다만 그대로 불러오면 번역 한 번에 메모리가 3GB 가까이 치솟는 문제가 있어, 같은 가중치를 "
    "CTranslate2로 int8 변환해 사용합니다(번역 품질은 거의 동일, 메모리는 약 1.3GB로 유지). "
    "BLIP·번역 모델 모두 처음 사용할 때만 다운로드되며, 번역을 요청하면 BLIP을 메모리에서 비운 뒤 로드합니다."
)

uploaded_files = st.file_uploader(
    "이미지를 업로드하세요 (여러 장 선택 가능)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
translate = st.checkbox("한국어로 번역", value=False)

if uploaded_files:
    images = [Image.open(f).convert("RGB") for f in uploaded_files]

    # BLIP으로 전체 캡션을 먼저 다 뽑은 뒤(BLIP 1회 로드 유지), 번역이 필요하면 그 다음에
    # NLLB로 한꺼번에 번역합니다 — 이미지마다 BLIP↔NLLB를 번갈아 로드하면 훨씬 느려집니다.
    with st.spinner(f"{len(images)}장 캡션 생성 중..."):
        captions = [cap.generate_caption(img) for img in images]

    koreans = [None] * len(images)
    if translate:
        with st.spinner("한국어로 번역 중..."):
            koreans = [cap.translate_to_korean(c) for c in captions]

    for image, caption, korean in zip(images, captions, koreans):
        col_img, col_text = st.columns([1, 2])
        with col_img:
            st.image(image, width="stretch")
        with col_text:
            st.markdown(f"**🇺🇸 영어**: {caption}")
            if korean is not None:
                st.markdown(f"**🇰🇷 한국어**: {korean}")
        st.divider()
else:
    st.info("이미지를 업로드하면 결과가 여기에 표시됩니다.")

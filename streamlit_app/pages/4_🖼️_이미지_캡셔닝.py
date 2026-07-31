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
st.caption("이미지를 업로드하면 BLIP이 영어 캡션을 생성하고, 원하면 한국어로 번역합니다.")

st.caption(
    "ℹ️ 번역 모델은 원본 노트북과 동일한 NLLB-600M(2.4GB)입니다. 더 가벼운 대안(opus-mt-tc-big-en-ko, "
    "~800MB)을 시도했으나 번역 품질이 심각하게 낮아(중국어 단어/문장부호가 섞여 나옴) 품질을 우선해 되돌렸습니다. "
    "BLIP·번역 모델 모두 처음 사용할 때만 다운로드되며, 두 모델을 합치면 메모리 사용량이 커서 "
    "무료 호스팅에서는 간헐적으로 재시작될 수 있습니다."
)

uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
translate = st.checkbox("한국어로 번역", value=False)

if uploaded is not None:
    image = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, use_container_width=True)

    with col_result:
        with st.spinner("캡션 생성 중..."):
            caption = cap.generate_caption(image)
        st.markdown("**🇺🇸 영어 캡션**")
        st.success(caption)

        if translate:
            with st.spinner("한국어로 번역 중..."):
                korean = cap.translate_to_korean(caption)
            st.markdown("**🇰🇷 한국어 번역**")
            st.success(korean)
else:
    st.info("이미지를 업로드하면 결과가 여기에 표시됩니다.")

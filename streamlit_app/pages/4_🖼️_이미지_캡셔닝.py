import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import streamlit as st

from lib import captioning as cap

st.set_page_config(page_title="이미지 캡셔닝", page_icon="🖼️", layout="wide")
st.title("🖼️ 실습9 — BLIP Image Captioning")
st.caption("이미지를 업로드하면 BLIP이 영어 캡션을 생성하고, 원하면 한국어로 번역합니다.")

st.caption(
    "ℹ️ 배포판은 원본 노트북의 번역 모델(NLLB-600M, 2.4GB) 대신 더 가벼운 "
    "Helsinki-NLP/opus-mt-tc-big-en-ko(약 800MB)를 사용합니다. 무료 호스팅 메모리 제약과 "
    "맞바꾼 선택이라 번역 품질이 NLLB보다 들쭉날쭉할 수 있습니다. 두 모델 모두 처음 사용할 때만 다운로드됩니다."
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

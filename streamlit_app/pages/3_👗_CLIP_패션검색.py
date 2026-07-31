import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import streamlit as st

from lib import clip_search as cs
from lib import memory_manager

st.set_page_config(page_title="CLIP 패션 검색", page_icon="👗", layout="wide")
memory_manager.activate("clip")

st.title("👗 실습6 — CLIP 멀티모달 패션 스타일 검색")
st.caption("텍스트 설명 또는 이미지로 12가지 패션 아이템 중 가장 잘 어울리는 스타일을 찾습니다.")

if not cs.get_unsplash_key():
    missing_local = [
        name for name in cs.IMAGE_QUERIES
        if not (cs.CACHE_DIR / f"{name.replace(' ', '_')}.jpg").exists()
    ]
    st.warning(
        "Unsplash API 키(`UNSPLASH_API_KEY`)가 secrets에 설정되어 있지 않습니다. "
        "캐시된 이미지가 없으면 해당 패션 아이템은 검색에서 제외됩니다. "
        "`.streamlit/secrets.toml`(로컬) 또는 Streamlit Cloud의 Secrets 설정에 키를 추가하세요.",
        icon="⚠️",
    )

with st.spinner(""):
    index = cs.build_fashion_index()

if index["missing_count"]:
    st.caption(f"⚠️ {index['missing_count']}개 아이템은 이미지를 준비하지 못해 검색에서 제외되었습니다.")

tab_text, tab_image = st.tabs(["📝 텍스트로 검색", "🖼️ 이미지로 검색"])

with tab_text:
    query = st.text_input("원하는 스타일을 설명해보세요", placeholder="예: 우아하고 세련된 검은색 옷이 필요해")
    n_results = st.slider("추천 개수", 1, 12, 5, key="text_n")
    if st.button("검색", type="primary", key="text_search", disabled=not query.strip()):
        results = cs.search_fashion_style(query, index["text_vectors"], n_results=n_results)
        cols = st.columns(min(len(results), 5) or 1)
        for i, (item, sim) in enumerate(results):
            with cols[i % len(cols)]:
                img_path = index["valid_images"].get(item)
                if img_path:
                    st.image(img_path, width="stretch")
                st.markdown(f"**{item}**")
                st.progress(min(max(sim, 0.0), 1.0), text=f"유사도 {sim:.1%}")

with tab_image:
    uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    n_results_img = st.slider("추천 개수", 1, 12, 5, key="image_n")
    if uploaded is not None:
        query_image = Image.open(uploaded)
        col_in, col_out = st.columns([1, 3])
        with col_in:
            st.image(query_image, caption="업로드한 이미지", width="stretch")
        with col_out:
            results = cs.search_by_image(
                query_image, index["image_vectors"], index["valid_images"], n_results=n_results_img
            )
            cols = st.columns(min(len(results), 5) or 1)
            for i, (item, sim, img_path) in enumerate(results):
                with cols[i % len(cols)]:
                    st.image(img_path, width="stretch")
                    st.markdown(f"**{item}**")
                    st.progress(min(max(sim, 0.0), 1.0), text=f"유사도 {sim:.1%}")

with st.expander("등록된 12가지 패션 프로필"):
    for name, desc in cs.FASHION_PROFILES.items():
        st.markdown(f"- **{name}**: {desc}")

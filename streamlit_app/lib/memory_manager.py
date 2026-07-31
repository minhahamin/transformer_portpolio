"""무료 호스팅 RAM 제약 대응: 무거운 페이지(CLIP/BLIP+번역) 간 전환 시
현재 페이지가 아닌 쪽의 모델 캐시를 비워 동시 메모리 사용량을 줄입니다.

st.cache_resource는 프로세스 전체에서 공유되는 캐시라, 세션별로 다른 페이지를
보고 있는 사용자가 여러 명이면 서로의 모델을 밀어낼 수 있습니다 — 무료 티어의
RAM 한도를 넘기지 않기 위한 트레이드오프입니다.
"""
import streamlit as st

from lib import captioning as cap
from lib import clip_search as cs
from lib import nllb_translate as nllb

_HEAVY_GROUPS = {
    "clip": [cs.load_clip, nllb.load_translator],
    "caption": [cap.load_blip, nllb.load_translator],
}


def activate(name: str) -> None:
    if st.session_state.get("_active_heavy_page") == name:
        return
    for other_name, fns in _HEAVY_GROUPS.items():
        if other_name != name:
            for fn in fns:
                fn.clear()
    st.session_state["_active_heavy_page"] = name

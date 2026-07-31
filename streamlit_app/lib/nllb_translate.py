"""공용 NLLB(CTranslate2 int8) 번역 유틸.

실습9 이미지 캡셔닝(영어→한국어)과 실습6 텍스트 검색(한국어→영어)이 같은
NLLB-200-distilled-600M 가중치를 방향만 바꿔 재사용합니다. 각자 따로
@st.cache_resource로 불러오면 캐시가 두 벌 잡혀 메모리에 같은 모델이 중복
로드되므로, 로더를 이 모듈 하나로 모았습니다.

원본 fp32 가중치를 PyTorch로 그대로 쓰면 generate() 한 번에 메모리가 3GB
가까이 치솟는 것이 실측으로 확인되어, 같은 가중치를 CTranslate2로 int8
변환해 사용합니다(로드+추론 합쳐 약 1.2~1.3GB). 변환 자체는 원본 fp32를
불러와야 해서 순간 4GB까지 치솟기 때문에 무료 호스팅에서 즉석 변환은
불가능 — 리소스가 넉넉한 로컬에서 미리 변환해 별도 HF 모델 저장소
(CT2_MODEL_REPO)에 올려두고, 앱은 그 완성된 변환본만 다운로드합니다.
"""
from pathlib import Path

import streamlit as st

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
CT2_MODEL_REPO = "hong28297/nllb-200-distilled-600m-ct2-int8"
CT2_CACHE_DIR = Path(__file__).resolve().parent / "nllb_ct2_int8"

LANG_CODES = {"ko": "kor_Hang", "en": "eng_Latn"}


@st.cache_resource(show_spinner="번역 모델 로드 중... (최초 1회, 약 600MB 다운로드)")
def load_translator():
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)

    if not (CT2_CACHE_DIR / "model.bin").exists():
        snapshot_download(repo_id=CT2_MODEL_REPO, local_dir=str(CT2_CACHE_DIR))

    translator = ctranslate2.Translator(str(CT2_CACHE_DIR), device="cpu")
    return translator, tokenizer


def translate(text: str, src: str, tgt: str, max_decoding_length: int = 50) -> str:
    translator, tokenizer = load_translator()
    tokenizer.src_lang = LANG_CODES[src]
    src_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch(
        [src_tokens], target_prefix=[[LANG_CODES[tgt]]], max_decoding_length=max_decoding_length
    )
    tgt_tokens = results[0].hypotheses[0][1:]
    tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
    return tokenizer.decode(tgt_ids, skip_special_tokens=True)

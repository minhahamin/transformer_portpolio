"""실습9 — BLIP 이미지 캡셔닝 + (선택) 한국어 번역.

번역 모델은 원래 무료 호스팅 메모리를 아끼려고 Helsinki-NLP/opus-mt-tc-big-en-ko(약 800MB)로
교체했었으나, 실제로 테스트해보니 이 체크포인트는 한국어 대신 중국어 단어/문장부호가 섞인
깨진 출력을 냈습니다(모델 카드의 공식 예제 코드로도 재현됨 — 우리 코드 문제가 아니라 체크포인트
자체 결함). 그래서 원본 노트북과 동일한 facebook/nllb-200-distilled-600M(2.4GB)으로 되돌렸습니다.

BLIP(990MB) + NLLB(2.4GB)를 동시에 메모리에 올리면 무료 호스팅에서 실제로 OOM으로 앱이
죽는 것이 확인되어, 번역을 요청하면 캡션 생성에 쓴 BLIP을 메모리에서 비운 뒤 번역 모델을
로드합니다(translate_to_korean 내부). 그 다음에 다시 캡셔닝하면 BLIP을 재로드합니다(디스크에는
이미 받아둔 가중치라 다운로드 없이 수 초 내로 다시 로드됨) — 매번 번역할 때마다 BLIP↔NLLB를
번갈아 로드하는 트레이드오프로 메모리 한도를 지킵니다.
"""
from PIL import Image

import streamlit as st
import torch

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
TRANSLATE_MODEL_ID = "facebook/nllb-200-distilled-600M"


@st.cache_resource(show_spinner="BLIP 캡셔닝 모델 로드 중... (최초 1회, 약 990MB 다운로드)")
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.eval()
    return model, processor


@st.cache_resource(show_spinner="번역 모델 로드 중... (최초 1회, 약 2.4GB 다운로드)")
def load_translator():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL_ID)
    model.eval()
    return model, tokenizer


def generate_caption(image: Image.Image, max_new_tokens: int = 50) -> str:
    model, processor = load_blip()
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)


def translate_to_korean(text: str) -> str:
    # BLIP(990MB)과 NLLB(2.4GB)가 동시에 메모리에 있으면 무료 호스팅에서 OOM이 나므로,
    # 번역 모델을 로드하기 전에 이미 사용이 끝난 BLIP을 먼저 비웁니다.
    load_blip.clear()
    model, tokenizer = load_translator()
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"),
            max_new_tokens=50,
        )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

"""실습9 — BLIP 이미지 캡셔닝 + (선택) 한국어 번역.

배포 환경의 메모리 제약 때문에 원본 노트북과 다른 점:
- 캡션 모델: Salesforce/blip-image-captioning-base (원본과 동일, 원래도 가벼움)
- 번역 모델: Helsinki-NLP/opus-mt-tc-big-en-ko (약 800MB, 원본 NLLB-600M 대비 약 1.6GB 절약).
  README([lab9_image_captioning.ipynb] 관련 메모)에 기록된 대로 이 체크포인트는 번역 품질이
  들쭉날쭉할 수 있음 — 무료 호스팅 메모리 제약과 맞바꾼 트레이드오프.
- 번역 모델은 사용자가 "한국어로 번역"을 켰을 때만 지연 로드됩니다.
"""
from PIL import Image

import streamlit as st
import torch

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
TRANSLATE_MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-en-ko"


@st.cache_resource(show_spinner="BLIP 캡셔닝 모델 로드 중... (최초 1회, 약 990MB 다운로드)")
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.eval()
    return model, processor


@st.cache_resource(show_spinner="번역 모델 로드 중... (최초 1회, 약 800MB 다운로드)")
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
    model, tokenizer = load_translator()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

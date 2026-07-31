"""실습9 — BLIP 이미지 캡셔닝 + (선택) 한국어 번역.

번역 모델은 원래 무료 호스팅 메모리를 아끼려고 Helsinki-NLP/opus-mt-tc-big-en-ko(약 800MB)로
교체했었으나, 실제로 테스트해보니 이 체크포인트는 한국어 대신 중국어 단어/문장부호가 섞인
깨진 출력을 냈습니다(모델 카드의 공식 예제 코드로도 재현됨 — 우리 코드 문제가 아니라 체크포인트
자체 결함). 그래서 원본 노트북과 동일한 facebook/nllb-200-distilled-600M으로 되돌렸습니다.

BLIP(990MB) + NLLB를 동시에 메모리에 올리면 무료 호스팅에서 실제로 OOM으로 앱이 죽는 것이
확인되어, 번역을 요청하면 캡션 생성에 쓴 BLIP을 메모리에서 비운 뒤 번역 모델을 로드합니다
(translate_to_korean 내부).

그런데도 배포판에서 여전히 죽는 게 재확인되어 실측해보니, transformers+PyTorch로 NLLB를
그냥 불러오면 로드 시점엔 1.6GB지만 실제로 generate() 한 번 호출하는 순간 메모리가 3GB
가까이 치솟습니다(짧은 문장 3개만 번역해도 동일 — 빔서치를 꺼도 동일하여 원인은 배치/빔
개수가 아니라 PyTorch CPU 추론 자체의 활성화 버퍼로 확인됨). 그래서 같은 NLLB-600M
가중치를 CTranslate2로 int8 변환해서 쓰도록 바꿨습니다 — 다른 모델로 바꾼 게 아니라
같은 가중치를 더 효율적인 추론 엔진에 올린 것이라 번역 품질은 거의 그대로면서
(로드 1.2GB, generate 후에도 1.2GB로 스파이크가 사실상 사라짐) 메모리 사용량만 크게
줄었습니다.

번역 모델 자체(NLLB, CTranslate2 int8 변환)는 실습6 텍스트 검색과 함께 쓰는
공용 모듈 `lib/nllb_translate.py`로 옮겼습니다 — 두 실습이 같은 모델을 방향만
바꿔(영→한 / 한→영) 재사용하므로, 따로 로드하면 메모리에 같은 모델이 두 벌
올라가는 걸 막기 위함입니다.
"""
from PIL import Image

import streamlit as st
import torch

from lib import nllb_translate

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"


@st.cache_resource(show_spinner="BLIP 캡셔닝 모델 로드 중... (최초 1회, 약 990MB 다운로드)")
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.eval()
    return model, processor


def generate_caption(image: Image.Image, max_new_tokens: int = 50) -> str:
    model, processor = load_blip()
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)


def translate_to_korean(text: str) -> str:
    # BLIP(990MB)과 번역 모델이 동시에 메모리에 있으면 무료 호스팅에서 OOM이 나므로,
    # 번역 모델을 로드하기 전에 이미 사용이 끝난 BLIP을 먼저 비웁니다.
    load_blip.clear()
    return nllb_translate.translate(text, src="en", tgt="ko")

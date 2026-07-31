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

다만 변환 자체(원본 fp32 가중치를 불러와 양자화하는 과정)는 순간 메모리가 4GB까지
치솟는 것으로 확인되어, 무료 호스팅에서 최초 요청 시점에 직접 변환하면 그때 죽습니다.
그래서 변환은 리소스가 넉넉한 로컬에서 미리 한 번만 수행했고, 결과물(약 600MB)은
GitHub 100MB 파일 제한을 넘기 때문에 별도 HF 모델 저장소(하단 CT2_MODEL_REPO)에
올려두고 앱은 그 완성된 변환본을 다운로드만 합니다.
"""
from pathlib import Path

from PIL import Image

import streamlit as st
import torch

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
TRANSLATE_MODEL_ID = "facebook/nllb-200-distilled-600M"
CT2_MODEL_REPO = "hong28297/nllb-200-distilled-600m-ct2-int8"
CT2_CACHE_DIR = Path(__file__).resolve().parent / "nllb_ct2_int8"


@st.cache_resource(show_spinner="BLIP 캡셔닝 모델 로드 중... (최초 1회, 약 990MB 다운로드)")
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.eval()
    return model, processor


@st.cache_resource(show_spinner="번역 모델 로드 중... (최초 1회, 약 600MB 다운로드)")
def load_translator():
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_ID)

    if not (CT2_CACHE_DIR / "model.bin").exists():
        snapshot_download(repo_id=CT2_MODEL_REPO, local_dir=str(CT2_CACHE_DIR))

    translator = ctranslate2.Translator(str(CT2_CACHE_DIR), device="cpu")
    return translator, tokenizer


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
    translator, tokenizer = load_translator()
    tokenizer.src_lang = "eng_Latn"
    src_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch(
        [src_tokens], target_prefix=[["kor_Hang"]], max_decoding_length=50
    )
    tgt_tokens = results[0].hypotheses[0][1:]
    tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
    return tokenizer.decode(tgt_ids, skip_special_tokens=True)

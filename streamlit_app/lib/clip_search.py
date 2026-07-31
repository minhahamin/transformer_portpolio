"""실습6 — CLIP 멀티모달 패션 스타일 검색 (노트북 로직 이식).

배포 환경 신뢰성을 위해 `git+https://github.com/openai/CLIP.git` 대신
transformers의 CLIPModel/CLIPProcessor(openai/clip-vit-base-patch16, 동일 가중치)를 사용합니다.
"""
from pathlib import Path

import requests
import streamlit as st
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "lab6_clip_multimodal_fashion_search" / "fashion_images"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "openai/clip-vit-base-patch16"
UNSPLASH_API_URL = "https://api.unsplash.com"

FASHION_PROFILES = {
    "클래식 셔츠": "깔끔한 흰색이나 검은색 셔츠, 어떤 상황에도 잘 어울리는 기본 아이템",
    "캐주얼 티셔츠": "편한 코튼 티셔츠, 루즈한 핏, 캐주얼한 일상복",
    "우아한 드레스": "세련되고 검은색인 드레스, 미니멀한 디자인, 저녁 약속에 어울림",
    "오버사이즈 아우터": "넉넉한 핏의 현대적인 재킷이나 코트, 캐주얼한 분위기",
    "슬림 핏 바지": "몸에 딱 붙는 심플한 바지, 포멀한 상황에 적합",
    "와이드 팬츠": "편안한 넓은 바지, 트렌디하면서도 세련된 실루엣",
    "미니 스커트": "짧은 길이의 스커트, 젊고 활기찬 분위기, 여름 스타일",
    "롱 드레스": "발목까지 내려오는 긴 드레스, 우아하고 특별한 자리에 적합",
    "화이트 스니커즈": "깨끗한 흰색 운동화, 캐주얼하고 어떤 옷과도 잘 어울림",
    "부츠": "무릎까지 올라오는 부츠, 겨울 스타일, 세련된 분위기",
    "미니 크로스백": "작고 단정한 가방, 미니멀한 럭셔리 스타일",
    "빈티지 선글라스": "복고풍의 트렌디한 선글라스, 여름 필수 액세서리",
}

IMAGE_QUERIES = {
    "클래식 셔츠": "classic white shirt fashion woman",
    "캐주얼 티셔츠": "casual t-shirt woman style",
    "우아한 드레스": "elegant black dress woman",
    "오버사이즈 아우터": "oversized jacket coat fashion",
    "슬림 핏 바지": "slim fit pants fashion woman",
    "와이드 팬츠": "wide pants fashion woman",
    "미니 스커트": "mini skirt fashion style",
    "롱 드레스": "long dress elegant woman",
    "화이트 스니커즈": "white sneakers fashion",
    "부츠": "boots fashion woman",
    "미니 크로스백": "mini crossbody bag fashion",
    "빈티지 선글라스": "vintage sunglasses fashion",
}


@st.cache_resource(show_spinner="CLIP 모델 로드 중... (최초 1회, 약 350MB 다운로드)")
def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def _as_embedding(output):
    """transformers 버전에 따라 get_image/text_features가 tensor 또는
    pooler_output에 임베딩이 담긴 ModelOutput을 반환하므로 둘 다 처리합니다."""
    return output if torch.is_tensor(output) else output.pooler_output


def get_unsplash_key() -> str | None:
    """secrets.toml이 아예 없으면 st.secrets.get()도 예외를 던지므로 방어적으로 처리합니다."""
    try:
        return st.secrets.get("UNSPLASH_API_KEY")
    except Exception:
        return None


def download_fashion_image(query: str, cache_name: str) -> str | None:
    cache_path = CACHE_DIR / f"{cache_name}.jpg"
    if cache_path.exists():
        return str(cache_path)

    api_key = get_unsplash_key()
    if not api_key:
        return None

    params = {"query": query, "per_page": 1, "orientation": "portrait", "client_id": api_key}
    try:
        response = requests.get(f"{UNSPLASH_API_URL}/search/photos", params=params, timeout=10)
        response.raise_for_status()
        results = response.json()["results"]
        if results:
            image_url = results[0]["urls"]["regular"]
            img_response = requests.get(image_url, timeout=10)
            with open(cache_path, "wb") as f:
                f.write(img_response.content)
            return str(cache_path)
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner="12가지 패션 아이템 이미지 준비 및 벡터화 중...")
def build_fashion_index():
    model, processor = load_clip()

    fashion_images = {}
    for item_name, query in IMAGE_QUERIES.items():
        cache_name = item_name.replace(" ", "_")
        fashion_images[item_name] = download_fashion_image(query, cache_name)

    image_vectors, valid_images = {}, {}
    with torch.no_grad():
        for item_name, image_path in fashion_images.items():
            if image_path is None:
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                feats = _as_embedding(model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                image_vectors[item_name] = feats.cpu().numpy().squeeze()
                valid_images[item_name] = image_path
            except Exception:
                continue

    text_vectors = {}
    with torch.no_grad():
        for item_name, desc in FASHION_PROFILES.items():
            inputs = processor(text=[desc], return_tensors="pt", padding=True)
            feats = _as_embedding(model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_vectors[item_name] = feats.cpu().numpy().squeeze()

    return {
        "image_vectors": image_vectors,
        "valid_images": valid_images,
        "text_vectors": text_vectors,
        "missing_count": len(FASHION_PROFILES) - len(image_vectors),
    }


def search_fashion_style(style_description: str, text_vectors: dict, n_results: int = 5):
    model, processor = load_clip()
    with torch.no_grad():
        inputs = processor(text=[style_description], return_tensors="pt", padding=True)
        feats = _as_embedding(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        query_vector = feats.cpu().numpy().squeeze()

    similarities = [(item, float(query_vector @ vec)) for item, vec in text_vectors.items()]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n_results]


def search_by_image(pil_image: Image.Image, image_vectors: dict, valid_images: dict, n_results: int = 5):
    model, processor = load_clip()
    with torch.no_grad():
        inputs = processor(images=pil_image.convert("RGB"), return_tensors="pt")
        feats = _as_embedding(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        query_vector = feats.cpu().numpy().squeeze()

    similarities = [
        (item, float(query_vector @ vec), valid_images[item]) for item, vec in image_vectors.items()
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n_results]


def embed_uploaded_images(images: list) -> list:
    """사용자가 업로드한 임의의 이미지 목록을 CLIP 임베딩으로 변환합니다 ("내 옷장" 기능용)."""
    model, processor = load_clip()
    vectors = []
    with torch.no_grad():
        for image in images:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            feats = _as_embedding(model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.cpu().numpy().squeeze())
    return vectors


def search_in_wardrobe(style_description: str, vectors: list, n_results: int = 5):
    """텍스트 설명으로 embed_uploaded_images() 결과(내 옷장) 중에서 검색합니다.

    Returns: [(인덱스, 유사도), ...] 유사도 내림차순
    """
    model, processor = load_clip()
    with torch.no_grad():
        inputs = processor(text=[style_description], return_tensors="pt", padding=True)
        feats = _as_embedding(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        query_vector = feats.cpu().numpy().squeeze()

    similarities = [(i, float(query_vector @ vec)) for i, vec in enumerate(vectors)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n_results]

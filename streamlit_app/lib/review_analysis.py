"""실습1 — 한국어 영화 리뷰 분석 로직 (노트북 코드를 그대로 이식)."""
import re
import warnings
from collections import Counter
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "lab1_korean_movie_review_analysis" / "data" / "input.txt"

BASIC_STOPWORDS = {
    "영화", "정말", "진짜", "평점", "너무",
    "하다", "되다", "있다", "없다",
    "좋다", "나쁘다", "크다", "작다",
}

IMPROVED_STOPWORDS = {
    "이", "의", "가", "를", "을", "인", "한", "는", "고", "며", "로",
    "와", "으로", "같이", "에서", "마다", "부터", "까지", "만", "면", "듯",
    "적", "성", "함", "음", "하", "어", "았",
    "영화", "정말", "진짜", "평점", "너무", "아주", "매우", "그냥",
    "것", "분", "점", "번", "중",
    "하", "되", "있", "없", "않",
    "있다", "없다", "되다", "하다",
    "같다", "다르다", "좋다", "나쁘다",
    "매우", "정말", "아주", "그냥", "거의", "완전", "뭔가",
}


def load_corpus(file_path: Path) -> list[str]:
    if not file_path.exists():
        warnings.warn(f"입력 파일을 찾을 수 없습니다: {file_path}. 샘플 문장으로 대체합니다.")
        return [
            "이 영화 정말 감동적이었어요",
            "연기가 최고였어",
            "너무 재미있었다",
            "시간이 빨리 갔어",
            "정말 추천합니다",
        ]

    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = re.sub(r"[^가-힣\s]", " ", line).strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


@st.cache_resource(show_spinner="Okt 형태소 분석기 준비 중... (최초 1회, JVM 기동)")
def get_okt():
    from konlpy.tag import Okt
    return Okt()


@st.cache_data(show_spinner=False)
def load_corpus_cached() -> list[str]:
    return load_corpus(CORPUS_PATH)


def tokenize_and_count(
    sentences: list[str],
    okt,
    stopwords: set,
    allowed_pos: set = frozenset({"Noun", "Verb", "Adjective"}),
    min_length: int = 2,
    min_freq: int = 1,
) -> Counter:
    tokens = []
    for sent in sentences:
        for morpheme, pos in okt.pos(sent):
            if pos not in allowed_pos:
                continue
            if morpheme in stopwords:
                continue
            if len(morpheme) < min_length:
                continue
            tokens.append(morpheme)
    return Counter(tokens)


@st.cache_data(show_spinner="형태소 분석 및 빈도 계산 중...")
def analyze(mode: str, min_freq: int = 1) -> Counter:
    """mode: 'basic' 또는 'improved'"""
    corpus = load_corpus_cached()
    okt = get_okt()
    stopwords = BASIC_STOPWORDS if mode == "basic" else IMPROVED_STOPWORDS
    counter = tokenize_and_count(corpus, okt, stopwords)
    if mode == "improved" and min_freq > 1:
        counter = Counter({w: f for w, f in counter.items() if f >= min_freq})
    return counter


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Streamlit Cloud (packages.txt: fonts-nanum)
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",  # 로컬 Windows
    "/System/Library/Fonts/AppleGothic.ttf",  # 로컬 macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


@st.cache_resource(show_spinner=False)
def get_korean_font_path() -> str | None:
    """한글 폰트 파일 경로를 찾아 matplotlib에 등록하고 경로를 반환합니다 (WordCloud에도 재사용).

    koreanize_matplotlib(0.1.1)는 Python 3.12+에서 제거된 distutils에 의존해 배포 환경에서
    깨질 수 있어, 폰트 파일을 직접 찾아 등록하는 방식으로 대체했습니다.
    """
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            fm.fontManager.addfont(candidate)
            family = fm.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return candidate

    warnings.warn("한글 폰트를 찾지 못했습니다. 그래프/워드클라우드에서 한글이 깨져 보일 수 있습니다.")
    return None

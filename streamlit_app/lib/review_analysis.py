"""실습1 — 한국어 영화 리뷰 분석 로직 (노트북 코드를 그대로 이식)."""
import ipaddress
import re
import socket
import warnings
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

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


MAX_FETCH_BYTES = 5 * 1024 * 1024  # 5MB — 과도하게 큰 응답으로부터 서버 보호


def _is_safe_url(url: str) -> tuple[bool, str]:
    """SSRF 방지: http(s) 스킴만 허용하고, 사설/루프백/링크로컬 IP로 해석되는 호스트는 차단합니다.

    공개 배포 앱이 사용자가 입력한 임의의 URL을 서버에서 대신 요청(fetch)하므로,
    내부망·클라우드 메타데이터 엔드포인트(예: 169.254.169.254)로 향하는 요청을 막아야 합니다.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "URL 형식이 올바르지 않습니다."

    if parsed.scheme not in ("http", "https"):
        return False, "http:// 또는 https:// 로 시작하는 URL만 지원합니다."
    if not parsed.hostname:
        return False, "URL에 호스트가 없습니다."

    try:
        resolved_ips = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False, f"호스트를 찾을 수 없습니다: {parsed.hostname}"

    for family, _, _, _, sockaddr in resolved_ips:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False, "내부/사설 네트워크 주소로는 요청할 수 없습니다."

    return True, ""


def fetch_and_extract_ko_text(url: str) -> list[str]:
    """URL을 가져와 본문 텍스트에서 한국어 문장만 추출합니다 (load_corpus와 동일한 후처리).

    특정 사이트(예: 네이버 영화) 마크업에 맞춘 전용 스크래퍼가 아니라, 페이지의 텍스트를
    범용으로 뽑아내는 방식이라 사이트 구조에 따라 품질 차이가 클 수 있습니다.
    """
    safe, reason = _is_safe_url(url)
    if not safe:
        raise ValueError(reason)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ReviewAnalysisBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=10, stream=True)
    response.raise_for_status()

    content = response.raw.read(MAX_FETCH_BYTES + 1, decode_content=True)
    if len(content) > MAX_FETCH_BYTES:
        raise ValueError("페이지 용량이 너무 큽니다 (5MB 제한).")

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    sentences = []
    for line in soup.get_text("\n").splitlines():
        cleaned = re.sub(r"[^가-힣\s]", " ", line).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) >= 5:  # 메뉴/버튼 등 짧은 노이즈 제거
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
def analyze(mode: str, min_freq: int = 1, corpus: list[str] | None = None) -> Counter:
    """mode: 'basic' 또는 'improved'. corpus를 지정하면 기본 제공 코퍼스 대신 사용합니다."""
    if corpus is None:
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

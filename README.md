# Transformer 실습 포트폴리오

AI Human 강의 · Part II. Transformer — 1차 실습 과제(실습1, 2, 6, 9) 제출용 저장소입니다.

전체 9개 실습 중 서로 다른 4가지 AI 응용 영역(텍스트 감성분석 · 번역 · 멀티모달 검색 · 이미지 캡셔닝)을 하나씩 골라 진행합니다. 네 실습 모두 모델을 처음부터 학습시키지 않고, 이미 학습된 사전학습 모델(형태소 분석기, Seq2Seq, CLIP, BLIP)을 가져와 직접 파이프라인을 조립하는 데 집중합니다.

## 실습 목록

| # | 실습명 | 응용 영역 | 핵심 기술 | 폴더 |
|---|--------|-----------|-----------|------|
| 1 | 한국어 영화 리뷰 분석 및 시각화 | 텍스트 전처리 · 시각화 | 정규표현식, Okt 형태소 분석, BarPlot/WordCloud | [`lab1_korean_movie_review_analysis/`](lab1_korean_movie_review_analysis/) |
| 2 | Seq2Seq 기반 번역 AI 모델링 | 인코더-디코더 번역 | BiGRU Encoder, Luong Attention, Teacher Forcing, Beam Search | [`lab2_seq2seq_translation_modeling/`](lab2_seq2seq_translation_modeling/) |
| 6 | CLIP 멀티모달 패션 스타일 검색 | 텍스트-이미지 교차 검색 | CLIP(ViT-B/16), 코사인 유사도 기반 검색 | [`lab6_clip_multimodal_fashion_search/`](lab6_clip_multimodal_fashion_search/) |
| 9 | Image Captioning | 이미지 → 설명 문장 생성 | BLIP(ViT Encoder + Transformer Decoder), MarianMT, BLEU 평가 | [`lab9_image_captioning/`](lab9_image_captioning/) |

## 실습별 소개

### 실습1 — 한국어 영화 리뷰 분석 및 시각화
영화 리뷰 텍스트를 정규표현식으로 정제하고, Okt 형태소 분석기로 의미 있는 단어(명사/동사/형용사)만 추출합니다. 불용어 필터를 기본형 → 개선형으로 강화해가며 결과 변화를 비교하고, 상위 빈도 단어를 BarPlot과 WordCloud로 시각화합니다.

### 실습2 — Seq2Seq 기반 번역 AI 모델링
한국어-영어 병렬 문장 720쌍으로 Bidirectional GRU Encoder + Luong Attention + GRU Decoder 구조의 번역 모델을 직접 구현합니다. 학습 시에는 Teacher Forcing, 추론 시에는 Greedy Decoding과 Beam Search를 각각 적용해 비교합니다.

### 실습6 — CLIP 멀티모달 패션 스타일 검색
OpenAI CLIP 모델로 패션 아이템 설명(텍스트)과 실제 이미지를 같은 512차원 벡터 공간에 임베딩합니다. 텍스트 쿼리(예: "우아하고 세련된 검은색 옷")와 이미지 쿼리 양쪽으로 코사인 유사도 기반 검색을 수행하고 결과를 시각화합니다.

### 실습9 — Image Captioning
BLIP(Salesforce/blip-image-captioning-base) 모델로 MSCOCO 이미지 50장에 대한 영어 캡션을 자동 생성하고, BLEU 점수로 정답 캡션과의 유사도를 평가합니다. 이어서 MarianMT로 영어 캡션을 한국어로 번역해 MSCOCO 한국어 정답과 비교합니다.

## 실행 방법

각 폴더의 `.ipynb` 파일을 Jupyter/VSCode에서 순서대로 실행하면 됩니다. 노트북 상단 셀에서 필요한 패키지를 `%pip install`로 자동 설치합니다.

- 실습1: `konlpy`(Java 런타임 필요), `koreanize_matplotlib`, `wordcloud`
- 실습2: PyTorch 순수 구현 (외부 NLP 라이브러리 불필요 — 아래 참고사항 참조)
- 실습6: `torch`, `git+https://github.com/openai/CLIP.git`
- 실습9: `transformers`, `torch`, `nltk`, `sentencepiece`, `sacremoses`

## 실행 참고사항 (환경 이슈 노트)

- **실습2**: 원래 `torchtext`를 사용하도록 작성되었으나, torchtext는 2023년 Meta가 공식적으로 개발을 중단해 최신 PyTorch와 호환되는 배포판이 없습니다. `get_tokenizer` / `build_vocab_from_iterator`를 순수 Python으로 재구현해 대체했습니다(외부 패키지 설치 불필요).
- **실습9**: `korean_image_captioning_dataset/`(MSCOCO 한국어 캡션, 117MB)과 `coco_images/`는 용량이 커서 `.gitignore`로 제외했습니다 — 노트북의 데이터 로드/다운로드 셀을 그대로 실행하면 다시 준비됩니다.
- **실습9**: 한국어 번역에 쓰는 `Helsinki-NLP/opus-mt-tc-big-en-ko` 체크포인트가 토크나이저 vocab 이슈로 번역 품질이 낮게 나오는 현상이 확인되어, 대체 모델 검증을 진행 중입니다.

## 프로젝트 구조

```
.
├── lab1_korean_movie_review_analysis/
│   ├── data/                  # 원본 리뷰 텍스트
│   ├── output/                # 시각화 결과, 상위 단어 목록
│   └── lab1_korean_movie_review_analysis.ipynb
├── lab2_seq2seq_translation_modeling/
│   ├── data/                  # 한/영 병렬 문장 쌍
│   └── lab2_seq2seq_translation_modeling.ipynb
├── lab6_clip_multimodal_fashion_search/
│   ├── fashion_images/        # (gitignore) CLIP 검색용 패션 이미지 캐시
│   └── lab6_clip_multimodal_fashion_search.ipynb
├── lab9_image_captioning/
│   ├── coco_images/                       # (gitignore) MSCOCO 샘플 이미지
│   ├── korean_image_captioning_dataset/   # (gitignore) MSCOCO 한국어 캡션(117MB)
│   └── lab9_image_captioning.ipynb
└── transformer4_assignment_guide.docx     # 과제 안내/제출 가이드
```

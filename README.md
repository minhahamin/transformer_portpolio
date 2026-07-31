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
- **실습2**: `data/input.txt`가 채점 시스템이 채워 넣기 전의 placeholder 텍스트("Test\nData\n...")를 담고 있어, 이 파일을 그대로 번역 테스트에 사용하면 `translate()`/`beam_search_translate()`가 학습 어휘에 없는 영어 단어(`Test` 등)에서 `KeyError`를 던집니다. 두 함수 모두 학습 어휘에 없는 토큰은 건너뛰도록 방어 코드를 추가했습니다 — 다만 `input.txt` 자체가 한국어 문장이 아니므로 번역 결과 자체는 여전히 의미가 없습니다. 실제 번역을 확인하려면 `input.txt`를 실제 한국어 문장으로 바꿔서 실행하세요.
- **실습9**: `korean_image_captioning_dataset/`(MSCOCO 한국어 캡션, 117MB)과 `coco_images/`는 용량이 커서 `.gitignore`로 제외했습니다 — 노트북의 데이터 로드/다운로드 셀을 그대로 실행하면 다시 준비됩니다.
- **실습9**: 한국어 번역에 쓰는 `Helsinki-NLP/opus-mt-tc-big-en-ko` 체크포인트가 토크나이저 vocab 이슈로 번역 품질이 낮게 나오는 현상이 확인되어, 노트북은 `facebook/nllb-200-distilled-600M`으로 교체했습니다. (실제로 이 체크포인트는 모델 카드의 공식 예제 코드로 재현해봐도 중국어 단어/문장부호가 섞인 깨진 출력을 냈습니다 — 배포판에서 메모리를 아끼려고 다시 시도했다가 확인된 내용이며, 아래 Streamlit 배포판도 결국 품질을 우선해 `nllb-200-distilled-600M`으로 되돌렸습니다.)

## Streamlit 배포판

`streamlit_app/`에 4개 실습을 하나로 묶은 멀티페이지 Streamlit 앱이 있습니다. 원본 `.ipynb`는 그대로 두고, 무료 호스팅(Streamlit Community Cloud, RAM 제약)에 맞춰 일부 모델만 배포판 코드에서 가볍게 교체했습니다.

| 실습 | 배포판에서 달라진 점 |
|---|---|
| 실습1 | 없음 (Okt 그대로, `packages.txt`로 JVM 설치) |
| 실습2 | 앱이 매번 즉석에서 3 epoch만 학습하면 loss가 높아 번역 품질이 나빴음(입력과 무관하게 "mike is ." 류로 출력이 수렴) → 로컬에서 60 epoch 미리 학습해 `streamlit_app/lib/seq2seq_checkpoint.pt`로 저장해두고, 앱은 이를 로드만 함(수십 ms). 체크포인트가 없으면 3 epoch 즉석 학습으로 자동 폴백 |
| 실습6 | `git+openai/CLIP` → `transformers.CLIPModel`(`openai/clip-vit-base-patch16`, 동일 가중치, 배포 신뢰성 목적) |
| 실습9 | 없음 (원본과 동일한 `nllb-200-distilled-600M`, 2.4GB). 메모리 절약을 위해 `opus-mt-tc-big-en-ko`(~800MB)로 교체를 시도했으나 번역 출력이 심각하게 깨져(중국어 혼입) 되돌림 — BLIP(990MB)과 합쳐 이 페이지의 메모리 사용량이 큼 |

무료 티어 RAM 한도(약 1GB)를 넘기지 않도록, CLIP(실습6) ↔ BLIP+번역모델(실습9) 페이지를 전환할 때 이전 쪽의 모델 캐시를 자동으로 비웁니다(`streamlit_app/lib/memory_manager.py`). 동시에 다른 페이지를 보는 사용자가 여러 명이면 서로의 캐시를 밀어낼 수 있다는 트레이드오프가 있습니다.

### 로컬 실행

```bash
pip install -r requirements.txt
# 실습1의 Okt는 Java(JRE/JDK)가 로컬에 설치되어 있어야 합니다.

# 실습2 번역 품질을 위해 최초 1회만 실행 (약 40분 소요, 이후에는 재실행 불필요)
python streamlit_app/scripts/pretrain_seq2seq.py

streamlit run streamlit_app/Home.py
```

실습6에서 Unsplash로 새 이미지를 받아오려면(이미 `fashion_images/`에 캐시가 있으면 불필요) `.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사하고 `UNSPLASH_API_KEY`를 채워 넣으세요.

### Streamlit Community Cloud 배포

1. 이 저장소를 GitHub에 push합니다.
2. [share.streamlit.io](https://share.streamlit.io)에서 New app → 이 저장소 선택 → Main file path에 `streamlit_app/Home.py` 지정.
3. 저장소 루트의 `requirements.txt`(파이썬 패키지)와 `packages.txt`(`default-jdk`, `fonts-nanum` — apt 패키지)가 자동으로 인식됩니다.
4. App 설정의 Secrets에 아래 내용을 추가합니다(선택, 실습6용):
   ```toml
   UNSPLASH_API_KEY = "your-unsplash-access-key"
   ```
5. 무거운 모델(CLIP ~350MB, BLIP ~990MB, 번역 모델 ~800MB)은 각 페이지를 처음 열 때만 다운로드됩니다. 무료 티어는 RAM이 넉넉하지 않으므로, 여러 페이지(특히 실습9)를 동시에/연속으로 열면 메모리 초과로 앱이 재시작될 수 있습니다 — 재현되면 Hugging Face Spaces 등 RAM이 더 큰 플랫폼을 권장합니다.

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

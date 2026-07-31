"""실습2 — Seq2Seq(BiGRU + Luong Attention) 번역 모델 (노트북 코드를 그대로 이식).

torchtext가 더 이상 유지되지 않아, 노트북과 동일하게 Vocab/tokenizer를 순수 파이썬으로 재구현합니다.

노트북 원본은 한국어를 공백 기준으로만 나누는데(`text.split()`), 학습 데이터가 720문장뿐이라
어휘가 아주 작고 "안녕하세요"처럼 학습 문장과 조사/띄어쓰기가 조금만 달라도 완전히 다른(미등록)
단어로 취급되어 번역이 크게 어긋납니다. 배포판에서는 lab1과 동일한 Okt 형태소 분석기로 한국어를
더 작은 단위(형태소)로 나눠, 학습 문장과 표현이 달라도 겹치는 형태소가 남을 확률을 높입니다.
"""
import random
import re
from collections import Counter
from pathlib import Path

import streamlit as st
import torch
from torch import nn

from lib import review_analysis as ra

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "lab2_seq2seq_translation_modeling" / "data"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "seq2seq_checkpoint.pt"

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<bos>", "<eos>"

_EN_PATTERNS = [
    (re.compile(r"\'"), " '  "),
    (re.compile(r"\""), ""),
    (re.compile(r"\."), " . "),
    (re.compile(r"<br \/>"), " "),
    (re.compile(r","), " , "),
    (re.compile(r"\("), " ( "),
    (re.compile(r"\)"), " ) "),
    (re.compile(r"\!"), " ! "),
    (re.compile(r"\?"), " ? "),
    (re.compile(r"\;"), " "),
    (re.compile(r"\:"), " "),
    (re.compile(r"\s+"), " "),
]


def tok_en(line: str) -> list:
    line = line.lower()
    for pattern, repl in _EN_PATTERNS:
        line = pattern.sub(repl, line)
    return line.split()


def tok_ko(text: str) -> list:
    return ra.get_okt().morphs(text.strip())


class Vocab:
    def __init__(self, itos: list):
        self._itos = itos
        self._stoi = {tok: i for i, tok in enumerate(itos)}

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, token: str) -> int:
        return self._stoi[token]

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def lookup_token(self, index: int) -> str:
        return self._itos[index]

    def get_itos(self) -> list:
        return list(self._itos)


def build_vocab(token_lists: list[list[str]], specials: list) -> Vocab:
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    for token in specials:
        counter.pop(token, None)
    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    itos = list(specials) + [token for token, _ in ranked]
    return Vocab(itos)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, emb_dim=64, hid_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        emb = self.embed(src)
        out, h = self.gru(emb)
        hidden = torch.cat([h[-2], h[-1]], dim=1)
        hidden = torch.tanh(self.fc(hidden))
        hidden = hidden.unsqueeze(0)
        return out, hidden


class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_out, mask):
        hidden = hidden.squeeze(0)
        seq_len = enc_out.size(1)
        hidden_rep = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat([hidden_rep, enc_out], dim=2)
        energy = torch.tanh(self.W(combined))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(~mask, -1e10)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, emb_dim=64, hid_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim + hid_dim * 2, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim * 3, vocab_size)
        self.attention = LuongAttention(hid_dim)

    def forward(self, input_tok, hidden, enc_out, mask):
        emb = self.embed(input_tok).unsqueeze(1)
        context, attn = self.attention(hidden, enc_out, mask)
        context = context.unsqueeze(1)
        gru_input = torch.cat([emb, context], dim=2)
        gru_out, new_hidden = self.gru(gru_input, hidden)
        combined = torch.cat([gru_out.squeeze(1), context.squeeze(1)], dim=1)
        output = self.out(combined)
        return output, new_hidden, attn


class TranslationBundle:
    def __init__(self, encoder, decoder, vocab_ko, vocab_en, pad, bos, eos, device, losses):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_ko = vocab_ko
        self.vocab_en = vocab_en
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.device = device
        self.losses = losses

    def translate(self, sentence: str, max_len: int = 20) -> str:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            tokens = tok_ko(sentence)
            unk_safe = [t for t in tokens if t in self.vocab_ko]
            src = [self.bos] + [self.vocab_ko[t] for t in unk_safe] + [self.eos]
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(self.device)

            enc_out, hidden = self.encoder(src)
            mask = src != self.pad

            input_tok = torch.tensor([self.bos]).to(self.device)
            result = []
            for _ in range(max_len):
                output, hidden, _ = self.decoder(input_tok, hidden, enc_out, mask)
                top_token = output.argmax(dim=1).item()
                if top_token == self.eos:
                    break
                result.append(self.vocab_en.lookup_token(top_token))
                input_tok = torch.tensor([top_token]).to(self.device)
            return " ".join(result)

    def beam_search_translate(self, sentence: str, beam_width: int = 3, max_len: int = 20) -> str:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            tokens = tok_ko(sentence)
            unk_safe = [t for t in tokens if t in self.vocab_ko]
            src = [self.bos] + [self.vocab_ko[t] for t in unk_safe] + [self.eos]
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(self.device)

            enc_out, hidden = self.encoder(src)
            mask = src != self.pad

            beams = [(0.0, [self.bos], hidden, enc_out, mask)]
            completed = []

            for _ in range(max_len):
                next_beams = []
                for score, toks, h, enc, m in beams:
                    if toks[-1] == self.eos:
                        completed.append((score, toks))
                        continue
                    input_tok = torch.tensor([toks[-1]]).to(self.device)
                    output, h_new, _ = self.decoder(input_tok, h, enc, m)
                    log_probs = torch.log_softmax(output, dim=1)[0]
                    if len(toks) > 1 and toks[-1] == toks[-2]:
                        log_probs[toks[-1]] -= 0.5
                    topk_probs, topk_indices = log_probs.topk(beam_width)
                    for prob, idx in zip(topk_probs, topk_indices):
                        next_beams.append((score + prob.item(), toks + [idx.item()], h_new, enc, m))

                next_beams.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
                beams = next_beams[:beam_width]
                if all(b[1][-1] == self.eos for b in beams):
                    break

            for score, toks, _, _, _ in beams:
                if toks[-1] != self.eos:
                    completed.append((score, toks))

            best_tokens = max(completed, key=lambda x: x[0])[1] if completed else beams[0][1]
            result = []
            for tok in best_tokens[1:]:
                if tok == self.eos:
                    break
                result.append(self.vocab_en.lookup_token(tok))
            return " ".join(result)


def _train_bundle_from_scratch() -> TranslationBundle:
    """체크포인트가 없을 때만 쓰는 폴백: 즉석에서 3 epoch만 학습합니다(로컬 개발용)."""
    device = torch.device("cpu")
    torch.manual_seed(42)
    random.seed(42)

    with open(DATA_DIR / "train_kor.txt", encoding="utf-8") as f_ko, \
         open(DATA_DIR / "train_eng.txt", encoding="utf-8") as f_en:
        kor_lines = f_ko.read().strip().splitlines()
        eng_lines = f_en.read().strip().splitlines()
    pairs = list(zip(kor_lines, eng_lines))

    specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    ko_tokens = [specials] + [tok_ko(ko) for ko, _ in pairs]
    en_tokens = [specials] + [tok_en(en) for _, en in pairs]
    vocab_ko = build_vocab(ko_tokens, specials)
    vocab_en = build_vocab(en_tokens, specials)

    pad, bos, eos = vocab_ko[PAD_TOKEN], vocab_ko[BOS_TOKEN], vocab_ko[EOS_TOKEN]

    def tensorize(pair):
        ko, en = pair
        src = [bos] + [vocab_ko[t] for t in tok_ko(ko)] + [eos]
        tgt = [bos] + [vocab_en[t] for t in tok_en(en)] + [eos]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

    data = [tensorize(p) for p in pairs]

    encoder = Encoder(len(vocab_ko), pad).to(device)
    decoder = Decoder(len(vocab_en), pad).to(device)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad)

    losses = []
    for _epoch in range(3):
        encoder.train()
        decoder.train()
        total_loss = 0
        for src, tgt in data:
            src = src.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)
            enc_out, hidden = encoder(src)
            mask = src != pad
            input_tok = tgt[:, 0]

            loss = 0
            for t in range(1, tgt.size(1)):
                output, hidden, _ = decoder(input_tok, hidden, enc_out, mask)
                loss += criterion(output, tgt[:, t])
                input_tok = tgt[:, t] if random.random() < 0.5 else output.argmax(dim=1)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() / (tgt.size(1) - 1)

        losses.append(total_loss / len(data))

    return TranslationBundle(encoder, decoder, vocab_ko, vocab_en, pad, bos, eos, device, losses)


def _load_bundle_from_checkpoint() -> TranslationBundle:
    device = torch.device("cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    vocab_ko = Vocab(checkpoint["vocab_ko_itos"])
    vocab_en = Vocab(checkpoint["vocab_en_itos"])
    pad, bos, eos = vocab_ko[PAD_TOKEN], vocab_ko[BOS_TOKEN], vocab_ko[EOS_TOKEN]

    encoder = Encoder(len(vocab_ko), pad).to(device)
    decoder = Decoder(len(vocab_en), pad).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])

    return TranslationBundle(encoder, decoder, vocab_ko, vocab_en, pad, bos, eos, device, checkpoint["losses"])


@st.cache_resource(show_spinner="번역 모델 준비 중...")
def train_bundle() -> TranslationBundle:
    """사전 학습된 체크포인트(streamlit_app/lib/seq2seq_checkpoint.pt)가 있으면 로드만 하고(수 초),
    없으면 즉석에서 3 epoch만 학습합니다(로컬에서 pretrain_seq2seq.py를 아직 안 돌린 경우의 폴백)."""
    if CHECKPOINT_PATH.exists():
        return _load_bundle_from_checkpoint()
    return _train_bundle_from_scratch()

"""로컬에서 한 번 실행해 Seq2Seq 번역 모델을 미리 학습시키고 체크포인트로 저장하는 스크립트.

배포판 Streamlit 앱은 이 체크포인트를 로드만 하므로(수 초), 앱 콜드스타트마다
학습을 다시 하지 않습니다. 노트북 원본(lab2_seq2seq_translation_modeling.ipynb)은
그대로 3 epoch를 유지하며, 이 스크립트는 배포판 전용입니다.

사용법: python streamlit_app/scripts/pretrain_seq2seq.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from lib import seq2seq_model as s2s

EPOCHS = 60
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "lib" / "seq2seq_checkpoint.pt"


def main():
    device = torch.device("cpu")
    torch.manual_seed(42)
    random.seed(42)

    with open(s2s.DATA_DIR / "train_kor.txt", encoding="utf-8") as f_ko, \
         open(s2s.DATA_DIR / "train_eng.txt", encoding="utf-8") as f_en:
        kor_lines = f_ko.read().strip().splitlines()
        eng_lines = f_en.read().strip().splitlines()
    pairs = list(zip(kor_lines, eng_lines))

    specials = [s2s.PAD_TOKEN, s2s.BOS_TOKEN, s2s.EOS_TOKEN]
    vocab_ko = s2s.build_vocab([specials] + [s2s.tok_ko(ko) for ko, _ in pairs], specials)
    vocab_en = s2s.build_vocab([specials] + [s2s.tok_en(en) for _, en in pairs], specials)
    pad, bos, eos = vocab_ko[s2s.PAD_TOKEN], vocab_ko[s2s.BOS_TOKEN], vocab_ko[s2s.EOS_TOKEN]

    def tensorize(pair):
        ko, en = pair
        src = [bos] + [vocab_ko[t] for t in s2s.tok_ko(ko)] + [eos]
        tgt = [bos] + [vocab_en[t] for t in s2s.tok_en(en)] + [eos]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

    data = [tensorize(p) for p in pairs]

    encoder = s2s.Encoder(len(vocab_ko), pad).to(device)
    decoder = s2s.Decoder(len(vocab_en), pad).to(device)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad)

    losses = []
    t0 = time.time()
    for epoch in range(EPOCHS):
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

        epoch_loss = total_loss / len(data)
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={epoch_loss:.4f}  elapsed={time.time() - t0:.1f}s", flush=True)

    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "vocab_ko_itos": vocab_ko.get_itos(),
            "vocab_en_itos": vocab_en.get_itos(),
            "losses": losses,
        },
        CHECKPOINT_PATH,
    )
    print(f"저장 완료: {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()

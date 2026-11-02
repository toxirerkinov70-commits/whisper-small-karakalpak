# transcribe.py dagi "halqa zaxirasi"ning test to'plamidagi ta'siri
# Ishga tushirish (07_eval ichida, CPU yoki GPU):  python eval_halqa_zaxira.py
# Faqat test_finetune_beam5.csv da halqa aniqlangan kliplar qayta o'qiladi (no_repeat_ngram_size=3),
# qolgan kliplar natijasi o'zgarmaydi. Sharoit eval_teng.py bilan bir xil: fp32, batch 1, max_new_tokens 440,
# model matni normalize.kichik_harf bilan tozalanadi.
# Zaxira sozlamalari (chegara, no_repeat=3) test to'plamida tanlanmagan, shuning uchun bu natija xolis.
import os
import sys

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging

logging.set_verbosity_error()

ILDIZ = ".."                                   # repo ildizi: DATASET_trimmed/ shu yerda
sys.path.insert(0, f"{ILDIZ}/04_normalize")
sys.path.insert(0, ILDIZ)
import normalize                               # noqa: E402
from transcribe import halqa_mi                # noqa: E402

FT_MODEL = os.environ.get("KAA_ASR_MODEL", "Toxirerkinovtohir/whisper-small-karakalpak")  # = checkpoint-5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

d = pd.read_csv("test_finetune_beam5.csv", keep_default_na=False)
halqa = [halqa_mi(r.hypothesis, r.davomiylik_s) for r in d.itertuples()]
print(f"test: {len(d)} klip, halqa aniqlangan: {sum(halqa)}", flush=True)

processor = WhisperProcessor.from_pretrained(FT_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(FT_MODEL).to(DEVICE).eval()   # float32

yangi = d.copy()
for i, r in d[halqa].iterrows():
    wav, sr = sf.read(f"{ILDIZ}/DATASET_trimmed/test/{r.idx}.flac", dtype="float32")
    x = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        out = model.generate(x, language="uz", task="transcribe", max_new_tokens=440,
                             num_beams=5, do_sample=False, no_repeat_ngram_size=3)
    yangi.at[i, "hypothesis"] = normalize.kichik_harf(processor.batch_decode(out, skip_special_tokens=True)[0])
    print(f"  idx {r.idx}: WER {r.wer:.1f}% -> {100 * jiwer.wer(r.reference, yangi.at[i, 'hypothesis']):.1f}%",
          flush=True)

yangi["wer"] = [round(100 * jiwer.wer(a, b), 2) for a, b in zip(yangi.reference, yangi.hypothesis)]
yangi["cer"] = [round(100 * jiwer.cer(a, b), 2) for a, b in zip(yangi.reference, yangi.hypothesis)]
yangi.to_csv("test_finetune_beam5_halqa_zaxira.csv", index=False)

for nom, x in [("beam5", d), ("beam5 + halqa zaxirasi", yangi)]:
    ref, hyp = x.reference.tolist(), x.hypothesis.tolist()
    print(f"{nom:24s} test WER {100 * jiwer.wer(ref, hyp):.2f}%  CER {100 * jiwer.cer(ref, hyp):.2f}%")

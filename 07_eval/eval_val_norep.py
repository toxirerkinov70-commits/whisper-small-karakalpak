# checkpoint-5000 ni VALIDATION'da beam5 + no_repeat_ngram_size=N bilan baholash
# Ishga tushirish (07_eval ichida, GPU'da):  python eval_val_norep.py 4   (yoki 6)
# Sharoit eval_val_beam5.py bilan bir xil, farqi faqat no_repeat_ngram_size
import os
import sys
import time

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging

logging.set_verbosity_error()

ILDIZ = ".."                                   # repo ildizi: DATASET_trimmed/ shu yerda
sys.path.insert(0, f"{ILDIZ}/04_normalize")
import normalize                               # noqa: E402

N = int(sys.argv[1])
FT_MODEL = os.environ.get("KAA_ASR_MODEL", "Toxirerkinovtohir/whisper-small-karakalpak")  # = checkpoint-5000
TIL = "uz"
TASK = "transcribe"
MAX_TOKEN = 440
KW = {"num_beams": 5, "do_sample": False, "no_repeat_ngram_size": N}
NOM = f"beam5_norep{N}"

df = pd.read_csv(f"{ILDIZ}/04_normalize/manifest_tayyor.csv")
val = df[df["split"] == "validation"].reset_index(drop=True)
val["yol"] = ILDIZ + "/" + val["fayl"].str.replace("\\", "/", regex=False)
print(f"validation: {len(val)} klip, {NOM}", flush=True)

model = WhisperForConditionalGeneration.from_pretrained(FT_MODEL, dtype=torch.float32).cuda().eval()
processor = WhisperProcessor.from_pretrained(FT_MODEL)

boshlanish = time.time()
satrlar = []
for n, r in val.iterrows():
    wav, sr = sf.read(r["yol"], dtype="float32")
    x = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.cuda()
    with torch.no_grad():
        out = model.generate(x, language=TIL, task=TASK, max_new_tokens=MAX_TOKEN, **KW)
    hyp = normalize.kichik_harf(processor.batch_decode(out, skip_special_tokens=True)[0])
    ref = r["matn_model"]
    satrlar.append({
        "idx": r["idx"], "davomiylik_s": r["yangi_s"], "reference": ref, "hypothesis": hyp,
        "wer": round(100 * jiwer.wer(ref, hyp), 2), "cer": round(100 * jiwer.cer(ref, hyp), 2),
    })
    if (n + 1) % 50 == 0:
        print(f"{n + 1}/{len(val)} klip, {round((time.time() - boshlanish) / 60, 1)} daqiqa", flush=True)

natija = pd.DataFrame(satrlar)
natija.to_csv(f"val_finetune_{NOM}.csv", index=False)
ref, hyp = natija["reference"].tolist(), natija["hypothesis"].tolist()
print({"model": "finetune", "dekodlash": NOM, "toplam": "validation", "klip": len(natija),
       "WER": round(100 * jiwer.wer(ref, hyp), 2), "CER": round(100 * jiwer.cer(ref, hyp), 2),
       "fayl": f"07_eval/val_finetune_{NOM}.csv",
       "daqiqa": round((time.time() - boshlanish) / 60, 1)}, flush=True)
print("TAYYOR", flush=True)

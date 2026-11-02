# Saqlangan checkpoint'larni validation'da greedy va beam5+no_repeat4 bilan baholash
# Ishga tushirish (06_finetune ichida, GPU'da):  python eval_checkpoints.py
# Kerak: train_whisper.py saqlagan whisper-small-kaa/checkpoint-* papkalari
import glob
import json

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging

logging.set_verbosity_error()

CHIQISH = "./whisper-small-kaa"
TIL = "uz"
BATCH = 16
DEKODLASH = {
    "greedy": {},
    "beam5_norep4": {"num_beams": 5, "no_repeat_ngram_size": 4},
}

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=TIL, task="transcribe")

df = pd.read_csv("../04_normalize/manifest_tayyor.csv")
df = df[df["split"] == "validation"].reset_index(drop=True)
df["fayl"] = "../" + df["fayl"].str.replace("\\", "/", regex=False)
haqiqiy = df["matn_model"].tolist()

# Audio bir marta o'qiladi va log-mel'ga o'tkaziladi
xususiyatlar = []
for i in range(0, len(df), BATCH):
    wavlar = [sf.read(f, dtype="float32")[0] for f in df["fayl"][i:i + BATCH]]
    xususiyatlar.append(processor.feature_extractor(wavlar, sampling_rate=16000, return_tensors="pt").input_features)

checkpointlar = sorted(glob.glob(f"{CHIQISH}/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
natijalar = []
for ckpt in checkpointlar:
    model = WhisperForConditionalGeneration.from_pretrained(ckpt, dtype=torch.bfloat16).cuda().eval()
    for nom, kw in DEKODLASH.items():
        taxmin = []
        for x in xususiyatlar:
            with torch.no_grad():
                out = model.generate(x.cuda().to(torch.bfloat16), language=TIL, task="transcribe",
                                     max_new_tokens=444, **kw)
            taxmin += [t.strip() for t in processor.batch_decode(out, skip_special_tokens=True)]
        # Halqa: taxmin haqiqiydan 1.5 baravardan ko'p so'zli
        halqa = sum(len(t.split()) > 1.5 * len(h.split()) for t, h in zip(taxmin, haqiqiy))
        qator = {
            "checkpoint": ckpt.split("/")[-1],
            "dekodlash": nom,
            "WER": round(100 * jiwer.wer(haqiqiy, taxmin), 2),
            "CER": round(100 * jiwer.cer(haqiqiy, taxmin), 2),
            "halqa_klip": halqa,
        }
        print(qator, flush=True)
        natijalar.append(qator)
        pd.DataFrame({"fayl": df["fayl"], "haqiqiy": haqiqiy, "taxmin": taxmin}).to_csv(
            f"{CHIQISH}/val_taxmin_{qator['checkpoint']}_{nom}.csv", index=False)
    del model
    torch.cuda.empty_cache()

pd.DataFrame(natijalar).to_csv(f"{CHIQISH}/checkpoint_baho.csv", index=False)
print(json.dumps(natijalar, ensure_ascii=False, indent=1))

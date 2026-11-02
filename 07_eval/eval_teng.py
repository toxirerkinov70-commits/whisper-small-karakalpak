# Baseline va fine-tune (checkpoint-5000) ni TENG sharoitda baholash
# Ishga tushirish (07_eval ichida, GPU'da):  python eval_teng.py
# Teng sharoit: fp32, batch=1, max_new_tokens=440 (baseline'dagi kabi), bitta reference (matn_model),
# model matni ikkala modelda ham normalize.kichik_harf bilan tozalanadi.
import os
import re
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

FT_MODEL = os.environ.get("KAA_ASR_MODEL", "Toxirerkinovtohir/whisper-small-karakalpak")  # = checkpoint-5000
BASE_MODEL = "openai/whisper-small"
FT_TIL = "uz"                                  # train_whisper.py dagi TIL va task
FT_TASK = "transcribe"
BASE_TIL = "kk"                                # baseline kk -> lotin varianti (whisper_baho.ipynb)
MAX_TOKEN = 440                                # whisper_baho.ipynb dagi max_new_tokens
DEKODLASH = {
    "greedy": {"num_beams": 1, "do_sample": False},
    "beam5": {"num_beams": 5, "do_sample": False},
    "beam5_norep4": {"num_beams": 5, "do_sample": False, "no_repeat_ngram_size": 4},
}

# kk_lotin.ipynb dagi jadval va funksiya, o'zgarishsiz
HARFLAR = {
    "а": "a", "ә": "á", "б": "b", "в": "v", "г": "g", "ғ": "ǵ", "д": "d",
    "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ń", "о": "o", "ө": "ó",
    "п": "p", "р": "r", "с": "s", "т": "t", "ұ": "u", "ү": "ú", "ф": "f",
    "х": "x", "һ": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "",
    "ы": "ı", "і": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
UNLILAR = "аәеёиоөұүыіэюяaáeioóuúı"


def lotinga(matn):
    matn = matn.lower()
    # "у": unlidan keyin -> "w" (жау -> jaw), aks holda -> "u"
    matn = re.sub(rf"(?<=[{UNLILAR}])у", "w", matn)
    matn = matn.replace("у", "u")
    natija = ""
    for harf in matn:
        natija += HARFLAR.get(harf, harf)
    return natija


# 1. Kliplar: to'liq test va eval_50 (Windows yo'li -> Linux yo'li)
df = pd.read_csv(f"{ILDIZ}/04_normalize/manifest_tayyor.csv")
test = df[df["split"] == "test"].reset_index(drop=True)
test["yol"] = ILDIZ + "/" + test["fayl"].str.replace("\\", "/", regex=False)
eval50 = pd.read_csv(f"{ILDIZ}/05_baseline/eval_50.csv")
eval50["yol"] = ILDIZ + "/" + eval50["fayl"].str.replace("\\", "/", regex=False)
for nom, d in [("eval_50", eval50), ("test", test)]:
    topildi = sum(os.path.exists(y) for y in d["yol"])
    print(f"{nom}: {topildi}/{len(d)} fayl topildi", flush=True)
    assert topildi == len(d)
assert eval50["idx"].isin(test["idx"]).all()

# Transliteratsiya baseline'dagi bilan bir xil natija berishini tekshirish
eski = pd.read_csv(f"{ILDIZ}/05_baseline/eval_50_natija.csv")
qayta = [normalize.kichik_harf(lotinga(str(t))) for t in eski["model_kk"]]
assert qayta == eski["model_kk_lotin"].fillna("").tolist(), "lotinga() baseline bilan mos emas"
print("lotinga(): eval_50_natija.csv dagi 50 ta natija aynan takrorlandi", flush=True)


def baholash(model, processor, d, til, kw, lotin=False):
    satrlar = []
    for _, r in d.iterrows():
        wav, sr = sf.read(r["yol"], dtype="float32")
        x = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.cuda()
        with torch.no_grad():
            out = model.generate(x, language=til, task=FT_TASK, max_new_tokens=MAX_TOKEN, **kw)
        xom = processor.batch_decode(out, skip_special_tokens=True)[0]
        hyp = normalize.kichik_harf(lotinga(xom) if lotin else xom)
        ref = r["matn_model"]
        satrlar.append({
            "idx": r["idx"], "davomiylik_s": r["yangi_s"], "reference": ref, "hypothesis": hyp,
            "wer": round(100 * jiwer.wer(ref, hyp), 2), "cer": round(100 * jiwer.cer(ref, hyp), 2),
        })
    return pd.DataFrame(satrlar)


def umumiy(natija):
    ref, hyp = natija["reference"].tolist(), natija["hypothesis"].tolist()
    return round(100 * jiwer.wer(ref, hyp), 2), round(100 * jiwer.cer(ref, hyp), 2)


ishlar = []  # (model, dekodlash, to'plam)
for dek in DEKODLASH:
    ishlar += [("finetune", dek, "eval50"), ("finetune", dek, "test")]
ishlar += [("baseline_kk_lotin", "greedy", "test")]
# Parallel ishlatish uchun: python eval_teng.py finetune:beam5_norep4 baseline_kk_lotin:greedy
# (argumentsiz - hammasi ketma-ket)
TANLOV = sys.argv[1:]
if TANLOV:
    ishlar = [i for i in ishlar if f"{i[0]}:{i[1]}" in TANLOV]

jadval = []
modellar = {}
for model_nom, dek, toplam in ishlar:
    if model_nom not in modellar:
        yol = FT_MODEL if model_nom == "finetune" else BASE_MODEL
        modellar.clear()
        torch.cuda.empty_cache()
        modellar[model_nom] = (
            WhisperForConditionalGeneration.from_pretrained(yol, dtype=torch.float32).cuda().eval(),
            WhisperProcessor.from_pretrained(yol),
        )
    model, processor = modellar[model_nom]
    d = eval50 if toplam == "eval50" else test
    til = FT_TIL if model_nom == "finetune" else BASE_TIL
    boshlanish = time.time()
    natija = baholash(model, processor, d, til, DEKODLASH[dek], lotin=(model_nom != "finetune"))
    fayl = f"{toplam}_{model_nom}_{dek}.csv"
    natija.to_csv(fayl, index=False)
    wer, cer = umumiy(natija)
    qator = {"model": model_nom, "dekodlash": dek, "toplam": toplam, "klip": len(natija),
             "WER": wer, "CER": cer, "fayl": f"07_eval/{fayl}", "daqiqa": round((time.time() - boshlanish) / 60, 1)}
    print(qator, flush=True)
    jadval.append(qator)

    # Baseline 50 klipda: to'liq test natijasidan aynan o'sha 50 ta idx
    if model_nom != "finetune":
        q50 = natija[natija["idx"].isin(eval50["idx"])].set_index("idx").loc[eval50["idx"]].reset_index()
        q50.to_csv(f"eval50_{model_nom}_{dek}.csv", index=False)
        wer, cer = umumiy(q50)
        qator = {"model": model_nom, "dekodlash": dek, "toplam": "eval50", "klip": len(q50),
                 "WER": wer, "CER": cer, "fayl": f"07_eval/eval50_{model_nom}_{dek}.csv", "daqiqa": 0}
        print(qator, flush=True)
        jadval.append(qator)

qoshimcha = "_" + "_".join(t.replace(":", "-") for t in TANLOV) if TANLOV else ""
pd.DataFrame(jadval).to_csv(f"yakuniy_jadval{qoshimcha}.csv", index=False)
print("TAYYOR", flush=True)

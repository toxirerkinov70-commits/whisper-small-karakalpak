# 50 klipda eski reference (matn_model) va gold reference (gold_50.csv) bilan WER/CER
# Ishga tushirish (07_eval ichida):  python gold50.py
# Model qayta ishlatilmaydi: hypothesis'lar eval50_*.csv fayllaridan olinadi
import sys

import jiwer
import pandas as pd

sys.path.insert(0, "../04_normalize")
import normalize                               # noqa: E402

gold = pd.read_csv("../05_baseline/gold_50.csv", keep_default_na=False)
eski = pd.read_csv("../05_baseline/eval_50.csv")
assert gold["idx"].tolist() == eski["idx"].tolist()

# 1. Holatlar va izohlar
print(gold["holat"].value_counts().to_string(), "\n")
for _, r in gold[gold["izoh"] != ""].iterrows():
    print(f"  {r['idx']:>4}  {r['holat']:5}  {r['izoh']}")
print()

# 2. Gold reference: "xato" -> togri_matn (kichik_harf bilan), aks holda matn_model
# 558: gold_50.csv da togri_matn qisqartma ("...model uchun matnning hammasi... n rahmonova óza"),
# foydalanuvchi tasdiqlagan ma'no: matn_model + " óza" (audioda "ÓzA" aytilgan)
TOGRI_MATN = dict(zip(gold["idx"], gold["togri_matn"]))
TOGRI_MATN[558] = eski.set_index("idx").loc[558, "matn_model"] + " óza"

gold["eski_ref"] = eski["matn_model"].tolist()
gold["gold_ref"] = [
    normalize.kichik_harf(TOGRI_MATN[i]) if h == "xato" else e
    for i, h, e in zip(gold["idx"], gold["holat"], gold["eski_ref"])
]
for _, r in gold[gold["holat"] == "xato"].iterrows():
    print(f"{r['idx']} ESKI: {r['eski_ref']}\n{r['idx']} GOLD: {r['gold_ref']}\n")

# 3. Mavjud hypothesis'lar bilan WER/CER
VARIANTLAR = {
    "baseline_kk_lotin greedy": "eval50_baseline_kk_lotin_greedy.csv",
    "finetune greedy": "eval50_finetune_greedy.csv",
    "finetune beam5": "eval50_finetune_beam5.csv",
    "finetune beam5_norep4": "eval50_finetune_beam5_norep4.csv",
}
audiosiz = gold["holat"] != "audio"
jadval = []
hyplar = {}
for nom, fayl in VARIANTLAR.items():
    d = pd.read_csv(fayl, keep_default_na=False)
    assert d["idx"].tolist() == gold["idx"].tolist()
    assert d["reference"].tolist() == gold["eski_ref"].tolist()
    hyplar[nom] = d["hypothesis"]
    for toplam, maska in [("50 klip", gold["idx"].notna()), ("audio'siz", audiosiz)]:
        hyp = d.loc[maska, "hypothesis"].tolist()
        qator = {"variant": nom, "toplam": toplam, "klip": int(maska.sum())}
        for ref_nom in ["eski_ref", "gold_ref"]:
            ref = gold.loc[maska, ref_nom].tolist()
            qator[f"WER_{ref_nom[:-4]}"] = round(100 * jiwer.wer(ref, hyp), 2)
            qator[f"CER_{ref_nom[:-4]}"] = round(100 * jiwer.cer(ref, hyp), 2)
        qator["fayl"] = f"07_eval/{fayl}"
        jadval.append(qator)

jadval = pd.DataFrame(jadval)
print(jadval.drop(columns="fayl").to_string(index=False))
jadval.to_csv("gold50_jadval.csv", index=False)

# 4. Per-clip (beam5)
b5 = hyplar["finetune beam5"]
per = pd.DataFrame({
    "idx": gold["idx"], "holat": gold["holat"], "izoh": gold["izoh"],
    "eski_ref": gold["eski_ref"], "gold_ref": gold["gold_ref"], "beam5_hyp": b5,
    "beam5_wer_eski": [round(100 * jiwer.wer(r, h), 2) for r, h in zip(gold["eski_ref"], b5)],
    "beam5_wer_gold": [round(100 * jiwer.wer(r, h), 2) for r, h in zip(gold["gold_ref"], b5)],
})
per.to_csv("gold50_taqqoslash.csv", index=False)
print("\nFarq qilgan kliplar (beam5):")
print(per[per["beam5_wer_eski"] != per["beam5_wer_gold"]][["idx", "holat", "beam5_wer_eski", "beam5_wer_gold"]].to_string(index=False))

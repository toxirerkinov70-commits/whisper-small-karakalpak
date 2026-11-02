# "ÓzA" olib tashlangan kliplar soni (train/validation/test)
# Ishga tushirish (07_eval ichida):  python oza_soni.py
# Qoida normalize.tuzat dagi bilan bir xil: NFC, keyin matn oxiridagi ",?\s*ÓzA\W*$" o'chiriladi.
# Python re ishlatiladi (normalize.py kabi): pandas 3 dagi str.contains pyarrow/RE2 bilan ishlaydi,
# unda \b \s \W faqat ASCII'ni taniydi va natija boshqacha chiqadi.
import re
import unicodedata

import pandas as pd

df = pd.read_csv("../04_normalize/manifest_tayyor.csv")   # yakuniy manifest (dublikatlarsiz)
matn = [unicodedata.normalize("NFC", t) for t in df["matn"]]
oxirida = pd.Series([bool(re.search(r",?\s*ÓzA\W*$", t)) for t in matn], index=df.index)
umuman = pd.Series(["ÓzA" in t for t in matn], index=df.index)

# Tekshiruv: olib tashlangandan keyin ham oxirida "óza" qolganlar (masalan "ÓzA ÓzA" - faqat bittasi o'chgan)
ikki_marta = [(s, i) for s, i, m in zip(df.loc[oxirida, "split"], df.loc[oxirida, "idx"], df.loc[oxirida, "matn_model"])
              if m.split()[-1] == "óza"]
print("Olib tashlangandan keyin ham oxirida 'óza' qolgan:", ikki_marta)

jadval = pd.DataFrame({
    "jami_klip": df.groupby("split").size(),
    "OzA_olib_tashlangan": oxirida.groupby(df["split"]).sum(),
    "OzA_matn_ichida_qolgan": (umuman & ~oxirida).groupby(df["split"]).sum(),
}).loc[["train", "validation", "test"]]
jadval.loc["jami"] = jadval.sum()
jadval["foiz"] = (100 * jadval["OzA_olib_tashlangan"] / jadval["jami_klip"]).round(1)
print(jadval.to_string())
jadval.to_csv("oza_soni.csv")

# Ichida qolganlar qanday ko'rinishda (masalan "ÓzAnıń xabarshısı")
qolgan = [re.search(r"\S*ÓzA\S*", t).group() for t, q in zip(matn, umuman & ~oxirida) if q]
print("\nIchida qolgan shakllar:", pd.Series(qolgan).value_counts().to_dict())

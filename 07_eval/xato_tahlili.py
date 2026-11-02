# Payshanba vazifasi: xatolar tahlili (top sabablar, har biriga misollar)
# Ishga tushirish (07_eval ichida):  python xato_tahlili.py
# Ma'lumot: val_finetune_beam5.csv (validation 1 034 klip, checkpoint-5000, beam5, no_repeat yo'q)
# Har bir so'z xatosi (S/D/I) bitta sababga biriktiriladi. Tartib (birinchi mos kelgani olinadi):
#   1 halqa  2 gap boshi  3 son so'zi  4 atoqli ot / qisqartma  5 qo'shimcha (o'zak bir xil)
#   6 harf chalkashligi (1-2 harf farq)  7 kichik yordamchi so'z (tushib qolish / qo'shilish)  8 boshqa
import re
from collections import Counter, defaultdict

import jiwer
import pandas as pd

d = pd.read_csv("val_finetune_beam5.csv", keep_default_na=False)
m = pd.read_csv("../04_normalize/manifest_tayyor.csv")
m = m[m["split"] == "validation"].set_index("idx")
t = pd.read_csv("../03_trim/trim_manifest_yakuniy.csv")
t = t[t["split"] == "validation"].set_index("idx")
a = pd.read_csv("../02_audio_features/audio_features_silero.csv")
a = a[a["split"] == "validation"].set_index("idx")

SON_ILDIZ = ("bir", "eki", "úsh", "tórt", "bes", "altı", "jeti", "segiz", "toǵız", "on", "jigirma", "otız",
             "qırq", "eliw", "alpıs", "jetpis", "seksen", "toqsan", "júz", "mıń", "million", "milliard", "pútin")
YORDAMCHI = {"da", "de", "hám", "menen", "penen", "benen", "ushın", "ǵana", "bul", "ol", "sol", "usı", "da", "aq", "ta", "te"}


def son_mi(soz):
    return any(soz == s or (soz.startswith(s) and re.search(r"(inshi|ınshı|nshi|nshı)$", soz)) for s in SON_ILDIZ)


def harf_masofasi(p, q):
    """Levenshtein masofasi: p ni q ga aylantirish uchun nechta harfni o'zgartirish/qo'shish/o'chirish kerak."""
    d = list(range(len(q) + 1))
    for i, a in enumerate(p, 1):
        oldingi, d[0] = d[0], i
        for j, c in enumerate(q, 1):
            oldingi, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, oldingi + (a != c))
    return d[-1]


def umumiy_bosh(x, y):
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def qoshimcha_mi(r, h):
    n = umumiy_bosh(r, h)
    return n >= 3 and n >= 0.5 * min(len(r), len(h))


def katta_harfli_sozlar(matn):
    """Asl matndagi so'zlar: (so'z, gap boshi emas va katta harf bilan) ro'yxati, matn_model tartibida."""
    natija = []
    oldingi_gap_oxiri = True
    for bolak in matn.split():
        qismlar = [q for q in re.sub(r"[^\w\s]|_", " ", bolak).split()]
        for q in qismlar:
            natija.append(q[:1].isupper() and not oldingi_gap_oxiri)
            oldingi_gap_oxiri = False
        if re.search(r"[.!?]$", bolak):
            oldingi_gap_oxiri = True
    return natija


sabab_soni = Counter()
misollar = defaultdict(list)
bosh_xato, bosh_jami = Counter(), Counter()
jami_xato = 0
klip_qatorlari = []

for r in d.itertuples():
    ref, hyp = r.reference, r.hypothesis
    rw = ref.split()
    halqa = len(hyp.split()) > 1.5 * len(rw)
    o = jiwer.process_words(ref, hyp)
    hw = o.hypotheses[0]
    katta = katta_harfli_sozlar(m.loc[r.idx, "matn"])
    katta_ok = len(katta) == len(rw)                 # normalizatsiya so'z sonini o'zgartirmagan kliplardagina
    xato_soni = o.substitutions + o.deletions + o.insertions
    jami_xato += xato_soni
    birinchi_xato = False

    for ch in o.alignments[0]:
        if ch.type == "equal":
            continue
        ri = list(range(ch.ref_start_idx, ch.ref_end_idx))
        hi = list(range(ch.hyp_start_idx, ch.hyp_end_idx))
        juftlar = []
        if ch.type == "substitute":
            juftlar = [("S", i, j) for i, j in zip(ri, hi)]
        elif ch.type == "delete":
            juftlar = [("D", i, None) for i in ri]
        else:
            juftlar = [("I", ch.ref_start_idx, j) for j in hi]
        for tur, i, j in juftlar:
            rs = rw[i] if tur != "I" and i < len(rw) else ""
            hs = hw[j] if j is not None else ""
            if halqa:
                sabab = "1 halqa"
            elif i == 0:
                sabab = "2 gap boshi"
                birinchi_xato = True
            elif son_mi(rs) or son_mi(hs):
                sabab = "3 son so'zi"
            elif tur != "I" and katta_ok and katta[i]:
                sabab = "4 atoqli ot / qisqartma"
            elif tur == "S" and qoshimcha_mi(rs, hs):
                sabab = "5 qo'shimcha (o'zak bir xil)"
            elif tur == "S" and harf_masofasi(rs, hs) <= 2:
                sabab = "6 harf chalkashligi (1-2 harf)"
            elif (tur == "D" and rs in YORDAMCHI) or (tur == "I" and hs in YORDAMCHI):
                sabab = "7 yordamchi so'z tushgan/qo'shilgan"
            else:
                sabab = "8 boshqa"
            sabab_soni[sabab] += 1
            if True:
                misollar[sabab].append({"idx": r.idx, "tur": tur, "haqiqiy": rs, "model": hs})

    # gap boshi: trim bilan bog'liqmi
    kesilgan = t.loc[r.idx, "kesilgan_bosh"] > 0 if r.idx in t.index else None
    if not halqa and rw:
        bosh_jami[kesilgan] += 1
        bosh_xato[kesilgan] += int(birinchi_xato)
    klip_qatorlari.append({"idx": r.idx, "wer": r.wer, "halqa": halqa, "snr": a.loc[r.idx, "snr"] if r.idx in a.index else None,
                           "kategoriya": a.loc[r.idx, "kategoriya"] if r.idx in a.index else None})

print("Jami so'z xatolari (S+D+I):", jami_xato)
print("\nSABABLAR ULUSHI:")
jadval = []
for sabab in sorted(sabab_soni):
    n = sabab_soni[sabab]
    jadval.append({"sabab": sabab, "xato_soni": n, "ulush_%": round(100 * n / jami_xato, 1)})
    print(f"  {sabab:38s} {n:5d}  {100 * n / jami_xato:5.1f}%")
pd.DataFrame(jadval).to_csv("xato_sabablari.csv", index=False)

print("\nGAP BOSHI va TRIM (halqasiz kliplar): birinchi so'z xato ulushi")
for k in [True, False]:
    if bosh_jami[k]:
        print(f"  boshi kesilgan={k}: {bosh_xato[k]}/{bosh_jami[k]} = {100 * bosh_xato[k] / bosh_jami[k]:.1f}%")

k = pd.DataFrame(klip_qatorlari)
k["snr_guruh"] = pd.cut(k["snr"].clip(upper=60), [-100, 15, 25, 35, 1000], labels=["<15 dB", "15-25", "25-35", ">35"])
print("\nAUDIO SIFATI (SNR) va WER (halqasiz kliplar, klip WER o'rtachasi):")
print(k[~k["halqa"]].groupby("snr_guruh", observed=True)["wer"].agg(["count", "mean"]).round(1))

pd.DataFrame([dict(sabab=s, **x) for s, xs in misollar.items() for x in xs]).to_csv("xato_misollari.csv", index=False)
for sabab in sorted(misollar):
    print(f"\n--- {sabab}: eng ko'p uchraganlar")
    c = Counter((x["tur"], x["haqiqiy"], x["model"]) for x in misollar[sabab])
    for (tur, h, mo), n in c.most_common(8):
        idlar = sorted({x["idx"] for x in misollar[sabab] if (x["tur"], x["haqiqiy"], x["model"]) == (tur, h, mo)})[:3]
        print(f"   {tur} '{h}' -> '{mo}'  x{n}  (idx {idlar})")

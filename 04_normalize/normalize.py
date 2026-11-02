import re
import unicodedata

KATTA = "A-ZÁÓÚǴŃ"

# Rim raqami deb o'qilib, so'zga aylanib ketgan bosh harflar
HARF_SOZ = {
    "M": "mıńınshı",
    "I": "birinshi",
    "V": "besinshi",
    "X": "onınshı",
    "L": "eliwinshi",
    "C": "júzinshi",
}

BIRLIK = {
    "mln": "million",
    "mlrd": "milliard",
    "trln": "trillion",
    "kg": "kilogramm",
    "km": "kilometr",
    "kVt": "kilovatt",
}


def rim_tuzat(matn, raw):
    # raw: "M.Yusupova"  ->  matn: "mıńınshı. Yusupova" ni "M. Yusupova" ga qaytaramiz
    juftlar = set(re.findall(rf"\b([IVXLCM])\.\s?([{KATTA}]\w+)", raw))
    for harf, ism in juftlar:
        soz = HARF_SOZ[harf]
        matn = re.sub(rf"\b{soz}\.\s+{re.escape(ism)}\b", f"{harf}. {ism}", matn)
    return matn


def dade_tuzat(matn, raw):
    # raw: "basqa da"  ->  matn: "basqada" ni "basqa da" ga ajratamiz
    juftlar = set(re.findall(r"(\w+) (da|de)\b", raw))
    for soz, yuk in juftlar:
        matn = re.sub(rf"\b{re.escape(soz)}{yuk}\b", f"{soz} {yuk}", matn)
    return matn


def birlik_tuzat(matn):
    for qisqa, toliq in BIRLIK.items():
        matn = re.sub(rf"\b{qisqa}\b\.?", toliq, matn)
    return matn


def tuzat(matn, raw):
    matn = unicodedata.normalize("NFC", matn)
    raw = unicodedata.normalize("NFC", raw)
    raw = raw.replace("ǝ", "á").replace("ə", "á")

    matn = matn.replace("ȳ", "w").replace("ỹ", "w")
    matn = rim_tuzat(matn, raw)
    matn = dade_tuzat(matn, raw)
    matn = birlik_tuzat(matn)
    matn = re.sub(r",?\s*ÓzA\W*$", "", matn)   # matn oxiridagi "ÓzA" audioda aytilmaydi
    return matn


def kichik_harf(matn):
    matn = matn.lower()
    # "Issıxana".lower() -> "issıxana" bo'ladi, to'g'risi "ıssıxana"
    matn = re.sub(r"\bissı", "ıssı", matn)
    matn = re.sub(r"\biqsham", "ıqsham", matn)
    matn = re.sub(r"[^\w\s]|_", " ", matn)       # harf va probeldan boshqa hamma belgi -> probel
    matn = re.sub(r"\s+", " ", matn).strip()
    return matn


def model_matni(matn, raw):
    return kichik_harf(tuzat(matn, raw))

"""04_normalize/normalize.py: modelga beriladigan matnni tayyorlash qoidalari."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_normalize"))
import normalize as N  # noqa: E402


@pytest.mark.parametrize("matn, raw, kutilgan", [
    # qisqartma birliklar to'liq so'zga
    ("5 mln. som", "5 mln. som", "5 million som"),
    ("2 mlrd", "2 mlrd", "2 milliard"),
    # rim raqami deb so'zga aylanib ketgan bosh harf asliga qaytadi
    ("mıńınshı. Yusupova aytti", "M.Yusupova aytti", "M. Yusupova aytti"),
    # "da/de" yuklamasi asl matndagidek ajratiladi
    ("basqada jumıslar", "basqa da jumıslar", "basqa da jumıslar"),
    # matn oxiridagi "ÓzA" audioda aytilmaydi
    ("Jańalıqlar, ÓzA", "Jańalıqlar, ÓzA", "Jańalıqlar"),
    # ȳ -> w
    ("ȳaqıt", "ȳaqıt", "waqıt"),
])
def test_tuzat(matn, raw, kutilgan):
    assert N.tuzat(matn, raw) == kutilgan


def test_kichik_harf_tinish_belgilari_va_probellar():
    assert N.kichik_harf("Sh. Mirziyoev: «Jańa Ózbekstan»_2026") == "sh mirziyoev jańa ózbekstan 2026"


@pytest.mark.parametrize("katta, kutilgan", [
    # Python "I".lower() -> "i" beradi, qoraqalpoqchada esa bu so'zlar "ı" bilan
    ("Issıxana", "ıssıxana"),
    ("Iqsham", "ıqsham"),
])
def test_kichik_harf_nuqtasiz_i(katta, kutilgan):
    assert N.kichik_harf(katta) == kutilgan


def test_model_matni_ikkala_bosqich():
    assert N.model_matni("Issıxana Qurıldı, 5 mln. som!", "Issıxana Qurıldı, 5 mln. som!") == \
        "ıssıxana qurıldı 5 million som"


def test_nfc_normalizatsiya():
    # "ó" ikki xil kodlanishi (bitta belgi va o + birikuvchi aksent) bir xil natija beradi
    assert N.tuzat("ózbek", "ózbek") == N.tuzat("ózbek", "ózbek")

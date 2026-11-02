"""transcribe.py ning modelsiz qismlari: halqa detektori, uzun audioni bo'laklash, fayl qidirish."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import transcribe as T  # noqa: E402


@pytest.mark.parametrize("matn, soniya, kutilgan", [
    ("parlament qadaǵalawın alıp barıwda tiykarǵı itibar", 4.0, False),   # oddiy gap
    ("qalqan qalqan qalqan qalqan", 10.0, True),                           # bir so'z 4 marta ketma-ket
    ("qalqan qalqan qalqan", 10.0, False),                                 # 3 marta hali halqa emas
    (" ".join(["bir", "eki"] * 30), 5.0, True),                            # soniyasiga 12 so'z
    ("", 0.0, False),
])
def test_halqa_mi(matn, soniya, kutilgan):
    assert T.halqa_mi(matn, soniya) is kutilgan


def _nutq(soniya, seed=0):
    """Nutqqa o'xshash signal: shovqinli tovush."""
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(int(soniya * T.SR))).astype(np.float32)


def test_qisqa_audio_bolinmaydi():
    wav = _nutq(12)
    bolaklar = T.bolaklarga_bol(wav)
    assert len(bolaklar) == 1 and len(bolaklar[0]) == len(wav)


def test_uzun_audio_pauzadan_kesiladi():
    # 25 s nutq + 1 s jimlik + 20 s nutq: kesish 20-30 s oralig'idagi jimlikka tushishi kerak
    wav = np.concatenate([_nutq(25, 1), np.zeros(T.SR, dtype=np.float32), _nutq(20, 2)])
    bolaklar = T.bolaklarga_bol(wav)
    assert len(bolaklar) == 2
    kesish = len(bolaklar[0]) / T.SR
    assert 25.0 <= kesish <= 26.0
    assert sum(len(b) for b in bolaklar) == len(wav)          # hech narsa yo'qolmaydi
    assert all(len(b) <= T.BOLAK for b in bolaklar)            # har bo'lak Whisper oynasiga sig'adi


def test_juda_uzun_audio_har_bolak_30_soniyagacha():
    wav = _nutq(125, 3)
    bolaklar = T.bolaklarga_bol(wav)
    assert len(bolaklar) >= 5
    assert all(len(b) <= T.BOLAK for b in bolaklar)
    assert sum(len(b) for b in bolaklar) == len(wav)


def test_audio_fayllar(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.FLAC").write_bytes(b"")
    (tmp_path / "izoh.txt").write_text("x")
    ichki = tmp_path / "ichki"
    ichki.mkdir()
    (ichki / "c.mp3").write_bytes(b"")
    topildi = sorted(Path(p).name for p in T.audio_fayllar([str(tmp_path)]))
    assert topildi == ["a.wav", "b.FLAC", "c.mp3"]


def test_namunalar_repoda_bor():
    for nom in ("namuna_1.flac", "namuna_2.flac", "namuna_3.flac"):
        assert (ROOT / "examples" / nom).stat().st_size > 10_000

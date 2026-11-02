# trim_utils.py: multiprocessing ishchi jarayonlari uchun funksiyalar
import io, json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

SR = 16000
PAD_MS = 200
THRESHOLD = 0.4
MIN_SPEECH_MS = 250
MIN_ULUSH = 0.7
MAX_CPS = 24

USTUNLAR = ["split", "idx", "eski_s", "yangi_s", "start_s", "end_s",
            "kesilgan_bosh", "kesilgan_oxir", "nutq_bosh_s", "nutq_oxir_s",
            "n_seg", "nutqsiz", "rad_sababi", "segmentlar", "matn", "xato"]

_ds = None      # har bir ishchi jarayonning o'z dataseti
_model = None   # har bir ishchi jarayonning o'z Silero modeli


def ishchi_boshlash(cache_dir):
    """Har bir ishchi jarayon ishga tushganda BIR MARTA chaqiriladi."""
    global _ds, _model
    torch.set_num_threads(1)
    from datasets import load_dataset, Audio as HFAudio
    from silero_vad import load_silero_vad
    _ds = load_dataset("atikuwu/karakalpak-speech-corpus", cache_dir=cache_dir)
    _ds = _ds.cast_column("audio", HFAudio(decode=False))
    _model = load_silero_vad()


def audio_oqish(split, idx):
    qator = _ds[split][idx]
    a = qator["audio"]
    if a["bytes"] is not None:
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    else:
        wav, sr = sf.read(a["path"], dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32), qator["text"]


def trim_qil(wav, matn=None, threshold=THRESHOLD, pad_ms=PAD_MS):
    from silero_vad import get_speech_timestamps
    segmentlar = get_speech_timestamps(
        torch.from_numpy(wav), _model,
        sampling_rate=SR, threshold=threshold,
        min_speech_duration_ms=MIN_SPEECH_MS, speech_pad_ms=0,
    )
    eski_s = len(wav) / SR
    info = dict(
        eski_s=eski_s, yangi_s=eski_s, start_s=0.0, end_s=eski_s,
        kesilgan_bosh=0.0, kesilgan_oxir=0.0,
        nutq_bosh_s=np.nan, nutq_oxir_s=np.nan,
        n_seg=len(segmentlar), nutqsiz=len(segmentlar) == 0,
        segmentlar=json.dumps([[s["start"], s["end"]] for s in segmentlar]),
        rad_sababi="",
    )
    if info["nutqsiz"]:
        return wav, info

    pad = int(pad_ms * SR / 1000)
    nutq_bosh = segmentlar[0]["start"]
    nutq_oxir = segmentlar[-1]["end"]
    bosh = max(0, nutq_bosh - pad)
    oxir = min(len(wav), nutq_oxir + pad)
    yangi_s = (oxir - bosh) / SR
    info.update(nutq_bosh_s=nutq_bosh / SR, nutq_oxir_s=nutq_oxir / SR)

    sabablar = []
    if yangi_s < MIN_ULUSH * eski_s:
        sabablar.append("qisqargan")
    if matn:
        if len(matn) / yangi_s > MAX_CPS and len(matn) / eski_s <= MAX_CPS:
            sabablar.append("cps")
    if sabablar:
        info["rad_sababi"] = ",".join(sabablar)
        return wav, info

    info.update(yangi_s=yangi_s, start_s=bosh / SR, end_s=oxir / SR,
                kesilgan_bosh=bosh / SR, kesilgan_oxir=eski_s - oxir / SR)
    return wav[bosh:oxir], info


def bitta_ishlov(vazifa):
    """Bitta audio: o'qish -> trim -> FLAC yozish -> manifest qatori."""
    split, idx, chiqish = vazifa
    natija = {"split": split, "idx": idx, "xato": ""}
    try:
        wav, matn = audio_oqish(split, idx)
        kesilgan, info = trim_qil(wav, matn)
        papka = Path(chiqish) / split
        papka.mkdir(parents=True, exist_ok=True)
        sf.write(papka / f"{idx}.flac", np.clip(kesilgan, -1.0, 1.0), SR,
                 format="FLAC", subtype="PCM_16")
        natija.update(info)
        natija["matn"] = matn
    except Exception as e:
        natija["xato"] = repr(e)
    return natija



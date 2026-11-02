# ============================================================
# INFERENCE: audio -> qoraqalpoqcha matn (fine-tune qilingan Whisper-small)
# ============================================================
# Nima qiladi: bitta audio fayl yoki butun papkani oladi va har biri uchun matn yozadi.
# Model: Hugging Face'dagi Toxirerkinovtohir/whisper-small-karakalpak (= checkpoint-5000).
#   Birinchi ishga tushirishda avtomatik yuklanadi (~970 MB), keyin keshdan olinadi.
#   GPU bo'lsa GPU'da, bo'lmasa CPU'da ishlaydi.
#
# Ishlatish:
#   python transcribe.py audio.wav                       -> bitta fayl
#   python transcribe.py audio1.flac audio2.mp3          -> bir nechta fayl
#   python transcribe.py papka/                          -> papkadagi barcha audio
#   python transcribe.py papka/ --csv natija.csv         -> natijani CSV ga ham yozadi
#   python transcribe.py audio.wav --greedy              -> tezroq, lekin biroz yomonroq
#   python transcribe.py audio.wav --model yol/modelga   -> lokal yoki boshqa HF model
#
# Dekodlash (07_eval tajribasida validation'da tanlangan): beam 5, no_repeat yo'q, til tokeni "uz", float32.
# 30 soniyadan uzun audio bo'laklarga bo'linadi: kesish joyi 20-30 soniya oralig'idagi eng jim nuqta
#   (pauza) tanlanadi, shunda so'z o'rtasidan kesilmaydi. Bo'laklar matni ketma-ket ulanadi.
# Halqa zaxirasi: model bir so'zni qayta-qayta yozsa (halqa), o'sha bo'lak
#   no_repeat_ngram_size=3 bilan qayta o'qiladi. Halqalar validation'dagi so'z xatolarining 11.7% ini bergan
#   (07_eval/xatolar_tahlili.md); zaxira faqat halqa aniqlangan bo'lakka tegadi.
import argparse
import csv
import os
import time

import numpy as np

MODEL = os.environ.get("KAA_ASR_MODEL", "Toxirerkinovtohir/whisper-small-karakalpak")
SR = 16000                         # Whisper 16 kHz audio kutadi
BOLAK = 30 * SR                    # Whisper bir martada ko'pi bilan 30 soniya eshitadi
FORMATLAR = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def bolaklarga_bol(wav):
    """Uzun audio -> 30 soniyadan oshmaydigan bo'laklar.
    Har bir kesish joyi: bo'lak boshidan 20-30 soniya oralig'idagi eng jim nuqta (gaplar orasidagi pauza)."""
    import librosa

    # Har 10 ms dagi ovoz balandligi (RMS); 0.2 s bo'yicha o'rtacha -> qisqa shovqinlarga aldanmaslik uchun
    rms = librosa.feature.rms(y=wav, frame_length=400, hop_length=160)[0]
    rms = np.convolve(rms, np.ones(20) / 20, mode="same")
    bolaklar, bosh = [], 0
    while len(wav) - bosh > BOLAK:
        a, b = (bosh + 20 * SR) // 160, (bosh + BOLAK) // 160           # qidiriladigan oraliq (kadrlarda)
        kesish = (a + int(np.argmin(rms[a:b]))) * 160                    # eng jim kadr -> namunalarda
        bolaklar.append(wav[bosh:kesish])
        bosh = kesish
    bolaklar.append(wav[bosh:])
    return bolaklar


def halqa_mi(matn, soniya):
    """Model "halqa"ga tushdimi? (haqiqiy matnsiz aniqlash)
    - soniyasiga 4.5 so'zdan ko'p -> odam bunchalik tez gapirmaydi
    - bitta so'z 4 marta ketma-ket -> takrorlanish"""
    s = matn.split()
    if len(s) > 4.5 * max(soniya, 1.0):
        return True
    ketma = 1
    for i in range(1, len(s)):
        ketma = ketma + 1 if s[i] == s[i - 1] else 1
        if ketma >= 4:
            return True
    return False


class Transkriptor:
    def __init__(self, model=MODEL, beam=True):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging

        logging.set_verbosity_error()      # ortiqcha ogohlantirishlarni o'chiradi
        self._torch = torch
        # GPU (cuda) bormi? Bo'lsa GPU, bo'lmasa CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model)   # float32 (standart)
        self.model = self.model.to(self.device).eval()
        self.beam = beam

    def _bolakni_oqi(self, wav, no_repeat=0):
        """30 soniyagacha bo'lgan bitta bo'lak -> matn"""
        x = self.processor(wav, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
        sozlamalar = {"num_beams": 5} if self.beam else {"num_beams": 1}
        if no_repeat:
            sozlamalar["no_repeat_ngram_size"] = no_repeat
        with self._torch.no_grad():
            out = self.model.generate(x, language="uz", task="transcribe", max_new_tokens=440,
                                      do_sample=False, **sozlamalar)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def massiv(self, wav):
        """16 kHz mono numpy massiv -> matn. Uzun audio pauzalardan bo'laklarga bo'linadi."""
        wav = np.asarray(wav, dtype=np.float32)
        qismlar = []
        for bolak in bolaklarga_bol(wav):
            if len(bolak) < 0.3 * SR:                        # 0.3 soniyadan qisqa qoldiq -> tashlab yuboramiz
                continue
            matn = self._bolakni_oqi(bolak)
            if halqa_mi(matn, len(bolak) / SR):              # halqa -> takrorni taqiqlab qayta o'qish
                matn = self._bolakni_oqi(bolak, no_repeat=3)
            qismlar.append(matn)
        return " ".join(q for q in qismlar if q)

    def fayl(self, yol):
        """Audio fayl -> (matn, davomiylik soniyada). Istalgan format va sample rate 16 kHz ga o'tkaziladi."""
        import librosa

        wav, _ = librosa.load(yol, sr=SR, mono=True)
        return self.massiv(wav), len(wav) / SR


def audio_fayllar(kirishlar):
    """Fayl va papkalar ro'yxati -> audio fayllar ro'yxati"""
    natija = []
    for k in kirishlar:
        if os.path.isdir(k):
            for ildiz, _, fayllar in os.walk(k):
                natija += [os.path.join(ildiz, f) for f in sorted(fayllar) if f.lower().endswith(FORMATLAR)]
        elif os.path.isfile(k):
            natija.append(k)
        else:
            print("Topilmadi:", k)
    return natija


def main():
    p = argparse.ArgumentParser(description="Qoraqalpoqcha nutqni matnga aylantirish (Whisper-small fine-tune)")
    p.add_argument("kirish", nargs="+", help="audio fayl(lar) yoki papka(lar)")
    p.add_argument("--csv", help="natijani shu CSV faylga ham yozish")
    p.add_argument("--greedy", action="store_true", help="beam o'rniga greedy (tezroq)")
    p.add_argument("--model", default=MODEL, help=f"HF model nomi yoki lokal papka (standart: {MODEL})")
    a = p.parse_args()

    fayllar = audio_fayllar(a.kirish)
    if not fayllar:
        print("Audio fayl topilmadi.")
        return

    t = Transkriptor(model=a.model, beam=not a.greedy)
    print(f"Model: {a.model} | {t.device} | {'greedy' if a.greedy else 'beam 5'} | {len(fayllar)} ta fayl\n")

    qatorlar = []
    for yol in fayllar:
        t0 = time.time()
        matn, soniya = t.fayl(yol)
        print(f"{yol}  ({soniya:.1f} s audio, {time.time() - t0:.1f} s ishladi)\n  -> {matn}\n")
        qatorlar.append({"fayl": yol, "davomiylik_s": round(soniya, 2), "matn": matn})

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["fayl", "davomiylik_s", "matn"])
            w.writeheader()
            w.writerows(qatorlar)
        print("CSV yozildi:", a.csv)


if __name__ == "__main__":
    main()

"""Video uchun ovoz: kod bilan sintez qilingan fon musiqasi + animatsiyaga mos effektlar + demo'dagi haqiqiy nutq.

Hamma vaqtlar scripts/make_video.py dagi sahnalar jadvalidan (S) olinadi, shuning uchun ovoz kadrga aniq tushadi.
Musiqa va effektlar noldan sintez qilingan (namuna fayl ishlatilmagan) - mualliflik huquqi muammosi yo'q.

Ishga tushirish (repo ildizidan, ffmpeg kerak):
    python scripts/make_audio.py              # assets/video_4x5.mp4 ga ovoz qo'yiladi (video oqimi o'zgarmaydi)
    python scripts/make_audio.py 16x9         # assets/video_16x9.mp4 ga
"""
import math
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_video as V  # noqa: E402  (sahnalar jadvali, audio davomiyligi)

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
DUR = V.DUR
N = int(DUR * SR) + SR
rng = np.random.default_rng(11)

BPM = 100
BEAT = 60 / BPM
BAR = 4 * BEAT
# Am - F - C - G (har biri bitta takt)
AKKORDLAR = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]   # MIDI
BASS = [45, 41, 48, 43]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def t_arr(n):
    return np.arange(n) / SR


def env_exp(n, tau):
    return np.exp(-t_arr(n) / tau)


def joyla(buf, t, x, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(x) <= 0:
        return
    a, b = max(i, 0), min(i + len(x), len(buf))
    buf[a:b] += gain * x[a - i:b - i]


def lowpass(x, fc, order=2):
    sos = butter(order, min(fc, SR / 2 - 100) / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, x)


def highpass(x, fc, order=2):
    sos = butter(order, fc / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, x)


# ------------------------------------------------------------------ musiqa
def pad():
    """Yumshoq pad: har takt akkordi, sekin attack, ikki kanal ozgina farqli (kenglik)."""
    L, Rr = np.zeros(N), np.zeros(N)
    n = int(BAR * SR)
    tt = t_arr(n)
    for k in range(int(DUR / BAR) + 2):
        notes = AKKORDLAR[k % 4]
        x = np.zeros(n)
        for m in notes:
            f = hz(m)
            for det in (-0.12, 0.12):                  # ozgina detune -> iliq pad
                x += np.sin(2 * np.pi * f * (1 + det / 100) * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)
        att = np.minimum(1, tt / 0.6) * np.minimum(1, (BAR - tt) / 0.4 + 0.15)
        x *= att / 8
        joyla(L, k * BAR, x)
        joyla(Rr, k * BAR + 0.011, x)
    return lowpass(L, 2200), lowpass(Rr, 2200)


def bass():
    b = np.zeros(N)
    n = int(BEAT / 2 * SR)
    for k in range(int(DUR / (BEAT / 2)) + 1):
        t = k * BEAT / 2
        m = BASS[int(t / BAR) % 4]
        tt = t_arr(n)
        x = np.sin(2 * np.pi * hz(m) * tt) + 0.25 * np.sin(2 * np.pi * 2 * hz(m) * tt)
        x *= np.minimum(1, tt / 0.01) * env_exp(n, 0.18)
        joyla(b, t, x, 0.35 if k % 2 == 0 else 0.22)
    return lowpass(b, 600)


def kick_hit():
    n = int(0.35 * SR)
    tt = t_arr(n)
    f = 45 + 85 * np.exp(-tt / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 0.12)


def hat_hit(open_=False):
    n = int((0.12 if open_ else 0.045) * SR)
    return highpass(rng.standard_normal(n), 7000) * env_exp(n, 0.04 if open_ else 0.012)


def barabanlar():
    k, h = np.zeros(N), np.zeros(N)
    kick = kick_hit()
    for i in range(int(DUR / BEAT) + 1):
        joyla(k, i * BEAT, kick, 0.9)
        joyla(h, i * BEAT + BEAT / 2, hat_hit(i % 4 == 3), 0.25)
        joyla(h, i * BEAT, hat_hit(), 0.12)
    return k, h


def arp():
    a = np.zeros(N)
    n = int(0.4 * SR)
    tt = t_arr(n)
    step = BEAT / 2
    for k in range(int(DUR / step) + 1):
        t = k * step
        notes = AKKORDLAR[int(t / BAR) % 4]
        m = notes[[0, 1, 2, 1][k % 4]] + 12
        x = (np.sin(2 * np.pi * hz(m) * tt) + 0.4 * np.sin(2 * np.pi * 2 * hz(m) * tt)) * env_exp(n, 0.11)
        joyla(a, t, x, 0.16)
    return lowpass(a, 4000)


# ------------------------------------------------------------------ effektlar
def whoosh(dur=0.75, yuqoriga=True):
    """Shovqin, markazi vaqt bo'ylab siljiydigan polosali filtr orqali (STFT maskasi)."""
    n = int(dur * SR)
    x = rng.standard_normal(n)
    S = librosa.stft(x, n_fft=1024, hop_length=256)
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024)[:, None]
    p = np.linspace(0, 1, S.shape[1])[None, :]
    mark = (300 * (14 ** p)) if yuqoriga else (4500 * (14 ** -p))
    S *= np.exp(-((np.log(fr + 1) - np.log(mark)) ** 2) / 0.35)
    y = librosa.istft(S, hop_length=256, length=n)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.6
    return y * env / (np.abs(y).max() + 1e-9)


def pop(f=900, dur=0.09):
    n = int(dur * SR)
    tt = t_arr(n)
    fr = f * (1 + 0.6 * (1 - np.exp(-tt / 0.02)))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * env_exp(n, 0.03) * np.minimum(1, tt / 0.002)


def qongiroq(f=1320, dur=1.2):
    n = int(dur * SR)
    tt = t_arr(n)
    x = sum(a * np.sin(2 * np.pi * f * r * tt) * env_exp(n, dur / (1 + i))
            for i, (r, a) in enumerate([(1, 1), (2.01, .45), (2.76, .3), (5.4, .12)]))
    return x * np.minimum(1, tt / 0.003) / 1.9


def glitch(dur=0.06):
    n = int(dur * SR)
    x = np.sign(np.sin(2 * np.pi * rng.uniform(150, 900) * t_arr(n))) * 0.6 + 0.4 * rng.standard_normal(n)
    x = np.round(x * 4) / 4                                   # bitcrush
    return x * env_exp(n, dur / 2.5)


def tushish(f0=1100, f1=180, dur=0.35):
    n = int(dur * SR)
    fr = f0 * (f1 / f0) ** np.linspace(0, 1, n)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * env_exp(n, dur / 2) * 0.8


def riser(dur=1.8):
    n = int(dur * SR)
    tt = t_arr(n)
    fr = 200 * (12 ** (tt / dur))
    ton = np.sin(2 * np.pi * np.cumsum(fr) / SR) * 0.35
    nz = highpass(rng.standard_normal(n), 2500) * 0.25
    return (ton + nz) * (tt / dur) ** 2


def tada():
    """Konfetti: akkord zarbasi + yuqori uchqunlar."""
    n = int(1.6 * SR)
    tt = t_arr(n)
    x = np.zeros(n)
    for m in (69, 72, 76, 81):                                 # A minor -> yorqin
        x += np.sin(2 * np.pi * hz(m) * tt) * env_exp(n, 0.7) + 0.3 * np.sin(2 * np.pi * 2 * hz(m) * tt) * env_exp(n, 0.3)
    x /= 4
    for _ in range(26):
        joyla(x, rng.uniform(0.02, 1.2), qongiroq(rng.uniform(2200, 5200), 0.25), rng.uniform(0.04, 0.12))
    return x


def oyna_kirish(t0, t1, ichi=0.6):
    """Silliq kuchayish/so'nish envelope'i (gain egri chizig'i)."""
    tt = t_arr(N)
    return np.clip(np.minimum((tt - t0) / ichi, (t1 - tt) / ichi), 0, 1)


# ------------------------------------------------------------------ yig'ish
def yasash():
    S = {k: (v[0], v[1]) for k, v in V.S.items()}
    tt = t_arr(N)
    musiqa_L, musiqa_R = pad()
    b = bass()
    k, h = barabanlar()
    a = arp()

    groove = oyna_kirish(S["malumot"][0] - 0.4, S["demo"][0] + 0.3, 0.8)          # baraban/bass/arp shu oraliqda
    yakun_groove = oyna_kirish(S["yakun"][0] + 0.2, DUR - 1.0, 0.8) * 0.7
    tension = oyna_kirish(S["muammo"][0], S["muammo"][1], 0.5)
    g = np.maximum(groove, yakun_groove)
    pulse = np.zeros(N)                                                             # muammo: past, xavotirli puls
    for i in range(int((S["muammo"][1] - S["muammo"][0]) / (BEAT / 2))):
        joyla(pulse, S["muammo"][0] + i * BEAT / 2, kick_hit() * 0.5)
    pad_gain = 0.55 + 0.25 * g
    L = musiqa_L * pad_gain + (b + a * 0.9 + k * 0.55 + h) * g + pulse * tension * 0.6
    R = musiqa_R * pad_gain + (b + a * 1.1 + k * 0.55 + h * 0.8) * g + pulse * tension * 0.6

    # demo paytida musiqa pastga (ducking), nutq aniq eshitilsin
    nutq_bosh = S["demo"][0] + 0.8
    duck = 1 - 0.82 * oyna_kirish(nutq_bosh - 0.5, nutq_bosh + V.AUDIO_S + 0.4, 0.5)
    yakun_fade = np.clip((DUR - tt) / 2.2, 0, 1)
    intro = np.clip(tt / 1.2, 0, 1)
    L *= duck * yakun_fade * intro
    R *= duck * yakun_fade * intro

    fx = np.zeros(N)
    # sahna o'tishlari: rangli wipe markazida whoosh
    for i, tb in enumerate(V.CHEGARALAR):
        joyla(fx, tb - 0.45, whoosh(0.9, yuqoriga=i % 2 == 0), 0.32)
    # sarlavha: so'zlar chiqishi
    joyla(fx, 0.15, riser(0.6) * 0.5)
    t0 = S["sarlavha"][0]
    for i in range(2):
        joyla(fx, t0 + 0.5 + i * 0.07, pop(700 + 160 * i), 0.22)
    joyla(fx, t0 + 1.2, pop(1050), 0.18)
    # muammo: hisoblagich + glitch
    t0 = S["muammo"][0]
    joyla(fx, t0 + 1.55, riser(1.4) * 0.6)
    for j in range(int(3.0 * 7)):
        t = t0 + 1.6 + j / 7
        if int(t * 7) % 3 == 0:
            joyla(fx, t, glitch(), 0.22)
    joyla(fx, t0 + 3.0, pop(500, 0.12), 0.2)
    # malumot: nuqtalar uchib keladi (yengil chirillash), qizillari to'kiladi
    t0 = S["malumot"][0]
    for j in range(16):
        joyla(fx, t0 + 0.15 + j * 0.1, pop(1500 + 90 * j, 0.05), 0.07)
    for j in range(5):
        joyla(fx, t0 + 2.4 + j * 0.12, tushish(1400 - 120 * j), 0.13)
    for i in range(3):
        joyla(fx, t0 + 1.0 + i * 0.35, pop(880 + 110 * i), 0.15)
    # trening: chiplar va eval nuqtalari, egri chiziq bilan ko'tariluvchi sweep
    t0 = S["trening"][0]
    for i in range(4):
        joyla(fx, t0 + 0.2 + i * 0.12, pop(780 + 120 * i), 0.17)
    joyla(fx, t0 + 0.7, whoosh(4.3, True) * 0.5, 0.25)
    xs = np.linspace(0, 1, 2001)
    ys = xs * xs * (3 - 2 * xs)                                     # make_video.ease_io
    for s_ in V.EV.step:
        tp = t0 + 0.7 + 4.3 * float(np.interp(min(s_ / 5900, 1), ys, xs))   # nuqta chiziq yetib kelgan paytda chiqadi
        joyla(fx, tp, pop(1200, 0.06), 0.1)
    # natija: pasayuvchi hisoblagich, riser, konfetti
    t0 = S["natija"][0]
    joyla(fx, t0 + 0.6, tushish(1600, 300, 2.0) * 0.6, 0.18)
    joyla(fx, t0 + 1.1, riser(1.7), 0.35)
    joyla(fx, t0 + 2.8, tada(), 0.5)
    joyla(fx, t0 + 2.8, kick_hit(), 0.6)
    joyla(fx, t0 + 3.2, pop(990), 0.18)
    # lora: kartalar va g'olib nishoni
    t0 = S["lora"][0]
    joyla(fx, t0 + 0.1, pop(700), 0.15)
    for i in range(2):
        joyla(fx, t0 + 0.6 + i * 0.3, whoosh(0.35, True), 0.18)
    joyla(fx, t0 + 2.0, qongiroq(1568), 0.3)
    # demo: tugagach muvaffaqiyat qo'ng'irog'i
    joyla(fx, nutq_bosh + V.AUDIO_S + 0.2, qongiroq(1760, 1.4), 0.32)
    joyla(fx, nutq_bosh + V.AUDIO_S + 0.32, qongiroq(2349, 1.2), 0.22)
    # yakun: kartalar
    t0 = S["yakun"][0]
    for i in range(3):
        joyla(fx, t0 + 0.5 + i * 0.2, pop(660 + 220 * i), 0.18)
    joyla(fx, DUR - 2.4, qongiroq(880, 2.2), 0.2)

    # haqiqiy nutq (demo)
    nutq, _ = librosa.load(V.AUDIO, sr=SR, mono=True)
    nutq = nutq / (np.abs(nutq).max() + 1e-9) * 0.85
    sp = np.zeros(N)
    joyla(sp, nutq_bosh, nutq)

    L = L + fx + sp
    R = R + fx + sp
    st = np.stack([L, R], axis=1)[: int(DUR * SR)]
    st = np.tanh(st / np.abs(st).max() * 1.6) / np.tanh(1.6) * 0.89    # yumshoq limiter, -1 dBFS
    return st


def main():
    nom = "video_16x9.mp4" if "16x9" in sys.argv[1:] else "video_4x5.mp4"
    st = yasash()
    wav = ROOT / "assets" / "_ovoz.wav"
    sf.write(wav, st.astype(np.float32), SR)
    video = ROOT / "assets" / nom
    tmp = ROOT / "assets" / ("_" + nom)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(tmp)],
                   check=True)
    tmp.replace(video)
    wav.unlink()
    print("tayyor:", video, f"{video.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

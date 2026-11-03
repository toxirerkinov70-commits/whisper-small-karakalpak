"""README uchun demo GIF (va ixtiyoriy ravishda loyiha videosi: MP4, 16:9 va 4:5; videolar repoga kirmaydi).

Barcha raqamlar repodagi natija fayllaridan o'qiladi. "Jonli demo" sahnasidagi ovoz examples/namuna_2.flac,
matn esa modelning shu audio uchun haqiqiy chiqishi (test to'plami, WER 0%).

Ishga tushirish (repo ildizidan, ffmpeg kerak):
    python scripts/make_video.py                 # demo GIF
    python scripts/make_video.py 16x9 | 4x5      # video (assets/*.mp4, .gitignore'da)
    python scripts/make_video.py kadr            # har sahnadan bitta tekshiruv kadri (assets/_kadr_*.png)
Shriftlar: Segoe UI (Windows); topilmasa DejaVu Sans.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
FPS = 30
AUDIO = ROOT / "examples/namuna_2.flac"
MATN = ("ilajda qaraqalpaqstan respublikası salıq basqarmasınıń baslıǵı xalmurza bekchanov sawda sanaat "
        "palatasınıń baslıǵı kazbek artikov taraw qánigeleri isbilermenler hám xalıq wákilleri qatnastı")

# ---------- neon palitra ----------
NAVY = (8, 10, 28)
VIOLET, PINK, ORANGE, AMBER = (139, 92, 246), (236, 72, 153), (249, 115, 22), (251, 191, 36)
CYAN, BLUE, EMERALD, RED = (34, 211, 238), (59, 130, 246), (16, 185, 129), (244, 63, 94)
WHITE, SOFT, DIM = (255, 255, 255), (226, 232, 255), (160, 170, 210)
NEON = [VIOLET, PINK, ORANGE, AMBER, CYAN, BLUE, EMERALD]


# ---------- shriftlar ----------
_FONT_CACHE = {}


def _font(names, size):
    size = max(1, size)                 # "pop" animatsiyasi boshida o'lcham 0 ga tushishi mumkin
    key = (names[0], size)
    if key not in _FONT_CACHE:
        for n in names:
            for d in ("C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/", ""):
                try:
                    _FONT_CACHE[key] = ImageFont.truetype(d + n, size)
                    break
                except OSError:
                    continue
            if key in _FONT_CACHE:
                break
        else:
            _FONT_CACHE[key] = ImageFont.load_default(size)
    return _FONT_CACHE[key]


def BLK(s): return _font(["seguibl.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"], int(s))
def B(s): return _font(["segoeuib.ttf", "DejaVuSans-Bold.ttf"], int(s))
def R(s): return _font(["segoeui.ttf", "DejaVuSans.ttf"], int(s))
def SB(s): return _font(["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"], int(s))
def MONO(s): return _font(["consolab.ttf", "consola.ttf", "DejaVuSansMono.ttf"], int(s))


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x): x = clamp(x); return 1 - (1 - x) ** 3
def ease_io(x): x = clamp(x); return x * x * (3 - 2 * x)


def back(x):
    """Biroz o'tib ketib qaytadigan (pop) easing."""
    x = clamp(x)
    c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def mix(c1, c2, t): return tuple(int(a + (b - a) * clamp(t)) for a, b in zip(c1, c2))


def grad_colors(stops, n):
    """Ko'p nuqtali gradient: n ta rang."""
    xs = np.linspace(0, 1, n)
    seg = len(stops) - 1
    out = []
    for x in xs:
        i = min(int(x * seg), seg - 1)
        out.append(mix(stops[i], stops[i + 1], x * seg - i))
    return np.array(out, dtype=np.uint8)


# ---------- ma'lumotlar ----------
yak = pd.read_csv(ROOT / "07_eval/yakuniy_jadval.csv").set_index(["model", "dekodlash"])
BASE_WER = yak.loc[("baseline_kk_lotin", "greedy"), "test_WER"]
BASE_CER = yak.loc[("baseline_kk_lotin", "greedy"), "test_CER"]
FT_WER = yak.loc[("finetune", "beam5"), "test_WER"]
FT_CER = yak.loc[("finetune", "beam5"), "test_CER"]
tarix = pd.read_csv(ROOT / "06_finetune/results/train_history.csv")
TR = tarix[tarix.turi == "train"]
EV = tarix[tarix.turi == "eval"]
summary = pd.read_csv(ROOT / "01_EDA/dataset_summary.csv", encoding="utf-8-sig")
ASL_KLIP, ASL_SOAT = int(summary["count"].sum()), float(summary["hours"].sum())
man = pd.read_csv(ROOT / "04_normalize/manifest_tayyor.csv", usecols=["split", "yangi_s"])
YAK_KLIP, YAK_SOAT = len(man), man.yangi_s.sum() / 3600
SPLIT = man.groupby("split").size()
LORA = json.load(open(ROOT / "08_lora/lora_natija.json", encoding="utf-8"))
FULL = json.load(open(ROOT / "08_lora/full_natija.json", encoding="utf-8"))

WAV, _ = librosa.load(AUDIO, sr=16000)
AUDIO_S = len(WAV) / 16000
RMS = librosa.feature.rms(y=WAV, frame_length=1024, hop_length=160)[0]
RMS = RMS / RMS.max()


def rms_at(t):
    i = int(clamp(t / AUDIO_S) * (len(RMS) - 1))
    return float(RMS[i])


def nom_son(x, nuqta=0):
    return f"{x:,.{nuqta}f}".replace(",", " ")


# ---------- sahnalar jadvali ----------
DEMO_LEN = AUDIO_S + 2.6
SAHNALAR = [
    ("sarlavha", 4.6, [VIOLET, CYAN, PINK]),
    ("muammo", 5.2, [ORANGE, RED, PINK]),
    ("malumot", 6.4, [CYAN, EMERALD, BLUE]),
    ("trening", 6.2, [BLUE, VIOLET, CYAN]),
    ("natija", 6.4, [EMERALD, CYAN, AMBER]),
    ("lora", 5.4, [VIOLET, PINK, BLUE]),
    ("demo", DEMO_LEN, [PINK, VIOLET, CYAN]),
    ("yakun", 5.6, [VIOLET, ORANGE, CYAN]),
]
S, _t = {}, 0.0
for nom, dur, ranglar in SAHNALAR:
    S[nom] = (_t, _t + dur, ranglar)
    _t += dur
DUR = _t
CHEGARALAR = [S[n][0] for n, _, _ in SAHNALAR[1:]]
WIPE = 0.36            # o'tish yarim davomiyligi


class Kadr:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.k = min(W, H) / 1080
        self.wide = W > H
        # fon gradienti past o'lchamda hisoblanadi, keyin kattalashtiriladi (silliq bo'lgani uchun sifat yo'qolmaydi)
        self.lw, self.lh = W // 6, H // 6
        yy, xx = np.mgrid[0:self.lh, 0:self.lw]
        self.gx, self.gy = xx / self.lw, yy / self.lh * (H / W)
        rng = np.random.default_rng(7)
        self.zarra = rng.random((80, 6))          # x, y, tezlik, o'lcham, rang, faza
        self.konf = rng.random((150, 6))
        self.dots = rng.random((440, 3))
        self.yulduz = rng.random((60, 3))

    def s(self, v):
        return int(round(v * self.k))

    # ---------- fon ----------
    def fon(self, t):
        ranglar = self._fon_ranglari(t)
        arr = np.zeros((self.lh, self.lw, 3)) + np.array(NAVY, float)
        for j, c in enumerate(ranglar):
            cx = 0.5 + 0.42 * math.sin(t * 0.35 + j * 2.1)
            cy = (0.5 + 0.38 * math.cos(t * 0.28 + j * 1.7)) * (self.H / self.W)
            r = 0.32 + 0.06 * math.sin(t * 0.5 + j)
            m = np.exp(-((self.gx - cx) ** 2 + (self.gy - cy) ** 2) / (r * r))[..., None]
            arr += np.array(c, float) * 0.62 * m
        arr = 255 * (1 - np.exp(-arr / 255 * 1.25))          # yumshoq tonemap
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).resize((self.W, self.H), Image.BILINEAR)
        return img.convert("RGBA")

    def _fon_ranglari(self, t):
        for nom, (t0, t1, r) in S.items():
            if t0 <= t < t1:
                # keyingi sahna ranglariga o'tish
                keyingi = [S[n][2] for n, _, _ in SAHNALAR if S[n][0] == t1]
                if keyingi and t > t1 - WIPE:
                    p = (t - (t1 - WIPE)) / WIPE
                    return [mix(a, b, p) for a, b in zip(r, keyingi[0])]
                return r
        return S["yakun"][2]

    def zarrachalar(self, img, t, a=1.0):
        lay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for x0, y0, v, sz, ci, ph in self.zarra:
            x = (x0 + 0.01 * math.sin(t * 0.7 + ph * 6)) * self.W
            y = ((y0 - t * (0.02 + 0.05 * v)) % 1.0) * self.H
            r = self.s(2 + 5 * sz)
            tw = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 2.5 + ph * 9))
            c = NEON[int(ci * len(NEON)) % len(NEON)]
            d.ellipse([x - r * 2.4, y - r * 2.4, x + r * 2.4, y + r * 2.4], fill=c + (int(40 * tw * a),))
            d.ellipse([x - r, y - r, x + r, y + r], fill=mix(c, WHITE, 0.5) + (int(200 * tw * a),))
        img.alpha_composite(lay)

    # ---------- matn ----------
    def matn(self, img, xy, s, font, rang=WHITE, a=1.0, ha="m", va="m", glow=0, grad=None, glitch=0.0):
        """Matnni alohida qatlamda chizadi: gradient, nur (glow) va glitch effektlari bilan."""
        if a <= 0.01 or not s:
            return 0
        l, _, r, _ = font.getbbox(s)
        _, tp, _, b = font.getbbox("ÁÓgjy")        # vertikal: shriftning umumiy balandligi (so'zlar bir chiziqda)
        pad = self.s(30) if glow else self.s(14)
        w, h = r - l + 2 * pad, b - tp + 2 * pad
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text((pad - l, pad - tp), s, font=font, fill=255)
        if grad:
            grad = [mix(c, WHITE, 0.22) for c in grad]      # gradient yozuv fondan yorqinroq bo'lsin
            g = grad_colors(grad, w)
            fill = Image.fromarray(np.repeat(g[None], h, axis=0)).convert("RGBA")
        else:
            fill = Image.new("RGBA", (w, h), rang + (255,))
        x = xy[0] - {"l": 0, "m": (r - l) / 2, "r": r - l}[ha] - pad
        y = xy[1] - {"t": 0, "m": (b - tp) / 2, "b": b - tp}[va] - pad
        x, y = int(x), int(y)
        # yumshoq soya: rangli fonda ham o'qilishi uchun
        sm = mask.filter(ImageFilter.GaussianBlur(max(2, self.s(5))))
        sh = Image.new("RGBA", (w, h), NAVY + (0,))
        sh.putalpha(sm.point(lambda v: int(v * 0.75 * a)))
        oy = y + max(1, self.s(3))
        img.alpha_composite(sh, dest=(max(x, 0), max(oy, 0)), source=(max(-x, 0), max(-oy, 0)))
        if glow:
            gm = mask.filter(ImageFilter.GaussianBlur(self.s(glow)))
            gc = (grad[len(grad) // 2] if grad else rang)
            gl = Image.new("RGBA", (w, h), gc + (0,))
            gl.putalpha(gm.point(lambda v: int(v * 0.85 * a)))
            img.alpha_composite(gl, dest=(max(x, 0), max(y, 0)), source=(max(-x, 0), max(-y, 0)))
        if glitch > 0:
            for dx, c in ((-1, (255, 40, 90)), (1, (40, 230, 255))):
                gl = Image.new("RGBA", (w, h), c + (0,))
                gl.putalpha(mask.point(lambda v: int(v * 0.7 * a)))
                ox = int(x + dx * glitch * self.s(14))
                img.alpha_composite(gl, dest=(max(ox, 0), max(y, 0)), source=(max(-ox, 0), max(-y, 0)))
        fill.putalpha(mask.point(lambda v: int(v * a)))
        img.alpha_composite(fill, dest=(max(x, 0), max(y, 0)), source=(max(-x, 0), max(-y, 0)))
        return r - l

    def sozlar(self, img, t, xy, text, font, rang=WHITE, grad=None, stagger=0.07, dur=0.5, ha="m", glow=0):
        """Kinetik tipografiya: so'zlar ketma-ket pastdan sakrab chiqadi."""
        words = text.split()
        sp = font.getlength(" ")
        widths = [font.getlength(w) for w in words]
        jami = sum(widths) + sp * (len(words) - 1)
        x = xy[0] - {"l": 0, "m": jami / 2, "r": jami}[ha]
        for i, (w, ww) in enumerate(zip(words, widths)):
            p = clamp((t - i * stagger) / dur)
            if p > 0:
                dy = self.s(46) * (1 - back(p))
                g = None
                if grad:   # gradient butun qator bo'ylab davom etadi
                    f0, f1 = (x - (xy[0] - jami / 2)) / jami, (x + ww - (xy[0] - jami / 2)) / jami
                    g = [tuple(int(v) for v in grad_colors(grad, 101)[int(clamp(f) * 100)]) for f in (f0, (f0 + f1) / 2, f1)]
                self.matn(img, (x, xy[1] + dy), w, font, rang, a=ease(p * 1.4), ha="l", va="m", grad=g, glow=glow)
            x += ww + sp

    def shisha(self, img, box, a=1.0, rad=24, chegara=WHITE, kuch=0.10):
        """Shisha (glassmorphism) kartochka."""
        x0, y0, x1, y1 = [int(v) for v in box]
        lay = Image.new("RGBA", (x1 - x0 + 4, y1 - y0 + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=self.s(rad), fill=(255, 255, 255, int(255 * kuch * a)),
                            outline=chegara + (int(110 * a),), width=self.s(2))
        img.alpha_composite(lay, dest=(x0, y0))

    def pill(self, img, xy, text, font, rang, a=1.0, ha="m"):
        w = font.getlength(text) + self.s(44)
        h = font.size * 1.6
        x = xy[0] - {"l": 0, "m": w / 2, "r": w}[ha]
        self.shisha(img, (x, xy[1] - h / 2, x + w, xy[1] + h / 2), a=a, rad=int(h / 2 / self.k), chegara=rang,
                    kuch=0.12)
        self.matn(img, (x + w / 2, xy[1]), text, font, WHITE, a=a)

    def neon_chiziq(self, img, pts, stops, width, a=1.0, glow=10):
        """Rangi chiziq bo'ylab o'zgaradigan, nurlanuvchi chiziq."""
        if len(pts) < 2:
            return
        lay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        cols = grad_colors(stops, len(pts))
        for i in range(len(pts) - 1):
            d.line([pts[i], pts[i + 1]], fill=tuple(int(v) for v in cols[i]) + (int(255 * a),), width=width)
        if glow:
            img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(self.s(glow))))
            img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(self.s(glow / 3))))
        img.alpha_composite(lay)

    def mic(self, img, cx, cy, r, a=1.0):
        lay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        w, h = r * 0.42, r * 0.7
        c = WHITE + (int(255 * a),)
        d.rounded_rectangle([cx - w / 2, cy - h * 0.75, cx + w / 2, cy + h * 0.25], radius=int(w / 2), fill=c)
        d.arc([cx - w * 0.85, cy - h * 0.55, cx + w * 0.85, cy + h * 0.5], 20, 160, fill=c, width=self.s(6))
        d.line([(cx, cy + h * 0.5), (cx, cy + h * 0.75)], fill=c, width=self.s(6))
        d.line([(cx - w * 0.45, cy + h * 0.75), (cx + w * 0.45, cy + h * 0.75)], fill=c, width=self.s(6))
        img.alpha_composite(lay)

    # =====================================================================================
    # SAHNALAR. Har biri (img, t) oladi: t - sahna boshidan beri o'tgan vaqt.
    # =====================================================================================
    def sarlavha(self, img, t):
        W, H = self.W, self.H
        cy = H * 0.5
        amp = self.s(120) * ease(t / 1.0)
        for j, (stops, f1, sp, ph) in enumerate([([VIOLET, PINK, ORANGE], 2.6, 1.9, 0),
                                                 ([CYAN, BLUE, VIOLET], 3.8, -1.4, 1.3),
                                                 ([EMERALD, CYAN, PINK], 1.9, 1.1, 2.4)]):
            pts = []
            for i in range(161):
                x = i / 160
                env = math.sin(math.pi * x) ** 1.5
                y = env * (math.sin(2 * math.pi * f1 * x + t * sp + ph) * 0.65
                           + math.sin(2 * math.pi * f1 * 2.3 * x - t * sp * 1.4) * 0.35)
                pts.append((x * W, cy - self.s(40) + y * amp * (1 - 0.22 * j)))
            self.neon_chiziq(img, pts, stops, self.s(5 - j), a=0.95 - 0.15 * j, glow=14)
        self.pill(img, (W / 2, cy - self.s(300)), "SPEECH-TO-TEXT  ·  FINE-TUNING", B(self.s(26)), CYAN,
                  a=ease((t - 0.2) / 0.5))
        if self.wide:
            self.sozlar(img, t - 0.5, (W / 2, cy + self.s(190)), "Whisper qoraqalpoqcha", BLK(self.s(104)),
                        grad=[CYAN, VIOLET, PINK, ORANGE], glow=18)
            self.sozlar(img, t - 1.2, (W / 2, cy + self.s(300)), "tinglashni o'rgandi", SB(self.s(62)), SOFT)
        else:                                    # tor ekranda sarlavha ikki qatorga bo'linadi
            self.sozlar(img, t - 0.5, (W / 2, cy + self.s(170)), "Whisper", BLK(self.s(104)),
                        grad=[CYAN, VIOLET], glow=18)
            self.sozlar(img, t - 0.7, (W / 2, cy + self.s(285)), "qoraqalpoqcha", BLK(self.s(104)),
                        grad=[VIOLET, PINK, ORANGE], glow=18)
            self.sozlar(img, t - 1.3, (W / 2, cy + self.s(395)), "tinglashni o'rgandi", SB(self.s(58)), SOFT)

    def muammo(self, img, t):
        W, H = self.W, self.H
        cy = H * 0.5
        self.sozlar(img, t - 0.1, (W / 2, cy - self.s(300)), "Whisper ~100 tilni tushunadi.", SB(self.s(58)), SOFT)
        self.sozlar(img, t - 0.8, (W / 2, cy - self.s(215)), "Qoraqalpoq tili ular orasida yo'q.",
                    B(self.s(58)), grad=[AMBER, ORANGE, RED])
        p = ease((t - 1.6) / 1.4)
        v = BASE_WER * p
        gl = 0.0
        if 1.6 < t < 4.6:
            gl = (0.6 + 0.4 * math.sin(t * 37)) * (1.0 if int(t * 7) % 3 == 0 else 0.25)
        self.matn(img, (W / 2, cy + self.s(40)), f"{v:.1f}%", BLK(self.s(230)), a=ease((t - 1.5) * 3),
                  grad=[AMBER, ORANGE, RED, PINK], glow=26, glitch=gl)
        self.pill(img, (W / 2, cy + self.s(220)), "fine-tuning'siz Whisper-small · test WER", R(self.s(30)), ORANGE,
                  a=ease((t - 2.2) * 2))
        self.sozlar(img, t - 3.0, (W / 2, cy + self.s(310)), "deyarli har bir so'z xato", SB(self.s(40)),
                    mix(SOFT, PINK, 0.3))

    def malumot(self, img, t):
        W, H = self.W, self.H
        # setka: har nuqta = 50 klip
        n = int(round(ASL_KLIP / 50))
        olib = int(round((ASL_KLIP - YAK_KLIP) / 50))
        cols = 31 if self.wide else 31
        rows = math.ceil(n / cols)
        cell = self.s(24 if self.wide else 28)
        if self.wide:
            gx0, gy0 = W * 0.07, H * 0.5 - rows * cell / 2 + self.s(10)
            tx = W * 0.5
        else:
            gx0, gy0 = W / 2 - cols * cell / 2, H * 0.1
            tx = W * 0.08
        olib_idx = set(np.argsort(self.dots[:n, 2])[:olib])
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        stops = grad_colors([CYAN, BLUE, EMERALD], n)
        for i in range(n):
            r_, c_ = divmod(i, cols)
            tx_, ty_ = gx0 + c_ * cell, gy0 + r_ * cell
            p = ease((t - 0.1 - i * 0.0022) / 0.7)
            sx, sy = self.dots[i, 0] * W, self.dots[i, 1] * H
            x, y = sx + (tx_ - sx) * p, sy + (ty_ - sy) * p
            col = tuple(int(v) for v in stops[i])
            al = 1.0
            if i in olib_idx:
                q = ease((t - 2.4 - (i % 7) * 0.05) / 1.0)
                col = mix(col, RED, min(1, q * 2))
                y += q * q * self.s(420)
                al = 1 - q
            rr = cell * 0.36
            if al > 0.02 and p > 0:
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col + (int(255 * al * min(1, p * 2)),))
        img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(self.s(4))))
        img.alpha_composite(lay)
        if self.wide:
            ty = H * 0.22
        else:
            ty = gy0 + rows * cell + self.s(70)
        self.matn(img, (tx, ty), "1 · MA'LUMOT", B(self.s(30)), CYAN, a=ease(t * 2), ha="l", glow=8)
        q = ease((t - 2.4) / 1.6)
        klip = ASL_KLIP + (YAK_KLIP - ASL_KLIP) * q
        soat = ASL_SOAT + (YAK_SOAT - ASL_SOAT) * q
        self.matn(img, (tx, ty + self.s(95)), nom_son(klip), BLK(self.s(110)), a=ease((t - 0.3) * 2), ha="l",
                  grad=[CYAN, EMERALD], glow=14)
        self.matn(img, (tx, ty + self.s(175)), "audio klip  ·  1 nuqta = 50 klip", R(self.s(30)), DIM,
                  a=ease((t - 0.5) * 2), ha="l")
        self.matn(img, (tx, ty + self.s(270)), f"{soat:.1f} soat", BLK(self.s(84)), a=ease((t - 0.6) * 2), ha="l",
                  grad=[BLUE, CYAN], glow=12)
        bandlar = ["Silero VAD: jimlik kesildi", "matn normalizatsiyasi", "dublikat va leakage olindi"]
        for i, bnd in enumerate(bandlar):
            aa = ease((t - 1.0 - i * 0.35) * 2.5)
            self.pill(img, (tx, ty + self.s(370 + i * 64)), bnd, R(self.s(27)), [CYAN, EMERALD, BLUE][i], a=aa,
                      ha="l")
        aa = ease((t - 3.6) * 2)
        self.matn(img, (tx, ty + self.s(580)),
                  f"−{nom_son(ASL_KLIP - YAK_KLIP)} klip  →  train {nom_son(SPLIT['train'])} · "
                  f"val {nom_son(SPLIT['validation'])} · test {nom_son(SPLIT['test'])}",
                  SB(self.s(28)), mix(SOFT, PINK, 0.2), a=aa, ha="l")

    def trening(self, img, t):
        W, H = self.W, self.H
        cy = H * 0.5
        self.matn(img, (W / 2, cy - self.s(360)), "2 · FINE-TUNING", B(self.s(30)), mix(BLUE, WHITE, 0.4),
                  a=ease(t * 2), glow=8)
        chips = [("5 epoch", VIOLET), ("5 840 qadam", BLUE), ("2 soat 33 daq", CYAN), ("NVIDIA DGX Spark", PINK)]
        f = SB(self.s(28))
        ws = [f.getlength(c) + self.s(44) for c, _ in chips]
        if self.wide:
            qatorlar = [list(range(4))]
        else:
            qatorlar = [[0, 1], [2, 3]]
        for qi, q in enumerate(qatorlar):
            jami = sum(ws[i] for i in q) + self.s(18) * (len(q) - 1)
            x = W / 2 - jami / 2
            for i in q:
                self.pill(img, (x, cy - self.s(275) + qi * self.s(66)), chips[i][0], f, chips[i][1],
                          a=back(clamp((t - 0.2 - i * 0.12) * 2.5)), ha="l")
                x += ws[i] + self.s(18)
        gx0, gx1 = W * 0.1, W * 0.9
        gy0, gy1 = cy - self.s(150) + (0 if self.wide else self.s(60)), cy + self.s(330)
        self.shisha(img, (gx0 - self.s(30), gy0 - self.s(40), gx1 + self.s(30), gy1 + self.s(40)), a=ease(t * 2),
                    kuch=0.06)
        p = ease_io((t - 0.7) / 4.3)
        maxstep, ymax = 5900, 1.5

        def xy(step, loss):
            return gx0 + (gx1 - gx0) * step / maxstep, gy1 - (gy1 - gy0) * min(loss, ymax) / ymax

        pts = [xy(s_, l_) for s_, l_ in zip(TR.step, TR.loss) if s_ <= p * maxstep]
        if len(pts) > 1:
            # chiziq ostidagi gradient maydon
            poly = pts + [(pts[-1][0], gy1), (pts[0][0], gy1)]
            m = Image.new("L", (W, H), 0)
            ImageDraw.Draw(m).polygon(poly, fill=70)
            area = Image.new("RGBA", (W, H), VIOLET + (0,))
            area.putalpha(m)
            img.alpha_composite(area)
            self.neon_chiziq(img, pts, [PINK, VIOLET, BLUE, CYAN], self.s(5), glow=12)
            ex, ey = pts[-1]
            r = self.s(11)
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(lay).ellipse([ex - r * 2.5, ey - r * 2.5, ex + r * 2.5, ey + r * 2.5], fill=CYAN + (90,))
            ImageDraw.Draw(lay).ellipse([ex - r, ey - r, ex + r, ey + r], fill=WHITE + (255,))
            img.alpha_composite(lay)
            cur = TR[TR.step <= p * maxstep].loss.iloc[-1]
            self.matn(img, (gx1, gy0 - self.s(4)), f"loss {cur:.3f}", MONO(self.s(40)), a=1, ha="r", va="t",
                      grad=[PINK, CYAN], glow=10)
        for s_, l_ in zip(EV.step, EV.loss):
            q = back(clamp((p * maxstep - s_) / 600))
            if q > 0:
                x, y = xy(s_, l_)
                r = self.s(8) * q
                lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ImageDraw.Draw(lay).ellipse([x - r, y - r, x + r, y + r], fill=AMBER + (255,),
                                            outline=NAVY + (255,), width=self.s(3))
                img.alpha_composite(lay)
        la = ease((t - 0.8) * 2)
        x = gx0
        for lab, c in (("train loss", PINK), ("validation loss", AMBER)):
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            r = self.s(8)
            yy = gy0 + self.s(16)
            ImageDraw.Draw(lay).ellipse([x, yy - r, x + 2 * r, yy + r], fill=c + (int(255 * la),))
            img.alpha_composite(lay)
            x += 2 * r + self.s(10)
            x += self.matn(img, (x, yy), lab, R(self.s(26)), SOFT, a=la, ha="l") + self.s(30)

    def natija(self, img, t):
        W, H = self.W, self.H
        cy = H * 0.5
        self.matn(img, (W / 2, cy - self.s(360)), "3 · NATIJA  ·  TEST TO'PLAMI, 1 018 KLIP", B(self.s(30)),
                  mix(EMERALD, WHITE, 0.3), a=ease(t * 2), glow=8)
        p = ease_io((t - 0.6) / 2.0)
        v = BASE_WER + (FT_WER - BASE_WER) * p
        grad = [mix(AMBER, CYAN, p), mix(ORANGE, EMERALD, p), mix(RED, EMERALD, p)]
        self.matn(img, (W / 2, cy - self.s(225)), "WER", SB(self.s(44)), SOFT, a=ease(t * 2))
        self.matn(img, (W / 2, cy - self.s(80)), f"{v:.1f}%", BLK(self.s(240)), grad=grad, glow=30, a=ease(t * 3))
        bw = W * (0.66 if self.wide else 0.8)
        x0 = W / 2 - bw / 2
        yb = cy + self.s(90)
        self.shisha(img, (x0, yb, x0 + bw, yb + self.s(44)), a=ease(t * 2), rad=22, kuch=0.08)
        w = bw * v / 105
        bar = grad_colors(grad, max(int(w), 2))
        bi = Image.fromarray(np.repeat(bar[None], self.s(44), axis=0)).convert("RGBA")
        m = Image.new("L", bi.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, bi.size[0] - 1, bi.size[1] - 1], radius=self.s(22), fill=255)
        bi.putalpha(m)
        img.alpha_composite(bi, dest=(int(x0), int(yb)))
        self.matn(img, (x0, yb + self.s(80)), f"fine-tuning'siz: {BASE_WER:.1f}%", R(self.s(28)), DIM, ha="l",
                  a=ease(t * 2))
        self.matn(img, (x0 + bw, yb + self.s(80)), f"fine-tune: {FT_WER:.1f}%", SB(self.s(28)), SOFT, ha="r",
                  a=ease((t - 2.4) * 2))
        q = back(clamp((t - 2.8) / 0.5))
        if q > 0:
            f = BLK(self.s(54 * q))
            self.matn(img, (W / 2, cy + self.s(270)), f"{BASE_WER / FT_WER:.0f}× kamroq xato", f,
                      grad=[AMBER, PINK, VIOLET], glow=16)
            self.pill(img, (W / 2, cy + self.s(360)), f"CER  {BASE_CER:.1f}%  →  {FT_CER:.1f}%", SB(self.s(30)),
                      EMERALD, a=ease((t - 3.2) * 2))
        self.konfetti(img, t - 2.8, W / 2, cy - self.s(80))

    def konfetti(self, img, t, cx, cy):
        if t <= 0 or t > 3.2:
            return
        lay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for ang, sp, ci, rot, sz, ph in self.konf:
            a = ang * 2 * math.pi
            v = self.s(900 + 900 * sp)
            x = cx + math.cos(a) * v * t
            y = cy + math.sin(a) * v * t * 0.8 + 0.5 * self.s(1500) * t * t
            r = self.s(7 + 9 * sz)
            th = rot * 6 + t * (4 + 8 * ph)
            c = NEON[int(ci * len(NEON)) % len(NEON)]
            al = int(255 * clamp(1 - t / 3.2))
            pts = [(x + r * math.cos(th + k * math.pi / 2) * (1 if k % 2 == 0 else 0.45),
                    y + r * math.sin(th + k * math.pi / 2) * (1 if k % 2 == 0 else 0.45)) for k in range(4)]
            d.polygon(pts, fill=c + (al,))
        img.alpha_composite(lay)

    def lora(self, img, t):
        W, H = self.W, self.H
        cy = H * 0.5
        self.matn(img, (W / 2, cy - self.s(360)), "BONUS TAJRIBA", B(self.s(30)), mix(PINK, WHITE, 0.3),
                  a=ease(t * 2), glow=8)
        self.sozlar(img, t - 0.1, (W / 2, cy - self.s(285)), "LoRA vs to'liq fine-tuning", BLK(self.s(66)),
                    grad=[VIOLET, PINK, ORANGE], glow=12)
        self.matn(img, (W / 2, cy - self.s(215)), "bir xil 3 000 klip · 3 epoch · 200 val klip", R(self.s(30)), DIM,
                  a=ease((t - 0.6) * 2))
        kartalar = [
            ("LoRA", LORA, [CYAN, BLUE], "G'OLIB"),
            ("To'liq FT", FULL, [ORANGE, RED], None),
        ]
        if self.wide:
            kw, kh = W * 0.3, self.s(400)
            pos = [(W / 2 - kw - self.s(30), cy - self.s(150)), (W / 2 + self.s(30), cy - self.s(150))]
        else:
            kw, kh = W * 0.82, self.s(250)
            pos = [(W / 2 - kw / 2, cy - self.s(160)), (W / 2 - kw / 2, cy + self.s(120))]
        for i, ((nom, r, grad, nishon), (x, y)) in enumerate(zip(kartalar, pos)):
            a = ease((t - 0.6 - i * 0.3) * 2)
            dy = self.s(40) * (1 - back(clamp((t - 0.6 - i * 0.3) * 2)))
            y += dy
            self.shisha(img, (x, y, x + kw, y + kh), a=a, chegara=grad[0], kuch=0.10)
            self.matn(img, (x + self.s(34), y + self.s(50)), nom, B(self.s(40)), a=a, ha="l", grad=grad)
            wer_y = y + self.s(150) if self.wide else y + self.s(140)
            self.matn(img, (x + self.s(34), wer_y), f"{r['val200_WER']:.1f}%", BLK(self.s(100)), a=a, ha="l",
                      grad=grad, glow=12)
            self.matn(img, (x + self.s(34), wer_y + self.s(75)), "validation WER", R(self.s(26)), DIM, a=a, ha="l")
            info = [f"{r['oqitiladigan_foiz']}% parametr", f"{r['gpu_xotira_gb']} GB GPU",
                    f"{r['saqlangan_mb']:g} MB fayl"]
            if self.wide:
                for j, s_ in enumerate(info):
                    self.matn(img, (x + self.s(34), y + self.s(300 + j * 34)), s_, SB(self.s(27)), SOFT, a=a, ha="l")
            else:
                for j, s_ in enumerate(info):
                    self.matn(img, (x + kw - self.s(34), y + self.s(70 + j * 50)), s_, SB(self.s(27)), SOFT, a=a,
                              ha="r")
            if nishon:
                q = back(clamp((t - 2.0) / 0.4))
                if q > 0:
                    self.pill(img, (x + kw - self.s(20), y - self.s(4)), "🏆 " + nishon if False else nishon,
                              B(self.s(24 * q + 1)), EMERALD, a=1, ha="r")

    def demo(self, img, t):
        W, H = self.W, self.H
        tt = t - 0.8                       # audio sahna boshidan 0.8 s keyin boshlanadi
        # 16:9 - chapda vizualizator, o'ngda matn; 4:5 - tepada vizualizator, pastda matn
        vx = W * 0.27 if self.wide else W / 2
        cy = H * 0.5 if self.wide else H * 0.32
        sx = vx if self.wide else W / 2
        self.matn(img, (sx, H * (0.09 if self.wide else 0.045)), "4 · JONLI DEMO", B(self.s(30)), mix(PINK, WHITE, 0.3), a=ease(t * 2),
                  glow=8)
        self.matn(img, (sx, H * (0.09 if self.wide else 0.045) + self.s(46)), "test audio  →  modelning haqiqiy chiqishi",
                  R(self.s(28)), DIM, a=ease((t - 0.2) * 2))
        # aylana vizualizator
        R0 = self.s(120)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        nb = 72
        cols = grad_colors([PINK, VIOLET, CYAN, EMERALD, AMBER, PINK], nb)
        jonli = 0 < tt < AUDIO_S
        for i in range(nb):
            ang = 2 * math.pi * i / nb - math.pi / 2 + t * 0.4
            lv = rms_at(tt - 0.25 + 0.5 * i / nb) if jonli else 0.08 + 0.05 * math.sin(t * 3 + i)
            ln = self.s(14) + lv * self.s(150 if self.wide else 105)
            x0, y0 = vx + math.cos(ang) * (R0 + self.s(14)), cy + math.sin(ang) * (R0 + self.s(14))
            x1, y1 = vx + math.cos(ang) * (R0 + self.s(14) + ln), cy + math.sin(ang) * (R0 + self.s(14) + ln)
            d.line([(x0, y0), (x1, y1)], fill=tuple(int(v) for v in cols[i]) + (255,), width=self.s(7))
        img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(self.s(10))))
        img.alpha_composite(lay)
        pul = 1 + 0.08 * (rms_at(tt) if jonli else 0)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        rr = R0 * pul
        d.ellipse([vx - rr, cy - rr, vx + rr, cy + rr], fill=mix(VIOLET, PINK, 0.4) + (235,))
        img.alpha_composite(lay)
        self.mic(img, vx, cy, R0 * 0.9)
        # progress va vaqt
        bw = W * (0.36 if self.wide else 0.8)
        x0, yb = vx - bw / 2, cy + R0 + self.s(190)
        prog = clamp(tt / AUDIO_S)
        self.shisha(img, (x0, yb, x0 + bw, yb + self.s(10)), rad=5, kuch=0.15)
        if prog > 0:
            bar = grad_colors([PINK, VIOLET, CYAN], max(int(bw * prog), 2))
            img.alpha_composite(Image.fromarray(np.repeat(bar[None], self.s(10), axis=0)).convert("RGBA"),
                                dest=(int(x0), int(yb)))
        self.matn(img, (x0, yb + self.s(36)), f"0:{clamp(tt, 0, AUDIO_S):04.1f}", MONO(self.s(26)), DIM, ha="l")
        self.matn(img, (x0 + bw, yb + self.s(36)), f"0:{AUDIO_S:04.1f}", MONO(self.s(26)), DIM, ha="r")
        # matn kartochkasi: so'zlar kursor bilan sinxron chiqadi
        if self.wide:
            kx0, kx1, ky0, ky1 = W * 0.5, W * 0.94, H * 0.16, H * 0.84
        else:
            kx0, kx1, ky0, ky1 = W * 0.06, W * 0.94, yb + self.s(90), H * 0.95
        self.shisha(img, (kx0, ky0, kx1, ky1), a=ease(t * 2), chegara=PINK, kuch=0.09)
        sozlar = MATN.split()
        k = int(len(sozlar) * clamp((tt - 0.3) / (AUDIO_S - 0.6)))
        f = SB(self.s(42 if self.wide else 38))
        sp = f.getlength(" ")
        x, y = kx0 + self.s(40), ky0 + self.s(56)
        lh = f.size * 1.45
        for i, w in enumerate(sozlar[:k]):
            ww = f.getlength(w)
            if x + ww > kx1 - self.s(40):
                x, y = kx0 + self.s(40), y + lh
            yangi = i == k - 1 and k < len(sozlar)
            if yangi:
                self.shisha(img, (x - self.s(10), y - f.size * 0.75, x + ww + self.s(10), y + f.size * 0.75),
                            rad=14, chegara=CYAN, kuch=0.18)
                self.matn(img, (x, y), w, f, ha="l", grad=[CYAN, PINK], glow=10)
            else:
                self.matn(img, (x, y), w, f, SOFT, ha="l")
            x += ww + sp
        if k >= len(sozlar):
            q = back(clamp((tt - AUDIO_S - 0.2) / 0.5))
            if q > 0:
                self.pill(img, ((kx0 + kx1) / 2, ky1 - self.s(56)), "haqiqiy matn bilan aynan bir xil  ·  WER 0%",
                          B(self.s(30 * q + 1)), EMERALD)

    def yakun(self, img, t):
        W, H = self.W, self.H
        cy = H * 0.5
        self.sozlar(img, t - 0.1, (W / 2, cy - self.s(250)), "O'zingiz sinab ko'ring", BLK(self.s(84)),
                    grad=[CYAN, VIOLET, PINK, ORANGE], glow=18)
        kartalar = [
            ("HF", "Model", "huggingface.co/Toxirerkinovtohir/whisper-small-karakalpak", [AMBER, ORANGE]),
            ("play", "Demo", "huggingface.co/spaces/Toxirerkinovtohir/karakalpak-asr", [PINK, VIOLET]),
            ("</>", "Kod", "github.com/toxirerkinov70-commits/whisper-small-karakalpak", [CYAN, BLUE]),
        ]
        kw = min(W * 0.9, self.s(1180))
        for i, (ikon, lab, url, grad) in enumerate(kartalar):
            a = ease((t - 0.5 - i * 0.2) * 2.5)
            y = cy - self.s(120) + i * self.s(130) + self.s(40) * (1 - back(clamp((t - 0.5 - i * 0.2) * 2.5)))
            x0 = W / 2 - kw / 2
            self.shisha(img, (x0, y, x0 + kw, y + self.s(104)), a=a, chegara=grad[0], kuch=0.10)
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            r = self.s(34)
            ImageDraw.Draw(lay).ellipse([x0 + self.s(26), y + self.s(52) - r, x0 + self.s(26) + 2 * r,
                                         y + self.s(52) + r], fill=grad[0] + (int(255 * a),))
            img.alpha_composite(lay)
            if ikon == "play":                 # uchburchak belgini o'zimiz chizamiz (shriftda yo'q bo'lishi mumkin)
                lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ccx, ccy, tr = x0 + self.s(26) + r + self.s(3), y + self.s(52), self.s(13)
                ImageDraw.Draw(lay).polygon([(ccx - tr * 0.8, ccy - tr), (ccx - tr * 0.8, ccy + tr), (ccx + tr, ccy)],
                                            fill=NAVY + (int(255 * a),))
                img.alpha_composite(lay)
            else:
                self.matn(img, (x0 + self.s(26) + r, y + self.s(52)), ikon, B(self.s(24)), NAVY, a=a)
            self.matn(img, (x0 + self.s(120), y + self.s(34)), lab, B(self.s(30)), WHITE, a=a, ha="l")
            self.matn(img, (x0 + self.s(120), y + self.s(72)), url, R(self.s(26 if self.wide else 22)), SOFT, a=a,
                      ha="l")
        self.matn(img, (W / 2, cy + self.s(330)), "Toxir Erkinov  ·  Whisper-small × qoraqalpoq tili",
                  SB(self.s(32)), DIM, a=ease((t - 1.4) * 2))

    # ---------- o'tish ----------
    def otish(self, img, t):
        """Sahnalar orasida rangli diagonal "wipe": kesim joyini yopadi."""
        for tb in CHEGARALAR:
            if tb - WIPE <= t <= tb + WIPE:
                p = (t - (tb - WIPE)) / (2 * WIPE)
                p = ease_io(p)
                W, H = self.W, self.H
                bw = W * 1.6
                cx = -bw / 2 - H * 0.6 + p * (W + bw + H * 1.2)
                poly = [(cx - bw / 2, 0), (cx + bw / 2, 0), (cx + bw / 2 - H * 0.6, H), (cx - bw / 2 - H * 0.6, H)]
                m = Image.new("L", (W, H), 0)
                ImageDraw.Draw(m).polygon(poly, fill=255)
                stops = [VIOLET, PINK, ORANGE, AMBER] if CHEGARALAR.index(tb) % 2 == 0 else [CYAN, BLUE, VIOLET,
                                                                                             PINK]
                g = grad_colors(stops, W)
                layer = Image.fromarray(np.repeat(g[None], H, axis=0)).convert("RGBA")
                layer.putalpha(m)
                img.alpha_composite(layer)

    # ---------- kadr ----------
    def chiz(self, t):
        img = self.fon(t)
        self.zarrachalar(img, t)
        for nom, (t0, t1, _) in S.items():
            if t0 <= t < t1 or (nom == "yakun" and t >= t0):
                getattr(self, nom)(img, t - t0)
                break
        self.otish(img, t)
        # pastki progress
        g = grad_colors([CYAN, VIOLET, PINK, ORANGE], max(int(self.W * t / DUR), 2))
        img.alpha_composite(Image.fromarray(np.repeat(g[None], self.s(8), axis=0)).convert("RGBA"),
                            dest=(0, self.H - self.s(8)))
        return img.convert("RGB")


def video(W, H, nom):
    k = Kadr(W, H)
    out = OUT / nom
    n = int(DUR * FPS)
    start = S["demo"][0] + 0.8
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(AUDIO),
         "-filter_complex", f"[1:a]adelay={int(start * 1000)}:all=1,apad,atrim=0:{DUR:.2f}[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "160k", str(out)],
        stdin=subprocess.PIPE)
    try:
        for i in range(n):
            ff.stdin.write(k.chiz(i / FPS).tobytes())
            if i % 300 == 0:
                print(f"  {nom}: {i}/{n}", flush=True)
    except BaseException:
        # chala video jim qolib ketmasin: ffmpeg'ni to'xtatib, faylni o'chiramiz
        ff.kill()
        ff.wait()
        out.unlink(missing_ok=True)
        raise
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg xato bilan tugadi: {out}")
    print("tayyor:", out, f"{out.stat().st_size / 1e6:.1f} MB", flush=True)


def demo_gif():
    """README tepasi uchun: aylana vizualizator + yozilib boradigan matn, 720x405, 12 fps.
    GIF hajmi kichik bo'lishi uchun fon qotirilgan va zarrachalar yo'q (ular siqishni buzadi)."""
    W, H, fps = 720, 405, 12
    k = Kadr(W, H)
    k.k = 0.375
    tmp = OUT / "_demo_frames"
    tmp.mkdir(exist_ok=True)
    t0, jami = S["demo"][0], DEMO_LEN
    fon = k.fon(t0 + 2.0)
    for i in range(int(jami * fps)):
        t = i / fps
        img = fon.copy()
        k.demo(img, t)
        img.convert("RGB").save(tmp / f"{i:04d}.png")
    out = OUT / "demo.gif"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "%04d.png"),
                    "-vf", "split[a][b];[a]palettegen=max_colors=200:stats_mode=full[p];"
                           "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                    "-loop", "0", str(out)], check=True)
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()
    print("tayyor:", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    nima = sys.argv[1:] or ["gif"]
    if "kadr" in nima:
        for W, H, nom in [(1920, 1080, "16x9"), (1080, 1350, "4x5")]:
            k = Kadr(W, H)
            for s_, (t0, t1, _) in S.items():
                k.chiz(t0 + (t1 - t0) * 0.72).save(OUT / f"_kadr_{nom}_{s_}.png")
        print("kadrlar tayyor")
        sys.exit()
    if "16x9" in nima:
        video(1920, 1080, "video_16x9.mp4")
    if "4x5" in nima:
        video(1080, 1350, "video_4x5.mp4")
    if "gif" in nima:
        demo_gif()

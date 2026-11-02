"""README grafiklari: har biri yorug' va qorong'i tema uchun (assets/*_light.png, assets/*_dark.png).
Barcha raqamlar repodagi natija fayllaridan o'qiladi, qo'lda yozilgan raqam yo'q.

Ishga tushirish (repo ildizidan):  python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
DPI = 200

# Palitra: dataviz reference palette, 3 slot (all-pairs CVD tekshiruvidan o'tgan, ikkala rejimda)
THEMES = {
    "light": dict(
        surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
        ramp=["#86b6ef", "#3987e5", "#1c5cab"], gray="#c3c2b7",
    ),
    "dark": dict(
        surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        s1="#3987e5", s2="#d95926", s3="#199e70",
        ramp=["#184f95", "#2a78d6", "#6da7ec"], gray="#52514e",
    ),
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------- yordamchilar ----------
def setup(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, pad=6)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def title(fig, t, text, sub):
    fig.text(0.02, 0.965, text, color=t["ink"], fontsize=15, fontweight="semibold", va="top")
    fig.text(0.02, 0.885, sub, color=t["ink2"], fontsize=10.5, va="top")


def px(ax, n, axis="y"):
    """n piksel -> data birligi (yumaloq uchlar uchun)."""
    bb = ax.get_window_extent()
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    size = bb.height if axis == "y" else bb.width
    return n * abs(hi - lo) / size     # abs: teskari o'qda (invert_yaxis) ham musbat radius


def rbar(ax, x, h, w, color, horizontal=False, r=4):
    """Uchi 4px yumaloq, asosi to'g'ri burchakli ustun/chiziq (mark spec)."""
    if h <= 0:
        return
    if not horizontal:
        ry, rx = min(px(ax, r, "y"), h / 2), min(px(ax, r, "x"), w / 2)
        x0, x1 = x - w / 2, x + w / 2
        verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    else:
        rx, ry = min(px(ax, r, "x"), h / 2), min(px(ax, r, "y"), w / 2)
        y0, y1 = x - w / 2, x + w / 2
        verts = [(0, y0), (h - rx, y0), (h, y0), (h, y0 + ry), (h, y1 - ry), (h, y1), (h - rx, y1), (0, y1), (0, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def legend(fig, t, items, x=0.02, y=0.80):
    """Rangli kvadrat + matn; keyingi element oldingi matn tugagan joydan boshlanadi."""
    renderer = fig.canvas.get_renderer()
    fw = fig.get_window_extent().width
    xx = x
    for label, color in items:
        fig.patches.append(plt.Rectangle((xx, y - 0.012), 0.012, 0.024, transform=fig.transFigure,
                                         facecolor=color, edgecolor="none"))
        txt = fig.text(xx + 0.018, y, label, color=t["ink2"], fontsize=9.5, va="center")
        xx += 0.018 + txt.get_window_extent(renderer).width / fw + 0.035


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}_{mode}.png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------- ma'lumotlar ----------
yakuniy = pd.read_csv(ROOT / "07_eval/yakuniy_jadval.csv")
tarix = pd.read_csv(ROOT / "06_finetune/results/train_history.csv")
xato = pd.read_csv(ROOT / "07_eval/xato_sabablari.csv")
test_b5 = pd.read_csv(ROOT / "07_eval/test_finetune_beam5.csv", keep_default_na=False)
manifest = pd.read_csv(ROOT / "04_normalize/manifest_tayyor.csv", usecols=["split", "yangi_s"])
trim = pd.read_csv(ROOT / "03_trim/trim_manifest_yakuniy.csv", usecols=["split", "yangi_s"])
summary = pd.read_csv(ROOT / "01_EDA/dataset_summary.csv", encoding="utf-8-sig")
lora = json.load(open(ROOT / "08_lora/lora_natija.json", encoding="utf-8"))
full = json.load(open(ROOT / "08_lora/full_natija.json", encoding="utf-8"))


def fig_natija(t, mode):
    """1. Asosiy natija: test WER (1 018 klip)."""
    y = yakuniy.set_index(["model", "dekodlash"])
    qatorlar = [
        ("Fine-tuning'siz\n(baseline)", y.loc[("baseline_kk_lotin", "greedy"), "test_WER"], t["s2"]),
        ("Fine-tune\ngreedy", y.loc[("finetune", "greedy"), "test_WER"], t["s1"]),
        ("Fine-tune\nbeam5 + no_repeat4", y.loc[("finetune", "beam5_norep4"), "test_WER"], t["s1"]),
        ("Fine-tune\nbeam5  ★ tanlangan", y.loc[("finetune", "beam5"), "test_WER"], t["s1"]),
    ]
    fig = plt.figure(figsize=(8, 4.4), facecolor=t["surface"])
    ax = fig.add_axes([0.08, 0.15, 0.9, 0.56])
    setup(ax, t)
    ax.set_xlim(-0.6, len(qatorlar) - 0.4)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    for i, (nom, v, c) in enumerate(qatorlar):
        rbar(ax, i, v, 0.34, c)
        ax.text(i, v + 3, f"{v:.1f}%", ha="center", va="bottom", color=t["ink"], fontsize=11,
                fontweight="semibold" if i == 3 else "normal")
    ax.set_xticks(range(len(qatorlar)))
    ax.set_xticklabels([q[0] for q in qatorlar], fontsize=9)
    base, best = qatorlar[0][1], qatorlar[3][1]
    ax.annotate("", xy=(2.95, best + 12), xytext=(0.3, base - 8),
                arrowprops=dict(arrowstyle="->", color=t["muted"], lw=1.2, connectionstyle="arc3,rad=-0.12"))
    ax.text(1.95, 82, f"{base / best:.1f}× kamroq xato", color=t["ink"], fontsize=11, ha="left")
    cer = y.loc[("finetune", "beam5"), "test_CER"]
    title(fig, t, f"Test WER: {base:.1f}% → {best:.1f}%",
          f"Test to'plami, 1 018 klip · bir xil normalizatsiya, fp32 · CER: "
          f"{y.loc[('baseline_kk_lotin', 'greedy'), 'test_CER']:.1f}% → {cer:.2f}%")
    legend(fig, t, [("Fine-tuning'siz Whisper-small", t["s2"]), ("Qoraqalpoqchaga fine-tune", t["s1"])], y=0.79)
    save(fig, "natija", mode)


def fig_trening(t, mode):
    """2. Trening: validation WER/CER (greedy, Trainer ichida) va loss."""
    ev = tarix[tarix.turi == "eval"]
    tr = tarix[tarix.turi == "train"]
    fig = plt.figure(figsize=(8, 4.2), facecolor=t["surface"])
    a1 = fig.add_axes([0.07, 0.13, 0.42, 0.58])
    a2 = fig.add_axes([0.57, 0.13, 0.41, 0.58])
    for ax in (a1, a2):
        setup(ax, t)
        ax.set_xlim(0, 5900)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels(["0", "1K", "2K", "3K", "4K", "5K"])
    a1.plot(ev.step, ev.wer, color=t["s1"], lw=2, solid_capstyle="round", zorder=3)
    a1.plot(ev.step, ev.cer, color=t["s2"], lw=2, solid_capstyle="round", zorder=3)
    for col, c in (("wer", t["s1"]), ("cer", t["s2"])):
        a1.scatter(ev.step, ev[col], s=34, color=c, edgecolor=t["surface"], linewidth=2, zorder=4)
    a1.set_ylim(0, 85)
    a1.set_yticks([0, 20, 40, 60, 80])
    a1.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80]])
    sp = ev[ev.step == 1000].iloc[0]
    a1.annotate("halqa: model so'zni\ntakrorlab qolgan", xy=(1000, sp.wer), xytext=(1700, 74),
                color=t["ink2"], fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    ck = ev[ev.step == 5000].iloc[0]
    a1.axvline(5000, color=t["muted"], lw=1, zorder=2)
    a1.text(4950, 62, "checkpoint-5000\ntanlandi", color=t["ink2"], fontsize=8.5, ha="right")
    a1.text(5080, ck.wer + 3, f"{ck.wer:.1f}%", color=t["ink"], fontsize=9, ha="left", va="bottom")

    a2.plot(tr.step, tr.loss, color=t["s1"], lw=1.6, alpha=0.9, zorder=3)
    a2.plot(ev.step, ev.loss, color=t["s3"], lw=2, zorder=3)
    a2.scatter(ev.step, ev.loss, s=34, color=t["s3"], edgecolor=t["surface"], linewidth=2, zorder=4)
    a2.set_ylim(0, 1.2)
    a2.text(5900, ev.loss.iloc[-1] + 0.1, f"val {ev.loss.iloc[-1]:.3f}", color=t["ink2"], fontsize=8.5, ha="right")
    for ax in (a1, a2):
        ax.set_xlabel("qadam", color=t["muted"], fontsize=9)
    title(fig, t, "Trening: 5 epoch, 2 soat 33 daqiqa (DGX Spark)",
          "whisper-small, lr 1e-5, batch 16, bf16 · trening paytidagi baho greedy dekodlash bilan")
    legend(fig, t, [("validation WER", t["s1"]), ("validation CER", t["s2"])], x=0.07, y=0.79)
    legend(fig, t, [("train loss", t["s1"]), ("validation loss", t["s3"])], x=0.57, y=0.79)
    save(fig, "trening", mode)


def fig_malumot(t, mode):
    """3. Ma'lumot quvuri: asl dataset -> trim -> normalizatsiya."""
    asl_klip = int(summary["count"].sum())
    asl_soat = float(summary["hours"].sum())
    bosqich = [
        ("Asl dataset", asl_klip, asl_soat),
        ("Silero VAD trim", len(trim), trim.yangi_s.sum() / 3600),
        ("Normalizatsiya +\ntozalash (yakuniy)", len(manifest), manifest.yangi_s.sum() / 3600),
    ]
    fig = plt.figure(figsize=(8, 3.9), facecolor=t["surface"])
    ax = fig.add_axes([0.2, 0.08, 0.5, 0.62])
    setup(ax, t, grid_axis="x")
    ax.set_xlim(0, 24500)
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_xticklabels(["0", "5K", "10K", "15K", "20K"])
    ax.set_yticks(range(3))
    ax.set_yticklabels([b[0] for b in bosqich], color=t["ink2"])
    oldingi = None
    for i, (nom, n, soat) in enumerate(bosqich):
        rbar(ax, i, n, 0.5, t["ramp"][i], horizontal=True)
        ax.text(n + 300, i - 0.08, f"{n:,}".replace(",", " ") + " klip", color=t["ink"], fontsize=10, va="center")
        ax.text(n + 300, i + 0.2, f"{soat:.1f} soat" + (f"  (−{oldingi - n})" if oldingi else ""),
                color=t["ink2"], fontsize=8.5, va="center")
        oldingi = n
    # split bo'linishi
    a2 = fig.add_axes([0.78, 0.08, 0.2, 0.62])
    a2.set_facecolor(t["surface"])
    a2.axis("off")
    sp = manifest.groupby("split").agg(klip=("yangi_s", "size"), soat=("yangi_s", lambda s: s.sum() / 3600))
    a2.text(0, 1.0, "Yakuniy bo'linish", color=t["ink"], fontsize=10, transform=a2.transAxes, va="top")
    for k, (s, c) in enumerate([("train", t["s1"]), ("validation", t["s2"]), ("test", t["s3"])]):
        yy = 0.78 - k * 0.24
        a2.add_patch(plt.Rectangle((0, yy - 0.04), 0.06, 0.08, transform=a2.transAxes, facecolor=c, edgecolor="none"))
        a2.text(0.12, yy + 0.02, s, color=t["ink2"], fontsize=9, transform=a2.transAxes, va="center")
        a2.text(0.12, yy - 0.08, f"{int(sp.loc[s, 'klip']):,}".replace(",", " ") + f" · {sp.loc[s, 'soat']:.1f} soat",
                color=t["ink"], fontsize=9, transform=a2.transAxes, va="center")
    title(fig, t, f"Ma'lumot: {asl_soat:.0f} soatdan {manifest.yangi_s.sum() / 3600:.0f} soat toza juftlik",
          "atikuwu/karakalpak-speech-corpus (CC-BY-4.0) · jimlik kesildi, matn tozalandi, "
          "dublikat va leakage olindi")
    save(fig, "malumot", mode)


def fig_xatolar(t, mode):
    """4. Xatolar tahlili: validation'dagi so'z xatolari sabablari."""
    d = xato.copy()
    d["nom"] = d["sabab"].str.replace(r"^\d+ ", "", regex=True)
    boshqa = d[d.nom == "boshqa"]
    d = pd.concat([d[d.nom != "boshqa"].sort_values("ulush_%", ascending=False), boshqa])
    misol = {
        "qo'shimcha (o'zak bir xil)": "mámleketimizde → mámleketimizdi",
        "halqa": "qalqan qalqan qalqan …",
        "harf chalkashligi (1-2 harf)": "baǵ → baq",
        "atoqli ot / qisqartma": "sh mirziyoev ↔ «Shavkat»",
        "gap boshi": "birinchi so'z tushib qolgan",
    }
    fig = plt.figure(figsize=(8, 4.6), facecolor=t["surface"])
    ax = fig.add_axes([0.3, 0.06, 0.6, 0.68])
    setup(ax, t, grid_axis="x")
    n = len(d)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels([f"{v}%" for v in [0, 10, 20, 30]])
    ax.spines["bottom"].set_bounds(0, 35)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0].upper() + s[1:] for s in d.nom], fontsize=9)
    for i, (_, r) in enumerate(d.iterrows()):
        c = t["gray"] if r.nom == "boshqa" else t["s1"]
        rbar(ax, i, r["ulush_%"], 0.56, c, horizontal=True)
        lab = f"{r['ulush_%']:.1f}%"
        if r.nom in misol:
            lab += f"   {misol[r.nom]}"
        ax.text(r["ulush_%"] + 0.6, i, lab, color=t["ink"] if r.nom in misol else t["ink2"], fontsize=9, va="center")
    title(fig, t, f"Xatolar tahlili: {int(xato.xato_soni.sum()):,} so'z xatosi".replace(",", " "),
          "Validation, beam5 · har xato bitta sababga biriktirilgan · til agglyutinativ: "
          "qo'shimcha xatolari eng ko'p")
    save(fig, "xatolar", mode)


def fig_lora(t, mode):
    """5. LoRA va to'liq fine-tuning: bir xil 3 000 klip, 3 epoch."""
    metr = [
        ("Validation WER (200 klip)", "val200_WER", "%", 100),
        ("GPU xotira", "gpu_xotira_gb", " GB", 14),
        ("Saqlanadigan fayl", "saqlangan_mb", " MB", 1100),
    ]
    fig = plt.figure(figsize=(8, 3.8), facecolor=t["surface"])
    for k, (nom, key, birlik, ymax) in enumerate(metr):
        ax = fig.add_axes([0.04 + k * 0.33, 0.1, 0.27, 0.56])
        setup(ax, t)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, ymax)
        ax.set_yticklabels([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LoRA", "To'liq FT"])
        for i, (r, c) in enumerate(((lora, t["s1"]), (full, t["s2"]))):
            v = r[key]
            rbar(ax, i, v, 0.5, c)
            ax.text(i, v + ymax * 0.03, f"{v:g}{birlik}", ha="center", color=t["ink"], fontsize=10)
        ax.text(0, 1.06, nom, transform=ax.transAxes, color=t["ink"], fontsize=10)
    title(fig, t, f"LoRA: WER {lora['val200_WER']:.1f}% vs to'liq fine-tuning {full['val200_WER']:.1f}%",
          f"Bir xil 3 000 klip, 3 epoch · LoRA parametrlarning atigi {lora['oqitiladigan_foiz']}% ini o'qitadi "
          f"va kam ma'lumotda ancha yaxshi umumlashtiradi")
    save(fig, "lora", mode)


def fig_taqsimot(t, mode):
    """6. Har bir test klipi uchun WER taqsimoti (beam5)."""
    w = test_b5["wer"].astype(float).clip(upper=100)
    bins = np.arange(0, 105, 5)
    hist, _ = np.histogram(w, bins=bins)
    fig = plt.figure(figsize=(8, 3.9), facecolor=t["surface"])
    ax = fig.add_axes([0.08, 0.14, 0.9, 0.56])
    setup(ax, t)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, hist.max() * 1.18)
    for b, h in zip(bins[:-1], hist):
        rbar(ax, b + 2.5, h, 3.9, t["s1"] if b < 20 else t["ramp"][0])
    med = float(np.median(w))
    ax.axvline(med, color=t["muted"], lw=1, zorder=2)
    ax.text(med + 1, hist.max() * 1.1, f"mediana {med:.1f}%", color=t["ink2"], fontsize=9)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "≥100%"])
    ax.set_xlabel("bitta klipning WER'i", color=t["muted"], fontsize=9)
    ulush = (w < 20).mean() * 100
    title(fig, t, f"Test kliplarining {ulush:.0f}% ida WER 20% dan past",
          f"1 018 test klipi, beam5 · {(w == 0).mean() * 100:.0f}% klip xatosiz o'qilgan · "
          f"to'q ustunlar: WER < 20%")
    save(fig, "taqsimot", mode)


if __name__ == "__main__":
    for mode, t in THEMES.items():
        for f in (fig_natija, fig_trening, fig_malumot, fig_xatolar, fig_lora, fig_taqsimot):
            f(t, mode)
    print("tayyor:", sorted(p.name for p in OUT.glob("*.png")))

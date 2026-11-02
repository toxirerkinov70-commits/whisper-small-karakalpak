# Gradio demo: mikrofon yoki audio fayl -> qoraqalpoqcha matn
# Lokal ishga tushirish:  python app.py   (keyin brauzerda http://127.0.0.1:7860)
# Xuddi shu fayl Hugging Face Space'da ham ishlaydi.
import time

import gradio as gr

from transcribe import MODEL, Transkriptor

NAMUNALAR = [
    # (fayl, test to'plamidagi haqiqiy matn) - atikuwu/karakalpak-speech-corpus, CC-BY-4.0
    ("examples/namuna_1.flac",
     "parlament qadaǵalawın alıp barıwda tiykarǵı itibar aymaqlarǵa qaratıladı mámleketimiz basshısı tárepinen "
     "sociallıq ekonomikalıq tarawlarda anıq wazıypalar hám úlken shekler qoyılmaqta"),
    ("examples/namuna_2.flac",
     "ilajda qaraqalpaqstan respublikası salıq basqarmasınıń baslıǵı xalmurza bekchanov sawda sanaat palatasınıń "
     "baslıǵı kazbek artikov taraw qánigeleri isbilermenler hám xalıq wákilleri qatnastı"),
    ("examples/namuna_3.flac",
     "sonıń ushın mektepler hám kásip ónerge oqıtıw oraylarında bunday sabaqlardı kóbeytiw texikumlarǵa sırt elden "
     "oqıtıwshılardı alıp keliw boyınsha kórsetpeler berildi"),
]
HAQIQIY = {fayl: matn for fayl, matn in NAMUNALAR}

transkriptor = Transkriptor(model=MODEL)


def transkripsiya(audio, tez_rejim):
    if audio is None:
        return "Audio yozib oling yoki fayl yuklang.", ""
    transkriptor.beam = not tez_rejim
    t0 = time.time()
    matn, soniya = transkriptor.fayl(audio)
    info = (f"{soniya:.1f} s audio · {time.time() - t0:.1f} s ishlov · "
            f"{'greedy' if tez_rejim else 'beam 5'} · {transkriptor.device.upper()}")
    for fayl, haqiqiy in HAQIQIY.items():
        if audio.replace("\\", "/").endswith(fayl.split("/")[-1]):
            info += f"\n\nHaqiqiy matn (test to'plamidan): {haqiqiy}"
    return matn or "(nutq topilmadi)", info


TAVSIF = """
# 🎙️ Qoraqalpoqcha nutqni tanish
**Whisper-small**, qoraqalpoq tiliga fine-tune qilingan · 99 soat nutq · test WER **100.3% → 20.2%**, CER **7.1%**

Mikrofonga gapiring yoki audio yuklang (wav, mp3, flac, m4a, ogg). 30 soniyadan uzun audio pauzalardan bo'laklanadi.
"""

with gr.Blocks(title="Qoraqalpoqcha ASR · Whisper-small",
               theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange")) as demo:
    gr.Markdown(TAVSIF)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
            tez = gr.Checkbox(label="Tez rejim (greedy) — tezroq, lekin sifati past (test WER 27.0% vs 20.2%)", value=False)
            tugma = gr.Button("Matnga aylantirish", variant="primary")
        with gr.Column():
            chiqish = gr.Textbox(label="Qoraqalpoqcha matn", lines=6, show_copy_button=True)
            malumot = gr.Markdown()
    gr.Examples([[f, False] for f, _ in NAMUNALAR], inputs=[audio, tez], label="Test to'plamidan namunalar")
    tugma.click(transkripsiya, inputs=[audio, tez], outputs=[chiqish, malumot])
    gr.Markdown(
        "Model: [`" + MODEL + "`](https://huggingface.co/" + MODEL + ") · "
        "Kod: [GitHub](https://github.com/toxirerkinov70-commits/whisper-small-karakalpak) · "
        "Ma'lumot: [atikuwu/karakalpak-speech-corpus](https://huggingface.co/datasets/atikuwu/karakalpak-speech-corpus) "
        "(CC-BY-4.0)"
    )

if __name__ == "__main__":
    demo.launch()

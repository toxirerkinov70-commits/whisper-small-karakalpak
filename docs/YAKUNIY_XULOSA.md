# Qoraqalpoq ASR: 3 haftalik intern loyihasi xulosasi

**Maqsad:** qoraqalpoq nutqini matnga aylantiradigan model yasash va ASR'ning hamma bosqichini o'z qo'lim bilan bosib o'tish: ma'lumot → model → o'qitish → baholash → xatolar tahlili.

## Ma'lumot (1-hafta)
- Dataset: `atikuwu/karakalpak-speech-corpus`, **21 702** audio, **106.92 soat**, 48 kHz, mono.
- Tozalash: Silero VAD bilan trim (232 ta audio chiqarildi), normalizatsiya (kichik harf, rim raqami artefaktlari, `mln` → `million`, oxiridagi "ÓzA"), qavsli matnlar, 30 s dan uzunlar, leakage va dublikatlar chiqarildi.
- Natija: **20 737** juftlik, **99.01 soat** (train 18 685 / 89.3 soat, validation 1 034, test 1 018).
- O'z baholash to'plamim: test'dan 50 klip (`eval_50`), ularni tinglab tekshirdim (`gold_50`): 37 tasi to'g'ri, 11 tasida audio muammosi, 2 tasida matn xatosi.

## Model (2-hafta)
- Attention va multi-head attention'ni noldan yozdim. wav2vec2 (94.4M parametr) ichini ochib, har bir qatlam tensor shaklini kuzatdim. CTC greedy'ni qo'lda yozdim, natija kutubxona bilan bir xil chiqdi.
- CTC'ni 3 usulda dekodladim (LibriSpeech, test WER): greedy **5.55%**, beam **5.55%**, beam + Wikipedia bigram LM **6.28%**. Saboq: LM o'sha tildagi va sohadagi matnda o'qitilmasa, zarar ham beradi.
- Decoder tanlovi: Whisper (AED) fine-tune qilindi, chunki u tayyor ko'p tilli model va baseline shu modelda o'lchangan. Uzoq muddat uchun tavsiya: ko'p tilli w2v-BERT 2.0 + CTC + harf lug'ati + qoraqalpoq LM.

## Fine-tuning va baholash (3-hafta)
- whisper-small (244M), til tokeni `uz`, DGX Spark (GB10), 5 epoch, 2 soat 33 daqiqa. Tanlangan model: **checkpoint-5000**.
- Teng sharoitda baholash (fp32, bir xil normalizatsiya):

| | eval_50 WER | **Test WER (1 018 klip)** | Test CER |
|---|---|---|---|
| Baseline (fine-tuning'siz, kk → lotin) | 100.72% | **100.29%** | 48.69% |
| **Fine-tune, beam5** | 19.39% | **20.22%** | **7.12%** |
| Fine-tune, greedy | 21.13% | 27.04% | 13.35% |
| Fine-tune, beam5 + no_repeat4 | 23.35% | 22.45% | 7.07% |

- Dekodlash sozlamasi **validation'da** tanlandi: beam5 WER 20.07%, beam5 + no_repeat4 21.64%. Validation va test natijalari bir-biriga yaqin (20.07% va 20.22%), demak raqam ishonchli.
- **LoRA va to'liq fine-tuning** (bir xil 3 000 klip, 3 epoch, 200 val klip): LoRA **37.9%** WER, to'liq FT **89.4%**. LoRA 1.44% parametr o'qitadi, 7.3 GB xotira oladi va 13.5 MB fayl saqlaydi (to'liq FT: 11.6 GB va 922 MB). Kam ma'lumotda LoRA ancha yaxshi umumlashtirdi.

## Xatolar tahlili (validation, 6 715 so'z xatosi)
1. **Qo'shimchalar: 33.7%** (`mámleketimizde` → `mámleketimizdi`). Til agglyutinativ.
2. **Halqalar: 11.7%**, bor-yo'g'i 10 ta klipdan (`qalqan qalqan qalqan...`).
3. **Harf chalkashligi: 10.6%** (`baǵ` → `baq`, `ǵ↔q`, `á↔a`).
4. **Atoqli ot va qisqartmalar: 7.2%**. Bir qismi reference muammosi: matnda `sh mirziyoev`, diktor esa "Shavkat" deb o'qigan.
5. **Gap boshi: 5.7%**. "Trim nutq boshini kesgan" gipotezasi raqamlar bilan **rad etildi** (kesilgan kliplarda 31.0%, kesilmaganlarda 37.5%).

## Asosiy saboqlar
1. **Raqamga darrov ishonmaslik.** Trening paytida greedy WER 44% dan 77% ga sakradi. Sabab model yomonlashgani emas, halqalar edi. Gipoteza → kichik tajriba → dalil usuli bilan isbotlandi.
2. **Faqat teng sharoitda solishtirish.** Normalizatsiya, dekodlash va to'plam bir xil bo'lmasa, raqamlar solishtirilmaydi. Sozlama validation'da tanlanadi, test esa faqat yakuniy baho uchun.
3. **Ma'lumot sifati model sifati bilan teng.** Xatolarning bir qismi reference matndagi nomuvofiqlikdan keladi. "Audio" muammoli kliplarsiz WER ~2 punkt past chiqdi (19.50% → 17.39%).

## Cheklovlar va keyingi qadamlar
- `no_repeat` 6 va LoRA/to'liq FT uchun val_loss o'lchanmagan. LoRA va to'liq FT'da LR har xil, vaqt esa server yuklamasi bilan birga o'lchangan.
- Keyingi qadamlar: qoraqalpoq matnidan LM qurib fusion qilish; to'liq ma'lumotda LoRA; w2v-BERT 2.0 + CTC bilan solishtirish; qisqartma va sonlarni normalizatsiyada audioga mos yozish.

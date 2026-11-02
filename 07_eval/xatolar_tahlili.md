# Xatolar tahlili: checkpoint-5000 ning top 5 xato sababi

**Model:** whisper-small, qoraqalpoq tiliga fine-tune qilingan (checkpoint-5000). Dekodlash: beam5, no_repeat'siz.
**Ma'lumot:** validation, 1 034 klip. WER 20.07%, CER 7.83%.
**Manba:** `07_eval/val_finetune_beam5.csv` (har bir klip uchun haqiqiy matn va model matni). Tahlil skripti: `07_eval/xato_tahlili.py`. Natijalar: `xato_sabablari.csv`, `xato_misollari.csv`.

## Usul

1. Har bir klipda haqiqiy matn va model matni so'zma-so'z tekislandi (`jiwer.process_words`). Natijada **6 715 ta so'z xatosi** chiqdi: almashtirish (S), tushib qolish (D) va ortiqcha qo'shilish (I).
2. Har bir xato bitta sababga biriktirildi. Qoidalar quyidagi tartibda tekshirildi va birinchi mos kelgani olindi:

| Tartib | Sabab | Qoida |
|---|---|---|
| 1 | Halqa | Klipda model haqiqiydan 1.5 baravardan ko'p so'z yozgan |
| 2 | Gap boshi | Xato gapning 1-so'zida |
| 3 | Son so'zi | Haqiqiy yoki model so'zi son (`bir`, `jeti`, `jetinshi` ...) |
| 4 | Atoqli ot / qisqartma | Asl matnda gap o'rtasida katta harf bilan yozilgan so'z |
| 5 | Qo'shimcha | O'zagi bir xil: kamida 3 ta va kamida yarmi bir xil bosh harf, oxiri boshqa |
| 6 | Harf chalkashligi | 1–2 harf farq qiladi (Levenshtein ≤ 2) |
| 7 | Yordamchi so'z | `da`, `de`, `hám`, `menen` ... tushib qolgan yoki qo'shilgan |
| 8 | Boshqa | qolgan hammasi |

## Natija: xatolar ulushi

| Sabab | Xatolar soni | Ulushi |
|---|---|---|
| **Qo'shimcha (o'zak bir xil)** | 2 263 | **33.7%** |
| **Halqa** | 783 | **11.7%** |
| **Harf chalkashligi (1–2 harf)** | 715 | **10.6%** |
| **Atoqli ot / qisqartma** | 485 | **7.2%** |
| **Gap boshi** | 381 | **5.7%** |
| Son so'zi | 239 | 3.6% |
| Yordamchi so'z | 97 | 1.4% |
| Boshqa (so'z umuman boshqacha eshitilgan, qo'shilib ketgan) | 1 752 | 26.1% |
| **Jami** | **6 715** | 100% |

## Top 5 sabab va misollar

### 1. Qo'shimcha xatosi: 33.7%
Model so'zning o'zagini to'g'ri eshitadi, lekin qo'shimchani adashtiradi. Qoraqalpoq tili agglyutinativ (so'zga ko'p qo'shimcha qo'shiladi), shuning uchun bitta harf ham butun so'zni "xato" qiladi.

| Klip | Haqiqiy | Model |
|---|---|---|
| 26 | ...tısqarı **mámleketimizde** isbilermen... | ...tısqarı **mámleketimizdi** isbilermen... |
| 31 | ...ushırasıwında **mámleketimizde** insan... | ...ushırasıwında **mámleketimizdi** insan... |
| 36 | ...mámleketler **arasındaǵı** ekonomikalıq... | ...mámleketler **arasında** ekonomikalıq... |
| 27 | ...ekspertizası **tarawındaǵı** jańalıqlar... | ...ekspertizası **tarawında** jańalıqlar... |

Kuzatuv: `-de/-di` va `-daǵı/-da` almashuvlari eng ko'p uchraydi. Bular talaffuzda juda yaqin.
Yechim g'oyalari: qoraqalpoq matn korpusidan qurilgan til modeli (LM fusion), ko'proq ma'lumot, yoki BPE o'rniga qo'shimchalarni hisobga oladigan tokenlash.

### 2. Halqa: 11.7% (atigi 10 ta klipdan!)
Model bitta so'zni to'xtamay takrorlaydi. 1 034 klipdan faqat 10 tasi (1%) halqaga tushgan, lekin barcha xatolarning 11.7% ini aynan ular beradi.

| Klip | Haqiqiy | Model |
|---|---|---|
| 19 (WER 325.8%) | sebebi mektepke shekemgi bilimlendiriw... | qaraqalpaqstan xalqan **qalqan qalqan qalqan** ... |
| 531 (WER 381.0%) | press tur dawamında ǵxq wákilleri... | **qorqat qorqat qorqat** ... |
| 580 (WER 250.0%) | twpalang suw saqlaǵıshınıń... | **qaraqalpaqa da qaraqalpaqa da** ... |

Yechim: `no_repeat_ngram_size=4` halqalarni 0 ga tushiradi, lekin haqiqiy takrorlarni ham buzadi (test, klip 396) va umumiy WER'ni oshiradi (validation: 20.07% → 21.64%). Yaxshiroq yo'llar: `repetition_penalty`, kattaroq `no_repeat` (6), yoki halqani aniqlab, shu klipni greedy bilan qayta yozish.

### 3. Harf chalkashligi (1–2 harf): 10.6%
O'xshash tovushlar adashtiriladi: `ǵ↔q`, `á↔a`, `e↔i`, `k↔q`, `ń↔m`.

| Klip | Haqiqiy | Model |
|---|---|---|
| 171 | ...siyasatın túpkilikli **jetilistiriw** hám... | ...siyasatın túpkilikli **jetistiriw** hám... |
| 59 | ...xalıqaralıq strukturalar **sheńberinde** bir... | ...xalıqaralıq strukturalar **shemberinde** bir... |
| 137 | ...shamalas jasıl **baǵ** hám jasıl... | ...shamalas jasıl **baq** hám jasıl... |

Yechim: ko'proq ma'lumot, til modeli (LM fusion), kattaroq model (whisper-medium).

### 4. Atoqli ot va qisqartmalar: 7.2%
Ismlar, joy nomlari va qisqartmalar. Ularning bir qismi **modelning emas, reference matnning muammosi**.

| Klip | Asl matn | Haqiqiy (`matn_model`) | Model |
|---|---|---|---|
| 512 | Sonday-aq, **Sh.** Mirziyoev... | ...sonday aq **sh** mirziyoev... | ...sonday aq **shavkat** mirziyoev... |
| 223 | CHexiyada **MSV**-eki mıń... | ...chexiyada **msv** eki mıń... | ...texiyada **massivi** eki mıń... |
| 214 | ...hár ay **BEMniń**... | ...hár ay **bemniń** eki esesi... | ...hár ay **bmniń** eki ese... |
| 472 | ...Ózbekstan hám **Ázerbayjan**... | ...ózbekstan hám **ázerbayjan**... | ...ózbekstan hám **ázarbayjan**... |

Kuzatuv: 512-klipda diktor ehtimol "Shavkat" deb to'liq o'qigan, ya'ni model to'g'ri yozgan, reference esa qisqartma. Buni tinglab tasdiqlash kerak.
Yechim: qisqartmalarni normalizatsiyada qanday o'qilishi bo'yicha yozish (`sh` → `shavkat`), atoqli otlar ro'yxati bilan til modeli.

### 5. Gap boshi: 5.7%
Halqasiz 1 024 klipdan **335 tasida (33%)** birinchi so'z xato. Model gap boshiga yo'q so'z qo'shadi (`qaraqalpaqta`, `qaraqalpaqstan`) yoki birinchi so'zga ortiqcha tovush yopishtiradi (`atap` → `qatap`).

| Klip | Haqiqiy | Model |
|---|---|---|
| 49 | **ilaj** juwmaǵında ózatom agentliginiń... | **qaraqalpaqta bul** jumanda ózi atamagentliginiń... |
| 60 | **issaǵa** aǵza eliw jeti... | **qaraqalpaqta ǵizaga** aǵza eliw jeti... |
| 100 | **atap** ótilgenindey ótken dáwirde... | **qatap** ótilgenindey ótken dáwirde... |
| 400 | **atap** ótiliwinshe konstituciyanıń... | **qatap** ótiwinshe konstituciyanıń... |

**Gipoteza "trim nutq boshini kesgan" RAD ETILDI:** boshi kesilgan kliplarda birinchi so'z xatosi 234/755 = **31.0%**, kesilmaganlarda esa 101/269 = **37.5%**. Ya'ni kesilmagan kliplarda xato ko'proq.
Yangi gipoteza: model gap boshida "kontekst"siz qoladi va treningda ko'p ko'rgan so'zlarni (`qaraqalpaq...`) "taxmin qilib" qo'yadi. Tekshirish usuli: decoder'ga `prompt` berish yoki boshiga qisqa jimlik qo'shib ko'rish.

## Qo'shimcha kuzatuvlar

**Son so'zlari (3.6%):** `jeti` → `jetinshi` (klip 281, 572), `bir` → `birinshi` (12), `eki` → `ekinshi`. Asl matnda `jeti-avgustta` deb yozilgan, diktor esa ehtimol "jetinshi avgustta" deb o'qigan. Bu ham reference (normalizatsiya) muammosi bo'lishi mumkin, tinglab tekshirish kerak.

**Audio sifati:** SNR < 15 dB bo'lgan kliplarda klip bo'yicha o'rtacha WER **24.2%**, 25–35 dB da **16.3%**. Shovqin xatoni oshiradi, lekin asosiy sabab emas (halqasiz kliplar bo'yicha; `02_audio_features/audio_features_silero.csv`).

**Yordamchi so'zlar (1.4%):** `da` (29 marta) va `de` (15) tushib qoladi, `hám` (17) ortiqcha qo'shiladi.

## Xulosa

1. Eng katta muammo **qo'shimchalar** (xatolarning uchdan biri). Bu qoraqalpoq tilining tuzilishi bilan bog'liq va til modeli yoki ko'proq ma'lumot bilan yaxshilanadi.
2. **Halqalar** juda kam klipda (1%), lekin xatolarning 11.7% i. Ularni dekodlash sozlamasi bilan qayta o'qitmasdan kamaytirish mumkin.
3. Xatolarning bir qismi (qisqartmalar, sonlar) **reference matndagi nomuvofiqlik**. Ularni tinglab tekshirmasdan model aybdor deb bo'lmaydi.
4. "Trim gap boshini kesgan" gipotezasi raqamlar bilan **rad etildi**.

## Cheklovlar
- Har bir xato faqat bitta sababga biriktirildi. Qoidalar tartibi natijaga ta'sir qiladi: masalan, gap boshidagi qo'shimcha xatosi "gap boshi" deb sanalgan.
- "Atoqli ot" faqat asl matn bilan so'z soni mos kelgan kliplarda aniqlangan.
- 512, 281, 572-kliplardagi "reference xatosi" gipotezasi hali tinglab tasdiqlanmagan.

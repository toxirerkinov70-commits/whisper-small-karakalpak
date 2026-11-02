# Seshanba vazifasi: ma'lumot qismida LoRA va to'liq fine-tuning'ni solishtirish
# Ishga tushirish (08_lora ichida, GPU'da):  python -u lora_vs_full.py full   (yoki lora)
# Ikkala run aynan bir xil: 3 000 train klip, 3 epoch, effektiv batch 16, bf16, seed 42.
# Farqi faqat: o'qitiladigan parametrlar va learning rate.
# Baholash 07_eval bilan bir xil: 200 val klip, beam5 (no_repeat'siz), fp32, batch 1, max_new_tokens 440,
# model matni normalize.kichik_harf bilan tozalanadi.
import json
import os
import sys
import time
from dataclasses import dataclass

import jiwer
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    logging,
)

logging.set_verbosity_error()

ILDIZ = ".."                                   # repo ildizi: DATASET_trimmed/ shu yerda
sys.path.insert(0, f"{ILDIZ}/04_normalize")
import normalize                               # noqa: E402

REJIM = sys.argv[1]
assert REJIM in ("full", "lora")
SINOV = len(sys.argv) > 2 and sys.argv[2] == "sinov"   # tez tekshiruv: 32 train, 4 val klip
MODEL = "openai/whisper-small"
TIL = "uz"
CHIQISH = f"./{REJIM}" + ("_sinov" if SINOV else "")
TRAIN_KLIP = 32 if SINOV else 3000
VAL_KLIP = 4 if SINOV else 200
LR = {"full": 1e-5, "lora": 1e-3}[REJIM]

# 1. Ma'lumot qismi: train'dan 3 000, validation'dan 200 klip (ikkala run uchun bir xil)
df = pd.read_csv(f"{ILDIZ}/04_normalize/manifest_tayyor.csv")
df["fayl"] = ILDIZ + "/" + df["fayl"].str.replace("\\", "/", regex=False)
train = df[df["split"] == "train"].sample(TRAIN_KLIP, random_state=42).reset_index(drop=True)
val = df[df["split"] == "validation"].sample(VAL_KLIP, random_state=42).reset_index(drop=True)
print(f"[{REJIM}] train {len(train)} klip, {train['yangi_s'].sum() / 3600:.2f} soat | val {len(val)} klip", flush=True)
train[["idx"]].to_csv(f"{CHIQISH}_train_idx.csv", index=False)
val[["idx"]].to_csv(f"{CHIQISH}_val_idx.csv", index=False)

ds = Dataset.from_dict({"fayl": train["fayl"].tolist(), "matn_model": train["matn_model"].tolist()})
ds = ds.cast_column("fayl", Audio(sampling_rate=16000))

# 2. Processor va model (train_whisper.py bilan bir xil)
processor = WhisperProcessor.from_pretrained(MODEL, language=TIL, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
model.generation_config.language = TIL
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

if REJIM == "lora":
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"))
oqitiladigan = sum(p.numel() for p in model.parameters() if p.requires_grad)
jami = sum(p.numel() for p in model.parameters())
print(f"[{REJIM}] o'qitiladigan parametr: {oqitiladigan:,} / {jami:,} = {100 * oqitiladigan / jami:.2f}%", flush=True)


# 3. Audio -> log-mel, matn -> tokenlar
def tayyorla(batch):
    audio = batch["fayl"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["matn_model"]).input_ids
    return batch


boshlanish = time.time()
ds = ds.map(tayyorla, remove_columns=["fayl", "matn_model"], num_proc=8)
ds = ds.filter(lambda x: len(x) <= 448, input_columns="labels")
print(f"[{REJIM}] ma'lumot tayyor: {len(ds)} klip, {(time.time() - boshlanish) / 60:.1f} daq", flush=True)


@dataclass
class Collator:
    processor: WhisperProcessor
    start_id: int

    def __call__(self, features):
        kirish = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        yorliq = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = yorliq["input_ids"].masked_fill(yorliq["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.start_id).all():
            labels = labels[:, 1:]
        kirish["labels"] = labels
        return kirish


# 4. Trening: batch 8 x 2 = effektiv 16 (umumiy xotira tejaladi), warmup ~9% (asl run: 500/5840)
args = Seq2SeqTrainingArguments(
    output_dir=CHIQISH,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    warmup_steps=50,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=False,
    eval_strategy="no",
    save_strategy="no",
    logging_steps=25,
    dataloader_num_workers=4,
    remove_unused_columns=False,   # PeftModel forward imzosi boshqa, input_features o'chib ketmasin
    label_names=["labels"],
    seed=42,
    report_to=["tensorboard"],
)
trainer = Seq2SeqTrainer(
    model=model, args=args, train_dataset=ds,
    data_collator=Collator(processor, model.config.decoder_start_token_id),
    processing_class=processor.feature_extractor,
)

torch.cuda.reset_peak_memory_stats()
boshlanish = time.time()
natija_train = trainer.train()
trening_daq = (time.time() - boshlanish) / 60
xotira_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
print(f"[{REJIM}] trening tugadi: {trening_daq:.1f} daq, eng yuqori GPU xotira {xotira_gb:.1f} GB, "
      f"train loss {natija_train.training_loss:.4f}", flush=True)

# 5. Saqlash va hajmini o'lchash (LoRA: faqat adapter)
trainer.save_model(f"{CHIQISH}/final")
hajm_mb = sum(os.path.getsize(f"{CHIQISH}/final/{f}") for f in os.listdir(f"{CHIQISH}/final")
              if f.endswith((".safetensors", ".bin")) and f != "training_args.bin") / 1024 ** 2
print(f"[{REJIM}] saqlangan og'irliklar: {hajm_mb:.1f} MB", flush=True)

# 6. Baholash (07_eval bilan bir xil sharoit). LoRA adapter asosiy modelga qo'shib yuboriladi
model = trainer.model
if REJIM == "lora":
    model = model.merge_and_unload()
model = model.float().cuda().eval()
model.config.use_cache = True

boshlanish = time.time()
satrlar = []
for n, r in val.iterrows():
    wav, sr = sf.read(r["fayl"], dtype="float32")
    x = processor(wav, sampling_rate=sr, return_tensors="pt").input_features.cuda()
    with torch.no_grad():
        out = model.generate(x, language=TIL, task="transcribe", max_new_tokens=440, num_beams=5, do_sample=False)
    hyp = normalize.kichik_harf(processor.batch_decode(out, skip_special_tokens=True)[0])
    ref = r["matn_model"]
    satrlar.append({
        "idx": r["idx"], "davomiylik_s": r["yangi_s"], "reference": ref, "hypothesis": hyp,
        "wer": round(100 * jiwer.wer(ref, hyp), 2), "cer": round(100 * jiwer.cer(ref, hyp), 2),
    })
    if (n + 1) % 50 == 0:
        print(f"[{REJIM}] baholash {n + 1}/{len(val)}, {(time.time() - boshlanish) / 60:.1f} daq", flush=True)

per = pd.DataFrame(satrlar)
per.to_csv(f"{CHIQISH}_val200.csv", index=False)
ref, hyp = per["reference"].tolist(), per["hypothesis"].tolist()
xulosa = {
    "rejim": REJIM, "train_klip": len(ds), "epoch": 3, "effektiv_batch": 16, "learning_rate": LR,
    "oqitiladigan_parametr": oqitiladigan, "jami_parametr": jami,
    "oqitiladigan_foiz": round(100 * oqitiladigan / jami, 2),
    "trening_daqiqa": round(trening_daq, 1), "gpu_xotira_gb": round(xotira_gb, 1),
    "saqlangan_mb": round(hajm_mb, 1), "train_loss": round(natija_train.training_loss, 4),
    "val200_WER": round(100 * jiwer.wer(ref, hyp), 2), "val200_CER": round(100 * jiwer.cer(ref, hyp), 2),
    "halqa_klip": sum(len(h.split()) > 1.5 * len(r.split()) for r, h in zip(ref, hyp)),
    "baholash_daqiqa": round((time.time() - boshlanish) / 60, 1),
}
json.dump(xulosa, open(f"{CHIQISH}_natija.json", "w"), ensure_ascii=False, indent=1)
print(xulosa, flush=True)
print("TAYYOR", flush=True)

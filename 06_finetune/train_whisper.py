# Whisper small'ni qoraqalpoq tiliga fine-tuning qilish (DGX Spark, GB10 da 2 soat 33 daqiqa)
# Ishga tushirish (06_finetune ichida, GPU'da):  python train_whisper.py
# Kerak: ../04_normalize/manifest_tayyor.csv va ../DATASET_trimmed/ (03_trim yaratadi)
from dataclasses import dataclass

import jiwer
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL = "openai/whisper-small"
TIL = "uz"  # qoraqalpoq Whisper'da yo'q; lotin yozuvi uchun eng yaqini - uz
CHIQISH = "./whisper-small-kaa"

# 1. Manifest -> HF dataset
df = pd.read_csv("../04_normalize/manifest_tayyor.csv")
df["fayl"] = "../" + df["fayl"].str.replace("\\", "/", regex=False)  # yo'llar repo ildiziga nisbatan; Windows -> Linux

# from_pandas emas: pandas 3.x matnni large_string qiladi, Audio ustuni uni qabul qilmaydi
ds = DatasetDict({
    split: Dataset.from_dict({
        "fayl": df.loc[df["split"] == split, "fayl"].tolist(),
        "matn_model": df.loc[df["split"] == split, "matn_model"].tolist(),
    })
    for split in ["train", "validation", "test"]
})
ds = ds.cast_column("fayl", Audio(sampling_rate=16000))

# 2. Processor va model
processor = WhisperProcessor.from_pretrained(MODEL, language=TIL, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
model.generation_config.language = TIL
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False  # o'qitishda kesh kerak emas


# 3. Audio -> log-mel, matn -> tokenlar
def tayyorla(batch):
    audio = batch["fayl"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["matn_model"]).input_ids
    return batch


ds = ds.map(tayyorla, remove_columns=["fayl", "matn_model"], num_proc=8)
ds = ds.filter(lambda x: len(x) <= 448, input_columns="labels")  # Whisper decoder chegarasi


# 4. Batch yig'uvchi: audio va matnni alohida padding qiladi
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
        if (labels[:, 0] == self.start_id).all():  # <|startoftranscript|> ni model o'zi qo'shadi
            labels = labels[:, 1:]
        kirish["labels"] = labels
        return kirish


collator = Collator(processor, model.config.decoder_start_token_id)


# 5. Metrika: WER va CER
def metrika(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    taxmin = processor.batch_decode(pred_ids, skip_special_tokens=True)
    haqiqiy = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * jiwer.wer(haqiqiy, taxmin), "cer": 100 * jiwer.cer(haqiqiy, taxmin)}


# 6. Trening sozlamalari
args = Seq2SeqTrainingArguments(
    output_dir=CHIQISH,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=5,
    bf16=True,
    gradient_checkpointing=False,  # small kichik, xotira yetadi; torch 2.14 bilan checkpointing backward xatosi berdi
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=448,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=4,
    report_to=["tensorboard"],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
    compute_metrics=metrika,
    processing_class=processor.feature_extractor,
)

trainer.train()

# 7. Yakuniy baho test'da va saqlash
print("TEST:", trainer.evaluate(ds["test"], metric_key_prefix="test"))
trainer.save_model(CHIQISH + "/final")
processor.save_pretrained(CHIQISH + "/final")
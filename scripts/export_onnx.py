"""Modelni transformers.js (brauzer demo) uchun ONNX'ga eksport qilish va kvantlash.

MUHIM: eksport alohida, eski muhitda qilinadi. transformers >= 4.43 bilan optimum Whisper decoder'iga
`cache_position` kirishini qo'shadi, transformers.js esa uni bermaydi ("Missing the following inputs: cache_position").
Kirishni shunchaki olib tashlash ham yaramaydi: pozitsiya grafikda konstanta bo'lib qotib, model halqaga tushadi.
transformers 4.42 da pozitsiya o'tgan kesh uzunligidan dinamik olinadi, shuning uchun to'g'ri ishlaydi.

Muhit (tekshirilgan):
    python -m venv venv_onnx
    venv_onnx/Scripts/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
    venv_onnx/Scripts/pip install transformers==4.42.4 "optimum[exporters]==1.21.4" "numpy<2" onnx==1.16.2 onnxruntime==1.19.2
    venv_onnx/Scripts/pip uninstall -y torchvision timm        # optimum torch'ni yangilab yuborsa: torch'ni 2.4.1 ga qaytaring

Ishga tushirish:
    venv_onnx/Scripts/python scripts/export_onnx.py Toxirerkinovtohir/whisper-small-karakalpak onnx_out

Kvantlash tanlovi (transformers.js orqali tekshirilgan):
  * encoder + decoder per-tensor 8-bit: 40 test klipida CER 8.84% -> 15.60% (yaroqsiz)
  * encoder fp32 + decoder per-tensor 8-bit: uzun audioda oxirgi bo'lak tushib qoldi
  * encoder fp32 + decoder 4-bit blokli (MatMulNBits, block 32): namunalar va 30 s uzun audioda PyTorch'ga deyarli aynan mos
    -> demo shu variantda (~610 MB)
"""
import sys

from optimum.exporters.onnx import main_export


def main():
    model, out = sys.argv[1], sys.argv[2]
    main_export(model, output=out, task="automatic-speech-recognition-with-past")

    # 4-bit decoder (MatMulNBits). Demo fayli onnxruntime 1.31 bilan kvantlangan; eski versiyada modul nomi boshqacha
    import onnx
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Kvant
    except ImportError:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Kvant

    q = Kvant(onnx.load(f"{out}/decoder_model_merged.onnx"), block_size=32, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(f"{out}/decoder_model_merged_q4.onnx", use_external_data_format=False)
    # transformers.js nomlash qoidasi: HF repoda onnx/encoder_model.onnx va onnx/decoder_model_merged_q4.onnx
    print("tayyor:", out)


if __name__ == "__main__":
    main()

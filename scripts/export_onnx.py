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
  * encoder fp32 + decoder 4-bit blokli (MatMulNBits, block 32): namunalar va 30 s uzun audioda PyTorch'ga deyarli aynan mos (~610 MB)
  * encoder ham 4-bit blokli + decoder'dagi token embedding fp16: yuqoridagi bilan bir xil matn (3 namuna va 30 s audio)
    -> demo shu variantda (~245 MB: encoder 66 MB + decoder 178 MB), telefon xotirasiga sig'adi
  * embedding'ni 4-bit qilish (GatherBlockQuantized) brauzerdagi onnxruntime-web'da ishlamaydi
"""
import sys

from optimum.exporters.onnx import main_export


def embedding_fp16(model):
    """Gather og'irligi bo'lgan katta embedding (51865 x 768, fp32 ~160 MB) ni fp16 ga o'tkazadi: Gather -> Cast(float32)."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    def gather_w(g, out):
        for node in g.node:
            if node.op_type == "Gather":
                out.add(node.input[0])
            for a in node.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    gather_w(a.g, out)
        return out

    gw = gather_w(model.graph, set())
    katta = {i.name: i for i in model.graph.initializer
             if i.name in gw and len(i.dims) == 2 and i.dims[0] > 50000 and i.data_type == onnx.TensorProto.FLOAT}
    for nom, init in katta.items():
        init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), nom))

    def yur(g):                                   # merged decoder: Gather'lar If tarmoqlari ichida ham bor
        yangi = []
        for node in g.node:
            for a in node.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    yur(a.g)
            yangi.append(node)
            if node.op_type == "Gather" and node.input[0] in katta:
                asl = node.output[0]
                node.output[0] = asl + "_fp16"
                yangi.append(helper.make_node("Cast", [node.output[0]], [asl], to=onnx.TensorProto.FLOAT, name=node.name + "_cast"))
        del g.node[:]
        g.node.extend(yangi)

    yur(model.graph)
    return model


def main():
    model, out = sys.argv[1], sys.argv[2]
    main_export(model, output=out, task="automatic-speech-recognition-with-past")

    # 4-bit (MatMulNBits, block 32). Demo fayllari onnxruntime 1.31 bilan kvantlangan; eski versiyada modul nomi boshqacha
    import onnx
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Kvant
    except ImportError:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Kvant

    def kvant(src, dst, fp16_emb=False):
        q = Kvant(onnx.load(src), block_size=32, is_symmetric=True)
        q.process()
        m = q.model.model
        if fp16_emb:
            m = embedding_fp16(m)
        onnx.save(m, dst)

    kvant(f"{out}/encoder_model.onnx", f"{out}/encoder_model_q4.onnx")
    kvant(f"{out}/decoder_model_merged.onnx", f"{out}/decoder_model_merged_q4.onnx", fp16_emb=True)
    # transformers.js nomlash qoidasi: HF repoda onnx/encoder_model_q4.onnx va onnx/decoder_model_merged_q4.onnx,
    # sahifada dtype: { encoder_model: "q4", decoder_model_merged: "q4" }
    print("tayyor:", out)


if __name__ == "__main__":
    main()

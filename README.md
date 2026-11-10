# Dịch máy Anh - Việt trên IWSLT 2015

Repo này cài đặt lại hai cách dịch máy để có thể so sánh trên **cùng một bộ dữ liệu**:

1. **Transformer** – theo hướng của notebook `demo_transformer.ipynb`.
2. **Seq2Seq + Luong Attention** – theo ý tưởng của dự án [tensorflow/nmt](https://github.com/tensorflow/nmt), nhưng viết lại bằng PyTorch hiện đại vì mã gốc dùng TensorFlow 1.x và đã được lưu trữ.

Mục tiêu là dịch **English → Vietnamese**. Cả hai mô hình dùng chung IWSLT 2015, cùng cách tách dữ liệu, vocabulary, batch và cách tính BLEU.

## Chạy nhanh trên Google Colab

Mở một trong hai notebook:

- `notebooks/01_transformer_colab.ipynb`
- `notebooks/02_seq2seq_colab.ipynb`

Chọn GPU trong Colab: **Runtime → Change runtime type → T4 GPU**, sau đó chạy lần lượt các ô.

## Chạy bằng dòng lệnh

```bash
pip install -r requirements.txt
python -m nmt.data --data-dir data

# Transformer
python -m nmt.train --model transformer --data-dir data --output-dir outputs/transformer

# Seq2Seq + attention
python -m nmt.train --model seq2seq --data-dir data --output-dir outputs/seq2seq

# So sánh hai checkpoint trên tst2013
python -m nmt.compare \
  --transformer outputs/transformer/best.pt \
  --seq2seq outputs/seq2seq/best.pt \
  --data-dir data \
  --output outputs/comparison.json
```

Trên PowerShell, thay dấu `\` cuối dòng bằng dấu backtick hoặc viết lệnh trên một dòng.

## Kiểm tra code mà không cần train lâu

```bash
python -m pytest -q
python -m nmt.train --model transformer --data-dir data --output-dir outputs/smoke-transformer --epochs 1 --max-train-batches 2 --max-eval-batches 1 --small
python -m nmt.train --model seq2seq --data-dir data --output-dir outputs/smoke-seq2seq --epochs 1 --max-train-batches 2 --max-eval-batches 1 --small
```

Smoke test chỉ kiểm tra pipeline có chạy; BLEU của nó **không có ý nghĩa**.

## Kết quả so sánh

Xem [docs/so_sanh_ket_qua.md](docs/so_sanh_ket_qua.md). File này nói rõ phần nào đã kiểm tra được, số tham khảo từ nguồn gốc, và bảng kết quả thật cần train để điền.

## Cấu trúc chính

```text
nmt/
  data.py          tải và chuẩn bị IWSLT 2015
  models.py        Transformer và Seq2Seq + Luong Attention
  train.py         train, validation, lưu checkpoint
  compare.py       chạy test chung và xuất bảng so sánh
notebooks/         hai notebook chạy trên Colab
tests/             kiểm tra shape, mask và một bước forward
docs/              giải thích kết quả bằng tiếng Việt
```

## Nguồn tham khảo

- TensorFlow NMT: <https://github.com/tensorflow/nmt>
- IWSLT 2015 đã tiền xử lý: <https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/>

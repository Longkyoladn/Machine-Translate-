# Dịch máy Anh - Việt với IWSLT 2015

Bài tập cài đặt và so sánh hai mô hình dịch từ tiếng Anh sang tiếng Việt:

1. Transformer, dựa trên nội dung của `demo_transformer.ipynb`.
2. Seq2Seq + Luong Attention, tham khảo từ [tensorflow/nmt](https://github.com/tensorflow/nmt) và viết lại bằng PyTorch.

Hai mô hình dùng cùng bộ dữ liệu, cách tiền xử lý và cách tính BLEU.

## Notebook

- `notebooks/01_transformer_colab.ipynb`
- `notebooks/02_seq2seq_colab.ipynb`

Các thí nghiệm được chạy với GPU Colab.

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

## Kiểm tra nhanh

```bash
python -m pytest -q
python -m nmt.train --model transformer --data-dir data --output-dir outputs/smoke-transformer --epochs 1 --max-train-batches 2 --max-eval-batches 1 --small
python -m nmt.train --model seq2seq --data-dir data --output-dir outputs/smoke-seq2seq --epochs 1 --max-train-batches 2 --max-eval-batches 1 --small
```

Hai lệnh trên chỉ chạy vài batch để kiểm tra chương trình, không dùng kết quả này để so sánh mô hình.

## Kết quả so sánh

Kết quả và nhận xét được ghi tại [docs/so_sanh_ket_qua.md](docs/so_sanh_ket_qua.md).

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

# So sánh Transformer và Seq2Seq

## Thiết lập

Transformer sử dụng self-attention, còn Seq2Seq sử dụng GRU hai chiều và Luong Attention. Hai mô hình được train và đánh giá trong cùng điều kiện:

| Phần | Dùng chung cho hai mô hình |
|---|---|
| Dữ liệu train | `train.en` / `train.vi` |
| Validation | `tst2012.en` / `tst2012.vi` |
| Test cuối | `tst2013.en` / `tst2013.vi` |
| Chiều dịch | English → Vietnamese |
| Tiền xử lý | chữ thường, tokenizer và vocabulary giống nhau |
| Đánh giá | SacreBLEU, `tokenize=none`, greedy decoding |
| Seed mặc định | 42 |

## Kết quả

Bảng này được cập nhật sau khi train xong cả hai mô hình trên Colab.

| Mô hình | BLEU tst2013 | Tốc độ dịch | Ghi chú |
|---|---:|---:|---|
| Seq2Seq + Luong attention | Chưa train | Chưa đo | Cần checkpoint |
| Transformer | Chưa train | Chưa đo | Cần checkpoint |

Kết quả chi tiết, gồm các câu dịch mẫu, được lưu trong `outputs/comparison.json`.

## Kết quả tham khảo

Dự án TensorFlow NMT công bố trên IWSLT English–Vietnamese: 25.5 BLEU với greedy và 26.1 với beam search trên `tst2013`. Một thử nghiệm Transformer Base công khai trên cùng bộ IWSLT 2015 báo cáo 28.54 BLEU (cased) sau 50.000 bước train.

Đây không phải kết quả chạy từ repo này vì cấu hình tiền xử lý và cách decode có thể khác.

## Nhận xét

Phần này sẽ được viết dựa trên BLEU, tốc độ dịch và các câu dịch mẫu sau khi hai lần train hoàn tất.

## Tài liệu tham khảo

- TensorFlow NMT: <https://github.com/tensorflow/nmt#iwslt-english-vietnamese>
- Transformer English–Vietnamese: <https://github.com/stefan-it/nmt-en-vi>

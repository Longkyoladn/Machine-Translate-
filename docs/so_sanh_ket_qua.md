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

Hai mô hình đều được train đủ 15 epoch trên cùng GPU Colab. BLEU dưới đây được tính trên tập validation `tst2012` bằng greedy decoding.

| Mô hình | Số tham số | BLEU tst2012 | Valid loss | Perplexity | Giây/epoch | Tổng thời gian train |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 15.44M | 8.76 | 2.854 | 17.35 | 214.9 | 53 phút 43 giây |
| Seq2Seq + Luong Attention | 19.39M | 13.82 | 2.571 | 13.08 | 549.6 | 2 giờ 17 phút 24 giây |

Số liệu được lấy trực tiếp từ output của hai notebook. Bản tổng hợp dạng JSON nằm tại `results/comparison_from_notebooks.json`.

## Nhận xét

Seq2Seq đạt BLEU cao hơn Transformer 5.06 điểm. Valid loss và perplexity của Seq2Seq cũng thấp hơn, cho thấy mô hình khớp tập validation tốt hơn với cấu hình hiện tại.

Đổi lại, Seq2Seq chậm hơn khá nhiều. Trung bình một epoch mất 549.6 giây, gấp khoảng 2.56 lần Transformer. Seq2Seq cũng có nhiều hơn khoảng 3.95 triệu tham số, tương đương 25.6%.

Với bộ tham số đang dùng, Seq2Seq cho chất lượng validation tốt hơn còn Transformer có lợi thế rõ ràng về thời gian train. Kết quả này chỉ áp dụng cho lần chạy hiện tại; nó không có nghĩa Seq2Seq luôn tốt hơn Transformer ở mọi cấu hình.

## Giới hạn của kết quả

Checkpoint `best.pt` của Transformer đã được lưu, nhưng checkpoint Seq2Seq không còn. Vì vậy không thể chạy lại `nmt.compare` trên `tst2013`, đo tốc độ sinh câu hoặc lấy cùng một nhóm câu dịch mẫu. Các mục này được để trống thay vì tạo số liệu giả.

BLEU 8.76 và 13.82 là BLEU validation trên `tst2012`, không phải BLEU test trên `tst2013`. Do khác tập đánh giá và cấu hình decode, không nên so trực tiếp hai số này với kết quả tham khảo bên dưới.

## Kết quả tham khảo

Dự án TensorFlow NMT công bố trên IWSLT English–Vietnamese: 25.5 BLEU với greedy và 26.1 với beam search trên `tst2013`. Một thử nghiệm Transformer Base công khai trên cùng bộ IWSLT 2015 báo cáo 28.54 BLEU (cased) sau 50.000 bước train.

Đây không phải kết quả chạy từ repo này vì cấu hình tiền xử lý và cách decode có thể khác.

## Tài liệu tham khảo

- TensorFlow NMT: <https://github.com/tensorflow/nmt#iwslt-english-vietnamese>
- Transformer English–Vietnamese: <https://github.com/stefan-it/nmt-en-vi>

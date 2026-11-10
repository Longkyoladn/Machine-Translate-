# So sánh kết quả Transformer và Seq2Seq

## Cần hiểu ngắn gọn

Hai mô hình đều học từ cùng các cặp câu Anh - Việt. Điểm khác nhau nằm ở cách chúng xử lý câu:

- **Seq2Seq + attention** đọc câu theo thứ tự bằng GRU. Cách này dễ hiểu, gần với dự án `tensorflow/nmt`, nhưng train chậm hơn vì các bước phụ thuộc nhau.
- **Transformer** dùng self-attention để nhìn toàn bộ câu cùng lúc. Nó thường tận dụng GPU tốt hơn và dịch câu dài ổn hơn.

## Điều kiện so sánh trong repo

| Phần | Dùng chung cho hai mô hình |
|---|---|
| Dữ liệu train | `train.en` / `train.vi` |
| Validation | `tst2012.en` / `tst2012.vi` |
| Test cuối | `tst2013.en` / `tst2013.vi` |
| Chiều dịch | English → Vietnamese |
| Tiền xử lý | chữ thường, tokenizer và vocabulary giống nhau |
| Đánh giá | SacreBLEU, `tokenize=none`, greedy decoding |
| Seed mặc định | 42 |

Việc dùng cùng điều kiện quan trọng hơn việc cố làm cho hai mô hình có số tham số giống tuyệt đối.

## Kết quả của lần chạy này

Hiện chưa có checkpoint train hoàn chỉnh, vì vậy **chưa có BLEU thực nghiệm của repo này**. Sau khi train xong, chạy `python -m nmt.compare ...`; kết quả thật sẽ nằm trong `outputs/comparison.json`.

| Mô hình | BLEU tst2013 | Tốc độ dịch | Ghi chú |
|---|---:|---:|---|
| Seq2Seq + Luong attention | Chưa train | Chưa đo | Cần checkpoint |
| Transformer | Chưa train | Chưa đo | Cần checkpoint |

Không nên dùng BLEU của smoke test để kết luận vì smoke test chỉ chạy vài batch.

## Số tham khảo, không phải kết quả của repo này

Dự án TensorFlow NMT công bố trên IWSLT English–Vietnamese: 25.5 BLEU với greedy và 26.1 với beam search trên `tst2013`. Một thử nghiệm Transformer Base công khai trên cùng bộ IWSLT 2015 báo cáo 28.54 BLEU (cased) sau 50.000 bước train.

Các số này chỉ giúp đặt kỳ vọng. Không so trực tiếp như một bảng xếp hạng vì cấu hình tokenizer, chiều dịch, cách decode và cách tính BLEU có thể khác.

## Sau khi train, nên đọc bảng thế nào?

- BLEU cao hơn thường có nghĩa câu dịch gần câu tham chiếu hơn trên toàn tập test.
- Chênh dưới khoảng 1 BLEU có thể thay đổi giữa các lần train; nên train thêm seed nếu cần kết luận chắc.
- Tốc độ dịch chỉ so được khi chạy cùng GPU, cùng batch size và cùng greedy decoding.
- Nên đọc thêm 10 câu ví dụ trong `comparison.json`: BLEU không cho biết lỗi cụ thể như thiếu chủ ngữ, sai tên riêng hoặc câu Việt chưa tự nhiên.

## Dự đoán hợp lý trước khi train

Transformer có khả năng đạt BLEU cao hơn và train nhanh hơn trên GPU. Seq2Seq có thể nhẹ hơn trong cấu hình nhỏ, nhưng xử lý tuần tự nên mất thời gian hơn. Đây mới là dự đoán từ kiến trúc, không phải kết quả thực nghiệm của repo.

## Vì sao cần train?

Code chỉ định nghĩa cách mô hình học. Trước khi train, trọng số là số ngẫu nhiên nên câu dịch gần như vô nghĩa. Muốn điền BLEU và ví dụ dịch thật, phải train cả hai mô hình trên `train`, chọn checkpoint tốt nhất bằng `tst2012`, rồi chỉ đánh giá một lần trên `tst2013`.

## Nguồn của số tham khảo

- TensorFlow NMT: <https://github.com/tensorflow/nmt#iwslt-english-vietnamese>
- Transformer English–Vietnamese: <https://github.com/stefan-it/nmt-en-vi>

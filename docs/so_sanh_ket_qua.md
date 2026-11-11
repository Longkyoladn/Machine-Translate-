# So sánh Transformer và Seq2Seq

## 1. Cách thực hiện

Hai mô hình cùng dịch từ tiếng Anh sang tiếng Việt trên IWSLT 2015. Phần train dùng `train.en` và `train.vi`, validation dùng `tst2012`, còn kết quả cuối được đo trên `tst2013`.

| Thiết lập | Giá trị |
|---|---|
| Vocabulary tiếng Anh | 15.000 token |
| Vocabulary tiếng Việt | 11.835 token |
| Batch size | 64 |
| Số epoch | 15 |
| Độ dài tối đa | 80 token |
| Optimizer | AdamW, learning rate 0,0003 |
| Cách sinh bản dịch | Greedy decoding |
| Cách tính BLEU | SacreBLEU, `tokenize=none` |
| Seed | 42 |

Hai checkpoint dùng cùng vocabulary. Khi chạy `nmt.compare`, có 1.243 câu của `tst2013` được đưa vào đánh giá sau bước lọc độ dài. Cùng một chương trình được chạy một lần trên CPU và một lần trên GPU.

## 2. Kết quả train và validation

| Mô hình | Số tham số | BLEU tst2012 | Valid loss | Perplexity | Giây/epoch | Tổng thời gian train |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 15,44M | 8,76 | 2,854 | 17,35 | 214,9 | 53 phút 43 giây |
| Seq2Seq + Luong Attention | 19,39M | 13,82 | 2,571 | 13,08 | 549,6 | 2 giờ 17 phút 24 giây |

Seq2Seq có BLEU validation cao hơn 5,06 điểm, đồng thời loss và perplexity thấp hơn. Tuy nhiên, mỗi epoch của Seq2Seq lâu gấp khoảng 2,56 lần Transformer. Nguyên nhân chính là GRU phải xử lý lần lượt theo từng bước thời gian, trong khi Transformer có thể xử lý nhiều token song song lúc train. Seq2Seq cũng nhiều hơn khoảng 3,95 triệu tham số.

## 3. Kết quả trên tập test

BLEU của CPU và GPU giống nhau vì cả hai lần đều dùng đúng checkpoint, cùng dữ liệu và greedy decoding. Việc đổi phần cứng chỉ làm thay đổi thời gian chạy, không làm thay đổi trọng số mô hình.

| Mô hình | BLEU tst2013 | CPU (câu/giây) | GPU (câu/giây) | GPU nhanh hơn CPU |
|---|---:|---:|---:|---:|
| Transformer | 7,82 | 1,50 | 76,21 | 50,81 lần |
| Seq2Seq + Luong Attention | 13,38 | 25,99 | 504,87 | 19,43 lần |

Trên `tst2013`, Seq2Seq cao hơn Transformer 5,56 BLEU. Seq2Seq cũng sinh nhanh hơn 17,33 lần trên CPU và 6,62 lần trên GPU trong chương trình hiện tại.

## 4. Vì sao Seq2Seq cho kết quả tốt hơn trong lần chạy này?

Kết quả này không có nghĩa Seq2Seq luôn tốt hơn Transformer. Với bài này, có một số nguyên nhân hợp lý:

- Tập train có khoảng 133 nghìn cặp câu, không quá lớn. Seq2Seq với GRU và attention có thể học tốt trên quy mô này sau 15 epoch, trong khi Transformer thường cần thêm bước train và điều chỉnh learning-rate schedule để phát huy tốt hơn.
- Transformer đang dùng cấu hình khá gọn gồm 3 lớp, `d_model=256` và 4 head. Mô hình chỉ dùng AdamW với learning rate cố định, chưa có warm-up như cách train Transformer thường dùng.
- Kết quả train cũng cho thấy Seq2Seq đã khớp dữ liệu tốt hơn: valid loss là 2,571 so với 2,854 và perplexity là 13,08 so với 17,35. BLEU cao hơn vì vậy phù hợp với diễn biến loss, không phải do GPU.
- Cả hai mô hình đều dùng greedy decoding. Beam search có thể cải thiện kết quả, nhưng bài này giữ greedy để điều kiện so sánh giống nhau.

## 5. Vì sao Seq2Seq sinh câu nhanh hơn Transformer trong code này?

Thoạt nhìn kết quả này có vẻ ngược với ưu điểm song song của Transformer. Tuy nhiên, ưu điểm đó thể hiện rõ nhất khi train. Khi dịch bằng greedy decoding, cả hai mô hình vẫn phải sinh từng token một vì token sau phụ thuộc token trước.

Trong phần cài đặt hiện tại, Seq2Seq giữ lại hidden state của GRU và chỉ chạy một `decode_step` cho token mới. Transformer giữ lại kết quả encoder nhưng decoder vẫn phải xử lý lại toàn bộ đoạn đích đã sinh ở mỗi bước. Với mô hình nhỏ, câu tương đối ngắn và batch 64, cách làm của Seq2Seq ít phép tính hơn nên đạt số câu/giây cao hơn.

GPU giúp cả hai mô hình tăng tốc rõ rệt nhờ thực hiện các phép toán tensor theo batch. Transformer hưởng lợi nhiều hơn về tỷ lệ tăng tốc, từ 1,50 lên 76,21 câu/giây. Dù vậy, Seq2Seq vẫn nhanh hơn trong phép đo này vì vòng decode của nó nhẹ hơn.

## 6. Nhìn vào bản dịch thực tế

Ví dụ 1:

- Câu gốc: `My family was not poor, and myself, I had never experienced hunger.`
- Bản tham chiếu: `Gia đình của tôi không nghèo, và bản thân tôi thì chưa từng phải chịu đói.`
- Transformer: `gia đình tôi không phải là người nghèo, và tôi không bao giờ trải qua.`
- Seq2Seq: `gia đình tôi không nghèo, và tôi chưa bao giờ trải qua đói.`

Seq2Seq giữ được ý “không nghèo” và “chưa từng đói”. Transformer làm mất từ chỉ đói nên câu bị thiếu nghĩa.

Ví dụ 2:

- Câu gốc: `Although I often wondered about the outside world, I thought I would spend my entire life in North Korea, until everything suddenly changed.`
- Transformer: `mặc dù tôi thường nghĩ về thế giới bên ngoài, tôi nghĩ rằng tôi sẽ dành toàn bộ cuộc sống của mình, cho đến khi mọi thứ đã thay đổi.`
- Seq2Seq: `mặc dù tôi thường tự hỏi về thế giới bên ngoài, tôi nghĩ rằng tôi sẽ dành cả cuộc đời ở bắc triều tiên, cho đến khi mọi thứ đều bất ngờ.`

Seq2Seq giữ được địa danh “Bắc Triều Tiên”, còn Transformer bỏ mất thông tin này. Dù phần cuối của Seq2Seq chưa tự nhiên, nội dung chính vẫn đầy đủ hơn.

Ví dụ 3:

- Câu gốc: `I was so shocked.`
- Bản tham chiếu: `Tôi đã bị sốc.`
- Transformer: `tôi rất sốc.`
- Seq2Seq: `tôi đã rất sốc.`

Với câu ngắn, cả hai mô hình đều dịch đúng ý. Điều này cho thấy chênh lệch rõ hơn thường xuất hiện ở câu dài, có nhiều thông tin hoặc tên riêng.

Hai mô hình vẫn còn các lỗi chung như `&quot;` bị tách thành `& quot;`, lặp từ, dịch sai tên riêng và câu tiếng Việt chưa tự nhiên. Đây là một lý do BLEU còn thấp so với các hệ thống đã được tối ưu kỹ.

## 7. Giới hạn của phép so sánh

- Tốc độ được đo bằng một lượt chạy nên có thể thay đổi theo CPU, loại GPU, tải của Colab và thời điểm chạy.
- Thời gian trong `nmt.compare` gồm greedy decoding, chuyển kết quả về CPU và giải mã token; không gồm thời gian tải checkpoint.
- Hai mô hình mới train 15 epoch và chưa dò nhiều bộ siêu tham số.
- Greedy decoding được dùng để giữ cách đánh giá đơn giản và giống nhau; chưa so sánh beam search.
- BLEU phản ánh mức trùng khớp với bản tham chiếu nhưng không đánh giá đầy đủ độ tự nhiên hoặc việc giữ đúng nghĩa.

## 8. Kết luận

Trong cấu hình của bài này, Seq2Seq + Luong Attention là mô hình cho kết quả tốt hơn: BLEU đạt 13,38 trên `tst2013`, cao hơn 5,56 điểm so với Transformer. Bản dịch mẫu của Seq2Seq cũng thường giữ được nhiều thông tin hơn.

Transformer có lợi thế ở thời gian train, chỉ mất khoảng 214,9 giây mỗi epoch so với 549,6 giây của Seq2Seq. Khi sinh câu bằng chương trình hiện tại, Seq2Seq lại nhanh hơn do decoder tái sử dụng hidden state, còn Transformer phải xử lý lại phần câu đích đã sinh.

GPU không thay đổi chất lượng dịch nhưng rút ngắn đáng kể thời gian đánh giá. Vì vậy, kết luận phù hợp cho lần thử nghiệm này là: Seq2Seq tốt hơn về BLEU và tốc độ sinh câu, còn Transformer nhanh hơn trong quá trình train và còn khả năng cải thiện nếu dùng lịch learning rate, warm-up, train lâu hơn hoặc beam search.

## Tệp kết quả

- Kết quả đầy đủ khi chạy CPU: `outputs/comparison_cpu.json`
- Kết quả đầy đủ khi chạy GPU: `outputs/comparison_gpu.json`
- Bản tổng hợp để lưu trong Git: `results/comparison_cpu_gpu_summary.json`

## Tài liệu tham khảo

- TensorFlow NMT: <https://github.com/tensorflow/nmt>
- IWSLT 2015 English–Vietnamese: <https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/>

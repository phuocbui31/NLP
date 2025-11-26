# Báo cáo Lab5 phần 4: RNN for Named Entity Recognition (NER)

## Mô tả
Bài lab này xây dựng một mô hình RNN (sử dụng Bidirectional LSTM) để giải quyết bài toán Nhận dạng Thực thể Tên (Named Entity Recognition) trên bộ dữ liệu CoNLL 2003.

## Nội dung đã thực hiện

### Task 1: Tải và Tiền xử lý Dữ liệu
- ✅ Tải dữ liệu CoNLL 2003 từ Hugging Face
- ✅ Trích xuất câu và nhãn NER
- ✅ Chuyển đổi nhãn từ dạng số sang string
- ✅ Xây dựng từ điển `word_to_ix` và `tag_to_ix`

### Task 2: Tạo PyTorch Dataset và DataLoader
- ✅ Tạo class `NERDataset` kế thừa từ `torch.utils.data.Dataset`
- ✅ Implement các phương thức `__init__`, `__len__`, `__getitem__`
- ✅ Tạo `collate_fn` để padding các câu trong batch
- ✅ Tạo DataLoader cho train, validation, và test sets

### Task 3: Xây dựng Mô hình RNN
- ✅ Tạo class `SimpleRNNForTokenClassification`
- ✅ Sử dụng Bidirectional LSTM với 2 layers
- ✅ Thêm Embedding layer, Dropout, và Linear layer
- ✅ Tổng số parameters: ~2.7 triệu parameters

### Task 4: Huấn luyện Mô hình
- ✅ Khởi tạo `CrossEntropyLoss` với `ignore_index` cho padding
- ✅ Sử dụng Adam optimizer với learning rate 0.001
- ✅ Thêm Learning Rate Scheduler (ReduceLROnPlateau)
- ✅ Implement training loop với 5 epochs
- ✅ Gradient clipping để tránh exploding gradients
- ✅ Vẽ đồ thị loss theo epochs

### Task 5: Đánh giá Mô hình
- ✅ Tính token-level accuracy trên validation set
- ✅ Tính F1-score chi tiết cho từng loại thực thể
- ✅ Tạo hàm `predict_sentence()` để dự đoán câu mới
- ✅ Test với nhiều câu ví dụ khác nhau

### Bonus
- ✅ Lưu và load mô hình với checkpoint đầy đủ
- ✅ Hàm `load_model()` để restore mô hình

## Kiến trúc Mô hình

```
SimpleRNNForTokenClassification(
  (embedding): Embedding(21010, 100, padding_idx=0)
  (rnn): LSTM(100, 128, num_layers=2, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=256, out_features=10, bias=True)
  (dropout): Dropout(p=0.3, inplace=False)
)
```

### Thông số:
- **Vocab size**: 21,010 từ
- **Embedding dimension**: 100
- **Hidden dimension**: 128
- **Number of layers**: 2
- **Bidirectional**: Yes
- **Output classes**: 10 (số lượng nhãn NER)
- **Dropout**: 0.3

## ⚠️ Lưu ý quan trọng về Dataset

**Vấn đề với CoNLL 2003 dataset**:
Từ phiên bản mới của thư viện `datasets`, Hugging Face đã ngừng hỗ trợ dataset scripts. Notebook đã được cập nhật để tự động thử nhiều cách tải dataset:

1. Load trực tiếp từ `conll2003` (cách mới)
2. Load từ `eriktks/conll2003` (fork community)
3. Load từ revision cũ

Nếu vẫn gặp lỗi, hãy xem phần Troubleshooting bên dưới.

## Cách chạy

### 1. Cài đặt các thư viện cần thiết
```bash
pip install torch torchvision
pip install datasets
pip install transformers
pip install tqdm
pip install matplotlib
pip install scikit-learn
```

### 2. Chạy notebook
```bash
jupyter notebook lab5_rnn_ner.ipynb
```
Hoặc mở trong VS Code/Cursor và chạy từng cell.

### 3. Huấn luyện mô hình
Chạy các cell theo thứ tự từ trên xuống dưới. Quá trình huấn luyện sẽ:
- Tải và xử lý dữ liệu (~5 phút)
- Huấn luyện 5 epochs (~20-30 phút trên CPU, ~5-10 phút trên GPU)
- Tự động lưu best model vào file `best_ner_model.pt`

### 4. Dự đoán câu mới
Sau khi huấn luyện, bạn có thể dùng hàm `predict_sentence()`:

```python
predict_sentence("VNU University is located in Hanoi", model, word_to_ix, ix_to_tag, device)
```

## Kết quả mong đợi

- **Validation Accuracy**: ~95-97%
- **F1-score**: ~0.85-0.90 (tùy loại entity)

### Các nhãn NER trong CoNLL 2003:
- `O`: Outside (không phải entity)
- `B-PER`: Beginning of Person
- `I-PER`: Inside Person
- `B-ORG`: Beginning of Organization
- `I-ORG`: Inside Organization
- `B-LOC`: Beginning of Location
- `I-LOC`: Inside Location
- `B-MISC`: Beginning of Miscellaneous
- `I-MISC`: Inside Miscellaneous

## Files được tạo ra

Sau khi chạy notebook, các file sau sẽ được tạo:

1. `best_ner_model.pt` - Trọng số của model tốt nhất
2. `ner_model_checkpoint.pt` - Checkpoint đầy đủ (bao gồm vocabularies và hyperparameters)

## Ví dụ Output

```
Câu: VNU University is located in Hanoi
======================================================================
Token                Predicted Tag       
----------------------------------------------------------------------
VNU                  B-ORG               
University           I-ORG               
is                   O                   
located              O                   
in                   O                   
Hanoi                B-LOC               
======================================================================
```

## 🔧 Troubleshooting

### Vấn đề: "RuntimeError: Dataset scripts are no longer supported"

**Nguyên nhân**: Hugging Face đã ngừng hỗ trợ dataset scripts từ phiên bản mới.

**Giải pháp 1** (Khuyến nghị): Sử dụng dataset từ community
```python
dataset = load_dataset("eriktks/conll2003")
```

**Giải pháp 2**: Downgrade thư viện datasets
```bash
pip install datasets==2.14.0
```

**Giải pháp 3**: Tải dataset thủ công
```python
# Tải từ URL trực tiếp
from datasets import load_from_disk
# ... (xem chi tiết trong notebook)
```

### Vấn đề: Dataset tải quá lâu

**Giải pháp**: 
- Kiểm tra kết nối internet
- Dataset khoảng 80-100MB, cần internet ổn định
- Nếu bị gián đoạn, xóa cache: `rm -rf ~/.cache/huggingface/datasets/conll2003*`

## Cải tiến có thể thực hiện

1. **Sử dụng Pre-trained Embeddings**: GloVe, Word2Vec, FastText
2. **Thay đổi kiến trúc**: 
   - Thử GRU thay vì LSTM
   - Tăng số layers
   - Thử attention mechanism
3. **CRF Layer**: Thêm Conditional Random Field layer để cải thiện dự đoán
4. **Data Augmentation**: Tăng cường dữ liệu
5. **Hyperparameter Tuning**: Tìm learning rate, batch size tối ưu

## Tài liệu tham khảo

- [CoNLL 2003 Dataset](https://huggingface.co/datasets/conll2003)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Named Entity Recognition Paper](https://arxiv.org/abs/1603.01360)

## License
Educational Use Only

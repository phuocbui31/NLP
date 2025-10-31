# Report Lab5: Giới thiệu PyTorch

Lab5 hướng dẫn thực hành, đi qua các khái niệm cơ bản nhất của PyTorch, từ cấu trúc dữ liệu cốt lõi cho đến cách xây dựng một mô hình mạng nơ-ron đơn giản.

## Cài đặt thư viện

*   Đảm bảo cài đặt thư viện `uv`. Nếu chưa cài đặt, có thể sử dụng lệnh: ```curl -LsSf https://astral.sh/uv/install.sh | sh```

*   Khởi tạo môi trường để chạy code:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

*   Cài đặt thư viện cần thiết (bao gồm cả thư viện của các bài thực hành trước):
    ```bash
    uv sync
    ```

## Nội dung

**Bài thực hành gồm 3 phần chính**

### Phần 1: Khám phá Tensor

Phần này giới thiệu đối tượng `Tensor` - cấu trúc dữ liệu trung tâm của PyTorch.

* **Task 1.1: Tạo Tensor:**
    * Tạo tensor từ Python `list`.
    * Tạo tensor từ `numpy.array`.
    * Tạo tensor bằng các hàm tiện ích như `torch.ones_like()` và `torch.rand_like()`.
* **Task 1.2: Các phép toán trên Tensor:**
    * Phép cộng: `x_data + x_data`
    * Phép nhân vô hướng: `x_data * 5`
    * Phép nhân ma trận: `x_data @ x_data.T`
* **Task 1.3: Indexing và Slicing:**
    * Truy cập hàng, cột, và các phần tử cụ thể trong tensor.
* **Task 1.4: Thay đổi hình dạng Tensor:**
    * Sử dụng `torch.rand()` để tạo tensor với giá trị ngẫu nhiên và kích thước chỉ định.
    * Sử dụng `.reshape()` để thay đổi kích thước của tensor (từ `(4, 4)` thành `(16, 1)`).

---

### Phần 2: Tự động tính Đạo hàm

Phần này khám phá `torch.autograd`, cơ chế mạnh mẽ của PyTorch cho phép tự động tính toán đạo hàm, nền tảng của việc huấn luyện mạng nơ-ron.

* **Task 2.1: Thực hành với `autograd`:**
    1.  Tạo tensor `x` với `requires_grad=True` để PyTorch bắt đầu theo dõi các phép toán trên nó.
    2.  Thực hiện một chuỗi các phép toán: `y = x + 2` và `z = y * y * 3`.
    3.  Gọi `z.backward()` để tự động tính đạo hàm của `z` theo `x` ($\frac{dz}{dx}$).
    4.  Kiểm tra kết quả đạo hàm được lưu trong `x.grad`.
* **Câu hỏi mấu chốt:** Chuyện gì xảy ra nếu gọi `z.backward()` một lần nữa?
    * **Trả lời:** Chương trình sẽ báo lỗi `RuntimeError`.
    * **Lý do:** Để tiết kiệm bộ nhớ, PyTorch mặc định sẽ **xóa đồ thị tính toán** ngay sau khi `backward()` được gọi lần đầu. Các giá trị trung gian cần thiết để tính đạo hàm đã bị giải phóng.
    * **Giải pháp:** Nếu cần tính đạo hàm nhiều lần, sử dụng `z.backward(retain_graph=True)`.

---

### Phần 3: Xây dựng Mô hình đầu tiên với `torch.nn`

Phần này hướng dẫn cách sử dụng các lớp (layers) và module được xây dựng sẵn trong `torch.nn` để định nghĩa một mạng nơ-ron.

* **Task 3.1: Lớp `nn.Linear`:**
    * Giới thiệu lớp fully-connected (dense) cơ bản, thực hiện phép biến đổi tuyến tính $y = xA^T + b$.
* **Task 3.2: Lớp `nn.Embedding`:**
    * Giới thiệu lớp embedding, một bảng tra cứu (lookup table) hiệu quả, thường dùng để chuyển đổi các chỉ số (ví dụ: ID của từ) thành các vector đặc trưng (word embeddings).
* **Task 3.3: Kết hợp thành một `nn.Module`:**
    * Định nghĩa một mô hình hoàn chỉnh `MyFirstModel` bằng cách kế thừa từ `nn.Module`.
    * Mô hình này kết hợp `nn.Embedding`, `nn.Linear`, và `nn.ReLU` (hàm kích hoạt).
    * Triển khai phương thức `forward(self, indices)` để định nghĩa luồng dữ liệu đi qua mô hình.

#### Cấu trúc `MyFirstModel`

```python
from torch import nn

class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        # Định nghĩa các lớp (layer) bạn sẽ dùng
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU() # Hàm kích hoạt
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, indices):
        # Định nghĩa luồng dữ liệu đi qua các lớp
        # 1. Lấy embedding
        embeds = self.embedding(indices)
        # 2. Truyền qua lớp linear và hàm kích hoạt
        hidden = self.activation(self.linear(embeds))
        # 3. Truyền qua lớp output
        output = self.output_layer(hidden)
        return output
```

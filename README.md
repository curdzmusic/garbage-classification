# Hệ thống Phân loại Rác thải thông minh ♻️

**Ứng dụng web sử dụng Deep Learning để phân loại rác thải và cung cấp thông tin tái chế**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Library: PyTorch](https://img.shields.io/badge/Library-PyTorch-orange.svg)](https://pytorch.org/)

Dự án này là một hệ thống phân loại rác thải tự động sử dụng mô hình Deep Learning (CNN) để nhận diện và phân loại các loại rác thải từ hình ảnh. Hệ thống không chỉ phân loại rác mà còn cung cấp thông tin hữu ích về khả năng tái chế của từng loại, giúp người dùng có ý thức hơn về việc phân loại và xử lý rác thải đúng cách.

## ✨ Tính Năng Nổi Bật

- 🖼️ **Phân loại rác từ hình ảnh**: Sử dụng mô hình CNN tùy chỉnh để phân loại 10 loại rác thải (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash).
- 📊 **Hiển thị xác suất dự đoán**: Hiển thị biểu đồ xác suất cho tất cả các lớp và top-5 dự đoán với độ tin cậy cao nhất.
- ♻️ **Thông tin tái chế**: Tự động cung cấp thông tin về khả năng tái chế của từng loại rác (Tái chế được / Tái chế đặc biệt / Không tái chế).
- 📸 **Nhiều nguồn đầu vào**: Hỗ trợ 3 cách nhập ảnh:
  - Upload ảnh từ thiết bị
  - Nhập URL ảnh từ internet
  - Chụp ảnh trực tiếp từ camera
- 🎨 **Giao diện trực quan**: Giao diện web đẹp mắt, dễ sử dụng với Streamlit, hiển thị kết quả dự đoán và biểu đồ một cách trực quan.
- 🚀 **Hiệu suất cao**: Sử dụng PyTorch với hỗ trợ GPU để tăng tốc độ xử lý.

## 🛠️ Công Nghệ Sử Dụng

- **Backend & Machine Learning**:
  - **Python**: Ngôn ngữ lập trình chính.
  - **PyTorch**: Framework deep learning để xây dựng và triển khai mô hình CNN.
  - **Torchvision**: Thư viện hỗ trợ xử lý và transform hình ảnh.
  - **Streamlit**: Framework web để xây dựng giao diện người dùng nhanh chóng.
  - **PIL/Pillow**: Xử lý và thao tác với hình ảnh.
  - **OpenCV**: Xử lý hình ảnh và video (nếu cần).
  - **NumPy**: Tính toán số học và xử lý mảng.
  - **Matplotlib**: Vẽ biểu đồ và hiển thị dữ liệu.
  - **Plotly**: Tạo biểu đồ tương tác.
  - **Pandas**: Xử lý và phân tích dữ liệu.

- **Frontend**:
  - **Streamlit UI**: Giao diện người dùng được xây dựng hoàn toàn bằng Streamlit.

## 🚀 Cài đặt & Khởi chạy

### Yêu cầu
- Python 3.10.x hoặc cao hơn
- Git (để clone repository)
- CUDA (tùy chọn, để sử dụng GPU nếu có)

### Hướng dẫn cài đặt

1. **Clone repository về máy**
   ```bash
   git clone <your-repository-link>
   cd <repository-folder>
   ```

2. **Tạo và kích hoạt môi trường ảo (Virtual Environment)**
   
   Đây là bước quan trọng để tránh xung đột thư viện giữa các dự án.
   ```bash
   # Tạo môi trường ảo
   python -m venv venv
   
   # Kích hoạt trên Windows
   venv\Scripts\activate.bat
   
   # Kích hoạt trên macOS/Linux
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện cần thiết**
   
   File `requirements.txt` chứa tất cả các thư viện Python cần thiết.
   ```bash
   pip install -r requirements.txt
   ```

### Chạy chương trình

1. **Khởi chạy ứng dụng Streamlit**
   
   Chạy file `app.py` để khởi động ứng dụng web.
   ```bash
   streamlit run app.py
   ```
   
   *Lưu ý: Nếu bạn muốn sử dụng file `streamlit_garbage_classifier.py` thay thế, chạy:*
   ```bash
   streamlit run streamlit_garbage_classifier.py
   ```

2. **Truy cập ứng dụng**
   
   Ứng dụng sẽ tự động mở trong trình duyệt web tại địa chỉ:
   [http://localhost:8501](http://localhost:8501)
   
   Nếu không tự động mở, bạn có thể truy cập thủ công bằng cách mở trình duyệt và nhập địa chỉ trên.

3. **Sử dụng ứng dụng**
   - Chọn một trong 3 phương thức nhập ảnh: Upload, URL, hoặc Camera
   - Tải ảnh lên hoặc chụp ảnh
   - Xem kết quả phân loại và thông tin tái chế

## 📂 Cấu Trúc Dự Án

```
├── README.md                          # File README
├── app.py                             # File Streamlit chính (ứng dụng chính)
├── streamlit_garbage_classifier.py    # File Streamlit thay thế (có thêm tính năng streaming)
├── requirements.txt                   # Danh sách các thư viện Python
├── Garbage_Classification.ipynb       # Notebook Jupyter: Xử lý dữ liệu và huấn luyện mô hình
├── Garbage_Classification_Standalone.ipynb  # Notebook standalone (nếu có)
│
├── model/                             # Thư mục chứa mô hình
│   ├── garbage_classifier_model.pth   # File mô hình đã được huấn luyện
│   ├── garbage_classifier.py          # Định nghĩa kiến trúc mô hình CNN
│   └── model_load.py                  # Module tải mô hình
│
├── components/                        # Các component Streamlit
│   ├── upload.py                      # Module xử lý đầu vào (upload, URL, camera)
│   └── result_display.py              # Module hiển thị kết quả và biểu đồ
│
├── util/                              # Các tiện ích và hàm hỗ trợ
│   ├── inference.py                   # Hàm suy luận (prediction)
│   ├── utilities.py                   # Các hàm tiện ích (transform, load image, ...)
│   ├── predict_analysis.py           # Phân tích và xử lý kết quả dự đoán
│   ├── recycle_info.py                # Thông tin về khả năng tái chế
│   └── chart_render.py                # Render biểu đồ
│
└── config/                            # Thư mục cấu hình (nếu có)
```

## 📝 Hướng Dẫn Sử Dụng

### Xử lý dữ liệu và Huấn luyện Mô hình

1. **Mở notebook Jupyter**:
   - Mở file `Garbage_Classification.ipynb` trong Jupyter Notebook hoặc JupyterLab
   - Notebook này chứa toàn bộ quy trình:
     - Tải và xử lý dữ liệu
     - Xây dựng kiến trúc mô hình
     - Huấn luyện mô hình
     - Đánh giá và lưu mô hình

2. **Chạy từng ô (cell)**:
   - Chạy các ô theo thứ tự để thực hiện từng bước
   - Đảm bảo bạn đã có dữ liệu huấn luyện trước khi chạy

3. **Lưu mô hình**:
   - Sau khi huấn luyện xong, mô hình sẽ được lưu vào `model/garbage_classifier_model.pth`
   - Mô hình này sẽ được sử dụng trong ứng dụng web

### Triển khai và Sử dụng Ứng dụng Web

1. **Đảm bảo mô hình đã được huấn luyện**:
   - File `model/garbage_classifier_model.pth` phải tồn tại
   - Nếu chưa có, hãy chạy notebook `Garbage_Classification.ipynb` trước

2. **Khởi chạy ứng dụng**:
   ```bash
   streamlit run app.py
   ```

3. **Sử dụng giao diện**:
   - **Upload từ máy**: Chọn file ảnh từ máy tính của bạn
   - **Ảnh từ URL**: Nhập URL của ảnh trên internet
   - **Camera**: Chụp ảnh trực tiếp từ webcam

4. **Xem kết quả**:
   - Kết quả phân loại sẽ hiển thị ngay sau khi tải ảnh
   - Xem biểu đồ xác suất cho tất cả các lớp
   - Xem top-5 dự đoán với độ tin cậy
   - Đọc thông tin về khả năng tái chế

## 🎯 Các Loại Rác được Phân loại

Hệ thống có thể phân loại 10 loại rác thải sau:

1. **battery** - Pin (Tái chế đặc biệt)
2. **biological** - Rác hữu cơ (Không tái chế)
3. **cardboard** - Bìa carton (Tái chế được)
4. **clothes** - Quần áo (Không tái chế)
5. **glass** - Thủy tinh (Tái chế được)
6. **metal** - Kim loại (Tái chế được)
7. **paper** - Giấy (Tái chế được)
8. **plastic** - Nhựa (Tái chế được)
9. **shoes** - Giày dép (Không tái chế)
10. **trash** - Rác hỗn hợp (Không tái chế)

## 🐛 Xử lý sự cố (Troubleshooting)

- **Lỗi `ModuleNotFoundError: No module named '...'`**:
  - Đảm bảo bạn đã kích hoạt môi trường ảo (`venv`) trước khi chạy ứng dụng.
  - Chạy lại lệnh `pip install -r requirements.txt` để chắc chắn rằng tất cả thư viện đã được cài đặt.

- **Lỗi `FileNotFoundError: model/garbage_classifier_model.pth`**:
  - Đảm bảo bạn đã huấn luyện mô hình bằng cách chạy `Garbage_Classification.ipynb`.
  - Kiểm tra xem file mô hình có tồn tại trong thư mục `model/` không.

- **Camera không hoạt động**:
  - Kiểm tra xem camera đã được kết nối đúng cách và không bị ứng dụng nào khác sử dụng.
  - Đảm bảo bạn đã cấp quyền cho trình duyệt truy cập camera.

- **Ứng dụng chạy chậm**:
  - Nếu có GPU, đảm bảo PyTorch đã được cài đặt với hỗ trợ CUDA.
  - Kiểm tra xem GPU có được sử dụng không bằng cách xem log khi khởi động ứng dụng.

- **Lỗi khi tải ảnh từ URL**:
  - Kiểm tra kết nối internet.
  - Đảm bảo URL ảnh hợp lệ và có thể truy cập được.

## 📄 Giấy Phép

Dự án này được phân phối dưới giấy phép MIT. Xem tệp `LICENSE` để biết thêm thông tin.

## 👥 Đóng Góp

Chúng tôi hoan nghênh các đóng góp từ cộng đồng. Để đóng góp:

1. Fork dự án.
2. Tạo nhánh tính năng mới (`git checkout -b feature/AmazingFeature`).
3. Commit các thay đổi của bạn (`git commit -m 'Add some AmazingFeature'`).
4. Push lên nhánh (`git push origin feature/AmazingFeature`).
5. Mở một Pull Request.

## 📧 Liên Hệ

Nếu bạn có câu hỏi hoặc góp ý, vui lòng tạo một issue trên GitHub repository.

## 🙏 Lời Cảm Ơn

Cảm ơn tất cả những người đã đóng góp và hỗ trợ cho dự án này. Dự án này được phát triển với mục tiêu nâng cao ý thức về phân loại rác thải và bảo vệ môi trường.


# Tìm hiểu về OCR

## I. Tìm hiểu về công nghệ

**OCR** (Optical Character Recognition – Nhận dạng ký tự quang học) là công nghệ cho phép máy tính chuyển đổi hình ảnh chứa chữ viết (trên sách, báo, bìa sách, hóa đơn, ảnh chụp màn hình…) thành văn bản số có thể tìm kiếm, chỉnh sửa hoặc lưu trữ.

---

## II. Quy trình hoạt động của OCR

Quy trình OCR thường trải qua 5 bước chính:

### 1. Tiền xử lý ảnh (Preprocessing)
Mục tiêu: cải thiện chất lượng ảnh để dễ nhận diện hơn.
- Chuyển ảnh màu → ảnh xám (*grayscale*).
- Làm sạch nhiễu (*noise reduction*).
- Cân chỉnh ảnh (*deskewing*) nếu ảnh bị nghiêng.
- Phân ngưỡng (*binarization*): biến ảnh thành trắng/đen để làm rõ chữ.  
Giúp OCR phân biệt chữ với nền dễ dàng hơn.

### 2. Phân vùng (Segmentation)
- Tách ảnh thành các khối văn bản, dòng chữ, sau đó đến từng ký tự.  
Ví dụ: một bìa sách có cả tên sách + tác giả + NXB, hệ thống cần chia ra từng vùng chữ để nhận dạng riêng.

### 3. Trích xuất đặc trưng (Feature Extraction)
- Mỗi ký tự sẽ được biểu diễn dưới dạng đặc trưng hình học (đường cong, nét thẳng, góc cạnh…) hoặc dưới dạng ma trận pixel.  
- Giúp máy tính “hiểu” hình dạng của ký tự đó.

### 4. Nhận dạng ký tự (Character Recognition)
Có 2 cách phổ biến:
1. **So khớp mẫu** (*Pattern Matching*): so sánh ký tự trong ảnh với bộ mẫu đã có sẵn (hữu ích cho font chữ chuẩn).
2. **Học máy** (*Machine Learning/Deep Learning*): dùng mô hình huấn luyện (CNN, RNN) để nhận diện ký tự, kể cả khi font chữ lạ hoặc ảnh mờ.  

Đây là bước cốt lõi giúp OCR nhận ra chữ.

### 5. Hậu xử lý (Post-processing)
- Sửa lỗi nhận dạng dựa trên từ điển tiếng Việt  
  *(VD: OCR nhận sai thành “Lap trinh C co ban” → sửa thành “Lập trình C cơ bản”)*.
- Chuẩn hóa kết quả: định dạng văn bản, tách từ, xử lý dấu câu.

---

## III. Tìm hiểu về thư viện Keras-OCR

Keras-OCR là một thư viện Python mã nguồn mở, dùng để nhận diện văn bản trong ảnh.

**Ưu điểm**:
- Hỗ trợ pipeline đầy đủ (phát hiện + nhận dạng).
- Có sẵn model pretrained, chỉ cần cài đặt là dùng ngay.
- Nhận diện được chữ nhiều ngôn ngữ (có thể *fine-tune* để hỗ trợ tiếng Việt).

**Nhược điểm**:
- Nặng hơn so với Tesseract hoặc EasyOCR.
- Cần GPU nếu xử lý ảnh nhiều/nhanh.

---

## IV.Quá trình cài đặt và triển khai OCR bằng Docker

### 1. Cài đặt ban đầu
- Cài `keras-ocr` bằng:
  ```bash
  pip install keras-ocr
  ```
- Khi chạy `import keras_ocr`, gặp lỗi không tìm thấy TensorFlow → Cài thêm:
  ```bash
  pip install tensorflow
  ```

### 2. Xung đột phiên bản
- TensorFlow mới nhất (2.16.1) gây lỗi:
  ```
  ValueError: Unrecognized keyword arguments passed to Dense: {'weights': ...}
  ```
- Nguyên nhân: Keras-OCR không tương thích với TensorFlow/Keras quá mới.
- Giải pháp:
  ```bash
  pip install tensorflow==2.10.1 numpy==1.23.5
  ```

### 3. Vấn đề môi trường & dung lượng ổ C
- Do cài nhiều gói Python nặng, ổ C đầy.
- Giải pháp: Chạy bằng Docker, lưu image/container ở ổ D.

### 4. Viết Dockerfile & build image
- Dùng base image `python:3.10-slim`.
- Cài `keras-ocr`, `tensorflow==2.10.1`, `numpy==1.23.5`, `matplotlib`, `opencv-python-headless`.
- Copy `testKeras.py` vào container.
- Build:
  ```bash
  docker build -t keras-ocr-app .
  ```
- Chạy:
  ```bash
  docker run --rm keras-ocr-app
  ```

### 5. Lỗi khi tải ảnh online
- Sử dụng ảnh từ Wikipedia → bị lỗi HTTP 403 (server chặn request).
- Giải pháp: dùng ảnh local (`.jpg` trong thư mục project).

### 6. Kết quả
- OCR trên thư mục `MauBiaSach` (5 ảnh bìa sách tiếng Việt).
- Keras-OCR phát hiện chữ nhưng **mất dấu tiếng Việt** do mô hình chỉ hỗ trợ tiếng Anh.
- Kết luận:
  - Keras-OCR mạnh ở **phát hiện vùng chữ** (*detection*).
  - Nếu cần chữ tiếng Việt đầy đủ dấu → Nên dùng **EasyOCR (lang='vi')** hoặc **Tesseract OCR với gói vie**.

---
## V.Cách cài đặt:
- Yêu cầu máy có Docker
- Clone Repo về máy
- Sau đó chạy lệnh `docker build -t keras-ocr-app .`
- Sau đó chạy lệnh `docker run --rm -v "$(pwd)/MauBiaSach:/app/MauBiaSach" keras-ocr-app` 
---
---


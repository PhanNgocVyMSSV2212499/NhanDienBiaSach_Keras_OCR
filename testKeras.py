import os
import keras_ocr

# Thư mục chứa ảnh
image_dir = "MauBiaSach"

# Khởi tạo pipeline của keras-ocr (gồm text detector + recognizer)
pipeline = keras_ocr.pipeline.Pipeline()

# Lấy danh sách file ảnh .jpg trong thư mục
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

if not image_files:
    print(f"⚠ Không tìm thấy ảnh .jpg trong thư mục: {image_dir}")
    exit()

results = []

# OCR từng ảnh
for img_path in image_files:
    print(f"\n👉 Đang OCR ảnh: {img_path}")
    try:
        prediction_groups = pipeline.recognize([img_path])
        # prediction_groups[0] là list [(text, box), ...]
        sentence = " ".join([text for text, box in prediction_groups[0]])
    except Exception as e:
        sentence = f"Lỗi OCR: {e}"
    
    results.append((os.path.basename(img_path), sentence))
    print(f"   → {sentence}")

# Xuất kết quả ra file
output_path = os.path.join(os.getcwd(), "ketqua_keras.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for img, sentence in results:
        f.write(f"Ảnh: {img}\n")
        f.write(f"{sentence}\n\n")

print(f"\n✅ Hoàn tất! Kết quả đã lưu vào {output_path}")
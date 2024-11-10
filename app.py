import gradio as gr
import numpy as np
import cv2
from keras.models import load_model

theme = gr.themes.Soft()

# Tải mô hình
model = load_model('emotion_model.h5')

# Nhãn cảm xúc
emotion_labels = ['Giận dữ', 'Ghê tởm', 'Sợ hãi', 'Hạnh phúc', 'Bình thường', 'Buồn bã', 'Ngạc nhiên']
emotion_icons = ['😡', '🤢','😱', '😊','🙂','😭','😮']

def resize_image(image, max_size=500):
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def predict_emotion(image):
    # Resize ảnh cho preview
    preview_image = resize_image(image)

    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    print(f"Số lượng khuôn mặt phát hiện: {len(faces)}")  # In ra số lượng khuôn mặt phát hiện

    if len(faces) == 0:
        return "Không phát hiện khuôn mặt", preview_image  # Trả về hình ảnh gốc nếu không phát hiện khuôn mặt

    # Lấy khuôn mặt đầu tiên
    (x, y, w, h) = faces[0]
    face_image = gray_image[y:y+h, x:x+w]  # Cắt khuôn mặt

    # Vẽ hình chữ nhật quanh khuôn mặt trên ảnh gốc
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hình chữ nhật màu xanh

    # Resize ảnh gốc đã khoanh vùng cho preview
    preview_image = resize_image(image)

    # Chuyển đổi hình ảnh thành định dạng mà mô hình có thể xử lý
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype('float32') / 255.0
    face_image = face_image.reshape(1, 48, 48, 1)

    # Dự đoán cảm xúc
    prediction = model.predict(face_image)
    print(prediction)
    emotion_index = np.argmax(prediction)
    print(f"Vị trí của cảm xúc: {emotion_index + 1}")
    emotion = emotion_labels[emotion_index]
    icon = emotion_icons[emotion_index]
    return f"Cảm xúc khuôn mặt là: {emotion} {icon}", preview_image  # Trả về cảm xúc và hình ảnh đã khoanh vùng

js = """
window.onload = function() {
        if (!window.location.href.includes('__theme=light')) {
            window.location.href = window.location.href + '?__theme=light';
        }
    }
"""

css = """
textarea {
    font-size: 20px;
}
body {
    background-color: white !important;
    color: black !important;
}
"""
# Tạo giao diện Gradio với CSS tùy chỉnh
description_text = "Ứng dụng này sử dụng mô hình CNN cho phép người dùng tải lên hình ảnh khuôn mặt và dự đoán cảm xúc của người trong ảnh.."

iface = gr.Interface(theme=theme,
                     fn=predict_emotion, 
                     inputs=gr.Image(type="numpy", label="Tải lên hình ảnh"),
                     outputs=[gr.Textbox(label="Cảm xúc dự đoán"), gr.Image(label="Hình ảnh đã khoanh vùng")],
                     title="Dự đoán cảm xúc khuôn mặt",
                     description=description_text,
                     css=css,
                     js=js
                     )  # Thêm mô tả cho ứng dụng

# Chạy ứng dụng
iface.launch()
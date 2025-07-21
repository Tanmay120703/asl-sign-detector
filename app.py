import streamlit as st
import torch
import cv2
import numpy as np
from model import ASLClassifier
from utils import get_transforms
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ASLClassifier(num_classes=29)
model.load_state_dict(torch.load("sign_model.pth", map_location=device))
model.to(device)
model.eval()

labels = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

IMG_SIZE = 128
_, test_tf = get_transforms(IMG_SIZE)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    size = max(h, w)
    padded = np.ones((size, size, 3), dtype=np.uint8) * 255
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = img
    img = Image.fromarray(padded)
    return test_tf(img).unsqueeze(0)

def crop_center(img, size=300):
    h, w, _ = img.shape
    ch, cw = size, size
    startx = w//2 - (cw//2)
    starty = h//2 - (ch//2)
    return img[starty:starty+ch, startx:startx+cw]

st.sidebar.header("📘 ASL Guide")
emoji_map = {
    'A': '✊', 'B': '✋', 'C': '👐', 'D': '☝️', 'E': '🤏',
    'F': '👌', 'G': '👉', 'H': '✌️', 'I': '🤙', 'J': '↪️',
    'K': '🖖', 'L': '👆', 'M': '✊', 'N': '✊', 'O': '🫦',
    'P': '👌', 'Q': '👇', 'R': '🤞', 'S': '✊', 'T': '🤏',
    'U': '✌️', 'V': '✌️', 'W': '🖐️', 'X': '☝️', 'Y': '🤙', 'Z': '✍️',
    'space': '␣', 'del': '❌', 'nothing': '⬜'
}
for k, v in emoji_map.items():
    st.sidebar.markdown(f"{k} : {v}")

st.title("🤟 Real-time ASL Sign Detection")
st.markdown("Show one ASL sign in the center of your webcam — the app predicts the symbol.")

class ASLTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hand = crop_center(img)

        if hand is not None:
            tensor = preprocess_image(hand).to(device)
            with torch.no_grad():
                output = model(tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

                label = labels[pred_idx] if pred_idx < len(labels) else "unknown"
                text = f"{label.upper()} ({confidence:.2f})"

                cv2.putText(img, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(img, "No input", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return img

webrtc_streamer(key="asl-detection", video_transformer_factory=ASLTransformer)

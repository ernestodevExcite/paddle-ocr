import cv2
import numpy as np
from paddleocr import PaddleOCR


# =============================
# CONFIG
# =============================
IMAGE_PATH = "images/CALLING.png"
DEBUG_IMAGE = "debug_paddle.png"


# =============================
# PREPROCESADO SEGURO
# =============================
def light_preprocess(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("No se pudo cargar la imagen")

    h, w = img.shape[:2]

    # Reescalar si es pequeña
    if w < 1000:
        img = cv2.resize(
            img, None, fx=2, fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    # ❗ NO convertir a gris
    # Reducir ruido conservando color
    denoise = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return denoise


# =============================
# OCR (API NUEVA)
# =============================
def run_ocr(image):
    ocr = PaddleOCR(
        lang="es",
        use_textline_orientation=True  # ✅ reemplaza use_angle_cls
    )

    return ocr.predict(image)


# =============================
# DEBUG VISUAL
# =============================
def draw_boxes(image_path, result, output_path):
    img = cv2.imread(image_path)

    for res in result:
        boxes = res["dt_polys"]
        texts = res["rec_texts"]
        scores = res["rec_scores"]

        for box, text, score in zip(boxes, texts, scores):
            box = np.array(box, dtype=np.int32)

            cv2.polylines(img, [box], True, (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{text} ({score:.2f})",
                (box[0][0], box[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imwrite(output_path, img)


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    processed = light_preprocess(IMAGE_PATH)
    result = run_ocr(processed)

    print("\n===== TEXTO DETECTADO =====\n")

    for res in result:
        for text, score in zip(res["rec_texts"], res["rec_scores"]):
            print(f"{text} ({score:.2f})")

    draw_boxes(IMAGE_PATH, result, DEBUG_IMAGE)
    print(f"\nDebug guardado en {DEBUG_IMAGE}")

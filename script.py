import cv2
import numpy as np
from paddleocr import PaddleOCR


IMAGE_PATH = "images/image.jpg"
DEBUG_IMAGE = "debug_paddle.png"


# =============================
# PREPROCESADO
# =============================
def light_preprocess(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("No se pudo cargar la imagen")

    h, w = img.shape[:2]
    scale = 1.0

    if w < 1000:
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoise = cv2.fastNlMeansDenoisingColored(
        img, None, 10, 10, 7, 21
    )

    return denoise, scale


# =============================
# OCR
# =============================
def run_ocr(image):
    ocr = PaddleOCR(
        lang="en",
        use_textline_orientation=True,
        use_doc_unwarping=True,
        use_doc_orientation_classify=True,
    )
    return ocr.predict(image)


# =============================
# DIBUJAR BOXES (CORRECTO)
# =============================
def draw_boxes(image, result, output_path):
    debug = image.copy()

    for res in result:
        for box, text, score in zip(
            res["dt_polys"],
            res["rec_texts"],
            res["rec_scores"]
        ):
            box = np.array(box, dtype=np.int32)

            # Polígono real (mejor precisión)
            x, y, w, h = cv2.boundingRect(box)
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

            x, y = box[0]
            cv2.putText(
                debug,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imwrite(output_path, debug)


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    processed_img, scale = light_preprocess(IMAGE_PATH)
    result = run_ocr(processed_img)

    print("\n===== TEXTO DETECTADO =====\n")
    for res in result:
        for t, s in zip(res["rec_texts"], res["rec_scores"]):
            print(f"{t} ({s:.2f})")

    draw_boxes(processed_img, result, DEBUG_IMAGE)
    print(f"\nDebug guardado en {DEBUG_IMAGE}")
    # crear archivo txt con el texto detectado
    with open("output.txt", "w", encoding="utf-8") as f:
        for res in result:
            for t, s in zip(res["rec_texts"], res["rec_scores"]):
                f.write(f"{t} ({s:.2f})\n")
    print("Texto guardado en output.txt")

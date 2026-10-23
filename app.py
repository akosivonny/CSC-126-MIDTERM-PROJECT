from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
import os
import uuid
import re

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
PROCESSED_FOLDER = os.path.join("static", "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "bmp", "webp", "avif"}
SEGMENTATION_MODES = {"none", "contours"}

# Load your five Haar Cascade classifiers
haarcascades = "haarcascades"

cascades = {
    "classifier1": cv2.CascadeClassifier(
        os.path.join(haarcascades, "haarcascade_fullbody.xml")
    ),
    "classifier2": cv2.CascadeClassifier(
        os.path.join(haarcascades, "haarcascade_frontalface_alt2.xml")
    ),
    "classifier3": cv2.CascadeClassifier(
        os.path.join(haarcascades, "cascadeside.xml")
    ),
    "classifier4": cv2.CascadeClassifier(
        os.path.join(haarcascades, "cascadefront.xml")
    ),
    "classifier5": cv2.CascadeClassifier(
        os.path.join(haarcascades, "cascadeback.xml")
    ),
}

# Choose the classifier you want to use
active_cascade = cascades["classifier2"]

# Check whether the selected XML loaded correctly
if active_cascade.empty():
    raise ValueError("Could not load the selected Haar Cascade XML.")


def file_extension(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def segment_contours(image, min_area_ratio=0.005):
    """Contour-based segmentation.

    Steps: grayscale -> blur -> Otsu threshold -> morphological clean-up ->
    findContours -> filter small areas -> approxPolyDP -> moments (centroids).
    Returns (list of contours, filled-region view on black).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If most of the image became "foreground", the background was the bright part: invert
    if cv2.countNonZero(thresh) > 0.5 * h * w:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * h * w
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Simplify each contour to a cleaner outline
    contours = [
        cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True) for c in contours
    ]

    # View of the segmented regions: each region filled with its own color on black
    seg_view = np.zeros_like(image)
    for i, c in enumerate(contours):
        hue = int((i * 47) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.drawContours(seg_view, [c], -1, tuple(int(v) for v in color), -1)

    return contours, seg_view


def detect_people(image, segmentation="none"):
    """Detect people, optionally segment, and return (count, annotated image, segmentation view)."""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes = active_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    count = len(boxes)
    seg_view = None

    if segmentation == "contours":
        contours, seg_view = segment_contours(original)
        cv2.drawContours(image, contours, -1, (0, 200, 255), 2)

        # Moments -> centroid of each segmented region
        for i, c in enumerate(contours, start=1):
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(image, str(i), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.drawContours(seg_view, [c], -1, (255, 255, 255), 1)

    for (x, y, w, h) in boxes:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 220, 0),
            2
        )

        cv2.putText(
            image,
            "Person",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2
        )

    return count, image, seg_view


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None

    if request.method == "POST":
        uploaded = request.files.get("media")
        segmentation = request.form.get("segmentation", "none")
        if segmentation not in SEGMENTATION_MODES:
            segmentation = "none"

        if not uploaded or uploaded.filename == "":
            error = "Please choose an image first."
        else:
            ext = file_extension(uploaded.filename)
            if ext not in ALLOWED_IMAGES:
                error = "Please choose a supported image type."
            else:
                # Build output names from the uploaded file's name
                original_name = os.path.basename(uploaded.filename)
                base_name = os.path.splitext(original_name)[0]
                base_name = re.sub(r'[<>:"/\\|?*]', "", base_name).strip() or "image"

                # OpenCV can't reliably write .avif, so fall back to .jpg
                out_ext = ext if ext in {"png", "jpg", "jpeg", "bmp", "webp"} else "jpg"

                token = uuid.uuid4().hex
                input_path = os.path.join(UPLOAD_FOLDER, f"{token}.{ext}")
                uploaded.save(input_path)
                try:
                    image = cv2.imread(input_path)
                    if image is None:
                        raise ValueError("Could not read this image. Please try another one.")
                    count, processed, seg_view = detect_people(image, segmentation)

                    output_name = f"{base_name}_output.{out_ext}"
                    output_path = os.path.join(PROCESSED_FOLDER, output_name)
                    if not cv2.imwrite(output_path, processed):
                        raise ValueError("Could not save the processed image.")

                    seg_url = None
                    if seg_view is not None:
                        seg_name = f"{base_name}_segmented.{out_ext}"
                        cv2.imwrite(os.path.join(PROCESSED_FOLDER, seg_name), seg_view)
                        seg_url = url_for("static", filename=f"processed/{seg_name}", v=token)

                    result = {
                        "media_url": url_for("static", filename=f"processed/{output_name}", v=token),
                        "people_count": count,
                        "segmentation": segmentation,
                        "seg_url": seg_url,
                    }
                except Exception as exc:
                    error = str(exc)
                finally:
                    try:
                        os.remove(input_path)
                    except OSError:
                        pass

    return render_template("index.html", error=error, result=result)


if __name__ == "__main__":
    app.run(debug=True)
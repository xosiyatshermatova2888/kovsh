from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import time
import threading
import logging
import os
from ultralytics import YOLO

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "exp-2.pt")
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "static", "video.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_THRESHOLD = 0.35
IMG_SIZE = 320
JPEG_QUALITY = 100

last_frame = None
last_frame_lock = threading.Lock()
stop_event = threading.Event()

frame_count = 0
last_log_time = time.time()
analysis_results = []
analysis_lock = threading.Lock()

logger.info(f"Загрузка модели: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.to('cpu')
model.fuse()
logger.info("Модель успешно загружена на CPU")


def save_annotated_video():
    logger.info("Сохранение аннотированного видео...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео: {VIDEO_PATH}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_vid, (w, h))

    results_data = []
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        count += 1
        results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False, device='cpu')
        annotated = results[0].plot(font_size=0.8, line_width=1, labels=True, conf=True)
        out.write(annotated)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                results_data.append({
                    "frame": count,
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        if count % 30 == 0:
            logger.info(f"Обработано {count}/{total_frames} кадров ({count*100//total_frames}%)")

    cap.release()
    out.release()

    with analysis_lock:
        analysis_results.clear()
        analysis_results.extend(results_data)

    logger.info(f"Аннотированное видео сохранено: {output_path}")
    logger.info(f"Всего обнаружений: {len(results_data)}")
    logger.info(f"Уникальные классы: {set(r['class'] for r in results_data)}")


def capture_loop():
    global last_frame, frame_count, last_log_time

    while not stop_event.is_set():
        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning("Не удалось открыть видео. Повтор...")
            time.sleep(1)
            continue

        logger.info("Видео успешно открыто")

        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            start_inf = time.time()

            results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False, device='cpu')
            annotated_frame = results[0].plot(font_size=0.8, line_width=1, labels=True, conf=True)

            ret, buffer = cv2.imencode('.jpg', annotated_frame,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ret:
                with last_frame_lock:
                    last_frame = buffer.tobytes()

            inf_time = time.time() - start_inf
            now = time.time()
            if now - last_log_time > 5:
                fps = frame_count / (now - last_log_time + 1e-6)
                logger.info(f"FPS: {fps:.1f} | Инференс: {inf_time*1000:.0f} мс")
                last_log_time = now
                frame_count = 0

        cap.release()
        if not stop_event.is_set():
            time.sleep(0.5)


threading.Thread(target=capture_loop, daemon=True).start()


def gen_frames():
    while not stop_event.is_set():
        with last_frame_lock:
            if last_frame is None:
                time.sleep(0.05)
                continue
            frame_bytes = last_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with analysis_lock:
        classes = {}
        for r in analysis_results:
            c = r["class"]
            classes[c] = classes.get(c, 0) + 1
    return jsonify({
        "status": "running",
        "device": "CPU",
        "last_frame_available": last_frame is not None,
        "total_detections": len(analysis_results),
        "classes": classes
    })


@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    try:
        logger.info("Сохранение аннотированного видео...")
        threading.Thread(target=save_annotated_video, daemon=True).start()

        logger.info("Запуск Flask сервера на порту 5000...")
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки...")
    except Exception as e:
        logger.error(f"Ошибка сервера: {e}")
    finally:
        stop_event.set()
        logger.info("Приложение завершено")

import cv2
import torch
import numpy as np
import time
import yaml
import threading

from pathlib import Path
from flask import Flask, Response, jsonify
from executorch.runtime import Runtime
from picamera2 import Picamera2


modeldir = Path("ex480")
modelpath = modeldir / "model.pte"
metafile = modeldir / "metadata.yaml"

inputsize = 480
confthreshold = 0.35
nmsthreshold = 0.45

camerawidth = 640
cameraheight = 480

numthreads = 4
serverport = 5000

serverip = input(
    "Enter server IP (0.0.0.0 for all interfaces): "
).strip()

if not serverip:
    serverip = "0.0.0.0"


def loadclasses():
    with open(metafile, "r") as f:
        metadata = yaml.safe_load(f)

    names = metadata["names"]

    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]

    return names


if not modelpath.exists():
    raise FileNotFoundError(modelpath)

if not metafile.exists():
    raise FileNotFoundError(metafile)


classes = loadclasses()

print("Classes:", classes)

torch.set_num_threads(numthreads)

print("PyTorch:", torch.__version__)
print("Loading ExecuTorch model...")

runtime = Runtime.get()
program = runtime.load_program(str(modelpath))

print("Available methods:", program.method_names)

model = program.load_method("forward")

print("ExecuTorch model loaded successfully")


picam = Picamera2()

cameraconfig = picam.create_video_configuration(
    main={
        "size": (camerawidth, cameraheight),
        "format": "RGB888"
    }
)

picam.configure(cameraconfig)
picam.start()

time.sleep(2)

print("Camera started successfully")


def preprocess(frame):
    h, w = frame.shape[:2]

    scale = min(inputsize / w, inputsize / h)

    neww = int(round(w * scale))
    newh = int(round(h * scale))

    resized = cv2.resize(
        frame,
        (neww, newh),
        interpolation=cv2.INTER_LINEAR
    )

    padw = inputsize - neww
    padh = inputsize - newh

    left = padw // 2
    top = padh // 2

    right = padw - left
    bottom = padh - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    image = padded.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)

    tensor = torch.from_numpy(image)

    return tensor, scale, left, top


def inference(inputtensor):
    with torch.inference_mode():
        output = model.execute((inputtensor,))

    return output


def decodepredictions(output):
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise RuntimeError(
                f"Expected one output tensor, received {len(output)}"
            )

        output = output[0]

    if isinstance(output, torch.Tensor):
        predictions = output.detach().cpu().numpy()
    else:
        predictions = np.asarray(output)

    predictions = np.squeeze(predictions)

    if predictions.ndim != 2:
        raise RuntimeError(
            f"Unexpected output shape: {predictions.shape}"
        )

    numfeatures = len(classes) + 4

    if predictions.shape[0] == numfeatures:
        predictions = predictions.T

    elif predictions.shape[1] != numfeatures:
        raise RuntimeError(
            f"Unexpected YOLO output shape: {predictions.shape}"
        )

    boxes = []
    scores = []
    classids = []

    for prediction in predictions:
        cx, cy, width, height = prediction[:4]

        classscores = prediction[4:]

        classid = int(np.argmax(classscores))
        confidence = float(classscores[classid])

        if confidence < confthreshold:
            continue

        x1 = cx - width / 2
        y1 = cy - height / 2

        x2 = cx + width / 2
        y2 = cy + height / 2

        boxes.append([
            float(x1),
            float(y1),
            float(x2 - x1),
            float(y2 - y1)
        ])

        scores.append(confidence)
        classids.append(classid)

    return boxes, scores, classids


def postprocess(boxes, scores, classids):
    detections = []

    for classid in set(classids):
        indices = [
            i for i, value in enumerate(classids)
            if value == classid
        ]

        classboxes = [boxes[i] for i in indices]
        classscores = [scores[i] for i in indices]

        selected = cv2.dnn.NMSBoxes(
            classboxes,
            classscores,
            confthreshold,
            nmsthreshold
        )

        if len(selected) > 0:
            for index in np.array(selected).flatten():
                originalindex = indices[index]

                detections.append((
                    boxes[originalindex],
                    scores[originalindex],
                    classids[originalindex]
                ))

    return detections


def drawdetections(frame, detections, scale, left, top):
    h, w = frame.shape[:2]

    for box, confidence, classid in detections:
        x, y, bw, bh = box

        x1 = int((x - left) / scale)
        y1 = int((y - top) / scale)

        x2 = int((x + bw - left) / scale)
        y2 = int((y + bh - top) / scale)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))

        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        label = f"{classes[classid]} {confidence:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame


app = Flask("pcbdetector")

framelock = threading.Lock()
latestframe = None

stats = {
    "fps": 0.0,
    "inference": 0.0,
    "loop": 0.0,
    "detections": 0,
    "status": "Starting"
}

running = threading.Event()
running.set()


def inferenceworker():
    global latestframe

    framecount = 0
    fpstime = time.perf_counter()
    fps = 0.0

    print("Warming up model...")

    warmup = torch.zeros(
        (1, 3, inputsize, inputsize),
        dtype=torch.float32
    )

    for _ in range(3):
        inference(warmup)

    print("Warm-up completed")
    print("Live inference started")

    try:
        while running.is_set():
            starttime = time.perf_counter()

            frame = picam.capture_array()

            inputtensor, scale, left, top = preprocess(frame)

            inferencestart = time.perf_counter()
            output = inference(inputtensor)

            inferencetime = (
                time.perf_counter() - inferencestart
            ) * 1000

            boxes, scores, classids = decodepredictions(output)

            detections = postprocess(
                boxes,
                scores,
                classids
            )

            outputframe = drawdetections(
                frame.copy(),
                detections,
                scale,
                left,
                top
            )

            framecount += 1

            currenttime = time.perf_counter()
            elapsed = currenttime - fpstime

            if elapsed >= 1.0:
                fps = framecount / elapsed
                framecount = 0
                fpstime = currenttime

            looptime = (
                time.perf_counter() - starttime
            ) * 1000

            cv2.putText(
                outputframe,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 0),
                2
            )

            cv2.putText(
                outputframe,
                f"Inference: {inferencetime:.1f} ms",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            cv2.putText(
                outputframe,
                f"Detections: {len(detections)}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            outputframe = cv2.cvtColor(
                outputframe,
                cv2.COLOR_RGB2BGR
            )

            success, encoded = cv2.imencode(
                ".jpg",
                outputframe,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if success:
                with framelock:
                    latestframe = encoded.tobytes()

                    stats["fps"] = round(fps, 2)
                    stats["inference"] = round(inferencetime, 2)
                    stats["loop"] = round(looptime, 2)
                    stats["detections"] = len(detections)
                    stats["status"] = "Running"

    except Exception as error:
        print("Inference error:", error)

        with framelock:
            stats["status"] = str(error)

    finally:
        picam.stop()
        print("Camera released")


def streamframes():
    while True:
        with framelock:
            frame = latestframe

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


@app.route("/")
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <title>PCB Fault Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
    body {
        background: #10141c;
        color: white;
        font-family: Arial;
        text-align: center;
        margin: 0;
        padding: 20px;
    }

    h1 {
        color: #00e5a8;
    }

    .container {
        max-width: 1100px;
        margin: auto;
    }

    .video {
        width: 100%;
        max-width: 850px;
        border: 2px solid #00e5a8;
        border-radius: 10px;
    }

    .stats {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 15px;
        margin-top: 20px;
    }

    .card {
        background: #202938;
        padding: 20px;
        border-radius: 10px;
        min-width: 150px;
    }

    .value {
        color: #00e5a8;
        font-size: 25px;
        font-weight: bold;
    }

    .label {
        color: #b0bec5;
        margin-bottom: 10px;
    }
    </style>
    </head>

    <body>
    <div class="container">

    <h1>PCB FAULT DETECTOR</h1>
    <p>ExecuTorch CPU Inference | Live Camera</p>

    <img class="video" src="/video">

    <div class="stats">

    <div class="card">
    <div class="label">FPS</div>
    <div class="value" id="fps">0</div>
    </div>

    <div class="card">
    <div class="label">Inference Time</div>
    <div class="value" id="inference">0 ms</div>
    </div>

    <div class="card">
    <div class="label">Loop Time</div>
    <div class="value" id="loop">0 ms</div>
    </div>

    <div class="card">
    <div class="label">Detections</div>
    <div class="value" id="detections">0</div>
    </div>

    </div>

    <p id="status">Connecting...</p>

    </div>

    <script>
    async function updatestats() {
        try {
            const response = await fetch("/stats");
            const data = await response.json();

            document.getElementById("fps").innerText = data.fps;
            document.getElementById("inference").innerText =
                data.inference + " ms";

            document.getElementById("loop").innerText =
                data.loop + " ms";

            document.getElementById("detections").innerText =
                data.detections;

            document.getElementById("status").innerText =
                "Status: " + data.status;

        } catch (error) {
            document.getElementById("status").innerText =
                "Connection lost";
        }
    }

    setInterval(updatestats, 1000);
    updatestats();
    </script>

    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        streamframes(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def getstats():
    with framelock:
        return jsonify(stats.copy())


worker = threading.Thread(
    target=inferenceworker,
    daemon=True
)

worker.start()

print("\nPCB FAULT DETECTOR SERVER")
print("-------------------------")
print("Server bind address:", serverip)
print("Port:", serverport)
print("\nFind Raspberry Pi IP using: hostname -I")

try:
    app.run(
        host=serverip,
        port=serverport,
        threaded=True,
        debug=False,
        use_reloader=False
    )

finally:
    running.clear()
    worker.join(timeout=5)

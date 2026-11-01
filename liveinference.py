import cv2
import ncnn
import numpy as np
import time
import yaml
import os

from picamera2 import Picamera2


# Model configuration
modeldir = "PCB_model_ncnn"
paramfile = os.path.join(modeldir, "model.ncnn.param")
binfile = os.path.join(modeldir, "model.ncnn.bin")
metafile = os.path.join(modeldir, "metadata.yaml")

inputname = "in0"
outputname = "out0"

inputsize = 640
confthreshold = 0.35
nmsthreshold = 0.45


# Load class names
def loadclasses():

    try:
        with open(metafile, "r") as f:
            metadata = yaml.safe_load(f)

        names = metadata.get("names")

        if isinstance(names, dict):
            return [names[k] if k in names else names[str(k)]
                    for k in range(len(names))]

        if isinstance(names, list):
            return names

    except Exception as e:
        print("Metadata loading error:", e)

    return [
        "Missing_Hole",
        "MouseBite",
        "Open_Circuit",
        "Short_Circuit",
        "Spur",
        "Spurious_Cooper"
    ]


classes = loadclasses()


# Initialize NCNN model
net = ncnn.Net()

net.opt.num_threads = 4
net.opt.use_vulkan_compute = False

if net.load_param(paramfile) != 0:
    raise RuntimeError("Model parameter loading failed")

if net.load_model(binfile) != 0:
    raise RuntimeError("Model loading failed")

print("NCNN model loaded successfully")
print("Classes:", classes)


# Initialize Raspberry Pi camera
picam = Picamera2()

config = picam.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam.configure(config)
picam.start()

time.sleep(2)

print("Camera started")


# Preprocess camera frame
def preprocess(frame):

    h, w = frame.shape[:2]

    # Center crop into a square
    cropsize = min(h, w)

    xoffset = (w - cropsize) // 2
    yoffset = (h - cropsize) // 2

    cropped = frame[
        yoffset:yoffset + cropsize,
        xoffset:xoffset + cropsize
    ]

    # Resize to model input resolution
    resized = cv2.resize(
        cropped,
        (inputsize, inputsize),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert image into NCNN format
    mat = ncnn.Mat.from_pixels(
        resized,
        ncnn.Mat.PixelType.PIXEL_RGB,
        inputsize,
        inputsize
    )

    # Normalize image pixels
    mat.substract_mean_normalize(
        [],
        [1 / 255.0, 1 / 255.0, 1 / 255.0]
    )

    return mat, xoffset, yoffset, cropsize


# Run NCNN inference
def inference(inputmat):

    extractor = net.create_extractor()

    extractor.input(inputname, inputmat)

    ret, output = extractor.extract(outputname)

    if ret != 0:
        raise RuntimeError("Inference failed")

    predictions = np.array(output, dtype=np.float32)

    # Convert output into expected format
    if predictions.ndim != 2:
        raise RuntimeError(
            f"Unexpected output shape: {predictions.shape}"
        )

    if predictions.shape[0] == 10 and predictions.shape[1] != 10:
        predictions = predictions.T

    if predictions.shape[1] != 10:
        raise RuntimeError(
            f"Unexpected prediction shape: {predictions.shape}"
        )

    return predictions


# Decode predictions and apply NMS
def postprocess(predictions):

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

        x1 = int(cx - width / 2)
        y1 = int(cy - height / 2)

        x2 = int(cx + width / 2)
        y2 = int(cy + height / 2)

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(confidence)
        classids.append(classid)

    detections = []

    for classid in set(classids):

        indices = [
            i for i, c in enumerate(classids)
            if c == classid
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


# Draw detection results
def drawdetections(frame, detections, xoffset, yoffset, cropsize):

    scale = inputsize / cropsize

    for box, confidence, classid in detections:

        x, y, w, h = box

        x1 = int(x / scale + xoffset)
        y1 = int(y / scale + yoffset)

        x2 = int((x + w) / scale + xoffset)
        y2 = int((y + h) / scale + yoffset)

        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))

        x2 = max(0, min(x2, frame.shape[1] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))

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


# Main inference loop
framecount = 0
fps = 0
fpstime = time.time()

try:

    while True:

        starttime = time.time()

        # Capture camera frame
        frame = picam.capture_array()

        # Preprocessing
        inputmat, xoffset, yoffset, cropsize = preprocess(frame)

        # Model inference
        predictions = inference(inputmat)

        # Postprocessing
        detections = postprocess(predictions)

        # Draw results
        outputframe = drawdetections(
            frame.copy(),
            detections,
            xoffset,
            yoffset,
            cropsize
        )

        # Calculate FPS
        framecount += 1
        elapsed = time.time() - fpstime

        if elapsed >= 1.0:

            fps = framecount / elapsed
            framecount = 0
            fpstime = time.time()

        inferencetime = (time.time() - starttime) * 1000

        cv2.putText(
            outputframe,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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

        cv2.imshow(
            "PCB Fault Detection",
            cv2.cvtColor(outputframe, cv2.COLOR_RGB2BGR)
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Inference stopped")

finally:

    picam.stop()
    cv2.destroyAllWindows()

    print("Camera released")

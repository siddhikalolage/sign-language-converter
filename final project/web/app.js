const video = document.getElementById("video");
const canvas = document.getElementById("captureCanvas");
const imageInput = document.getElementById("imageInput");
const uploadPreview = document.getElementById("uploadPreview");
const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");
const resultWord = document.getElementById("resultWord");
const resultMeta = document.getElementById("resultMeta");
const statusPill = document.getElementById("statusPill");
const cameraView = document.getElementById("cameraView");
const uploadView = document.getElementById("uploadView");

let activeMode = "landmark";
let activeSource = "camera";
let uploadedImageData = "";
let autoPredictTimer = null;
let requestInFlight = false;
let stableHistory = [];
let missStreak = 0;
let committedLabel = "";

const CAMERA_PREDICT_INTERVAL_MS = 950;
const CAMERA_HISTORY_LIMIT = 6;
const CAMERA_RELEASE_MISSES = 4;
const LANDMARK_MIN_CONFIDENCE = 0.30;
const LANDMARK_MIN_QUALITY = 0.45;
const LANDMARK_MIN_MARGIN = 0.08;
const YOLO_MIN_CONFIDENCE = 0.55;

function setStatus(text) {
  statusPill.textContent = text;
}

function setResult(label, meta) {
  resultWord.textContent = label;
  resultMeta.textContent = meta;
}

function setActiveToggle(containerId, value, attrName) {
  const buttons = document.querySelectorAll(`#${containerId} [data-${attrName}]`);
  buttons.forEach((button) => {
    const isActive = button.dataset[attrName] === value;
    button.classList.toggle("active", isActive);
  });
}

function resetCameraConsensus({ clearCommitted = false } = {}) {
  stableHistory = [];
  missStreak = 0;
  if (clearCommitted) {
    committedLabel = "";
  }
}

function shouldKeepPrediction(result) {
  if (!result || !result.label) {
    return false;
  }

  if (result.mode === "landmark") {
    const quality = Number(result.quality ?? 0);
    const margin = Number(result.margin ?? 0);
    return (
      result.confidence >= LANDMARK_MIN_CONFIDENCE &&
      quality >= LANDMARK_MIN_QUALITY &&
      margin >= LANDMARK_MIN_MARGIN
    );
  }

  return result.confidence >= YOLO_MIN_CONFIDENCE;
}

function commitStablePrediction(result) {
  stableHistory.push(result);
  if (stableHistory.length > CAMERA_HISTORY_LIMIT) {
    stableHistory.shift();
  }
  missStreak = 0;

  const labelCounts = new Map();
  stableHistory.forEach((item) => {
    labelCounts.set(item.label, (labelCounts.get(item.label) || 0) + 1);
  });

  let bestLabel = "";
  let bestCount = 0;
  labelCounts.forEach((count, label) => {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  });

  const requiredVotes = Math.max(
    2,
    Math.ceil(Math.min(stableHistory.length, CAMERA_HISTORY_LIMIT) * 0.55),
  );

  if (bestLabel && bestCount >= requiredVotes) {
    const winners = stableHistory.filter((item) => item.label === bestLabel);
    const averageConfidence =
      winners.reduce((sum, item) => sum + item.confidence, 0) / winners.length;
    committedLabel = bestLabel;
    setResult(
      committedLabel,
      `${result.mode === "landmark" ? "Landmark" : "YOLO"} confidence ${Math.round(
        averageConfidence * 100,
      )}%.`,
    );
    setStatus("Recognized");
  } else {
    setStatus("Hold the phrase steady");
  }
}

function handleCameraMiss(message) {
  missStreak += 1;
  if (!committedLabel) {
    setResult("Waiting", message);
  }
  if (missStreak >= CAMERA_RELEASE_MISSES) {
    stableHistory = [];
    if (!committedLabel) {
      setStatus("Show a phrase");
    } else {
      setStatus("Ready for next phrase");
    }
  }
}

function handleCameraPrediction(result) {
  if (!shouldKeepPrediction(result)) {
    handleCameraMiss("Move closer and hold the phrase still.");
    return;
  }

  commitStablePrediction(result);
}

function stopAutoPredictLoop() {
  if (autoPredictTimer !== null) {
    clearInterval(autoPredictTimer);
    autoPredictTimer = null;
  }
}

function startAutoPredictLoop() {
  stopAutoPredictLoop();
  if (activeSource !== "camera") {
    return;
  }

  autoPredictTimer = window.setInterval(async () => {
    if (requestInFlight || document.hidden) {
      return;
    }

    try {
      requestInFlight = true;
      const imageData = captureCameraFrame();
      const result = await sendPrediction(imageData);
      handleCameraPrediction(result);
    } catch (error) {
      console.error(error);
      handleCameraMiss(error.message || "Prediction failed.");
    } finally {
      requestInFlight = false;
    }
  }, CAMERA_PREDICT_INTERVAL_MS);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera unavailable");
    setResult("Unavailable", "Your browser does not support webcam access.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    setStatus("Camera ready");
    video.onloadedmetadata = () => {
      setStatus("Show a phrase");
      startAutoPredictLoop();
    };
  } catch (error) {
    console.error(error);
    setStatus("Camera blocked");
    setResult("Permission needed", "Allow webcam access or switch to upload mode.");
  }
}

function captureCameraFrame() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    throw new Error("Camera frame is not ready yet.");
  }

  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.92);
}

async function sendPrediction(imageData) {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_data: imageData,
      mode: activeMode,
    }),
  });

  const rawText = await response.text();
  let payload = {};
  try {
    payload = rawText ? JSON.parse(rawText) : {};
  } catch (error) {
    payload = { detail: rawText || "Prediction failed." };
  }

  if (!response.ok) {
    throw new Error(payload.detail || "Prediction failed.");
  }
  return payload;
}

async function runPrediction() {
  try {
    setStatus("Predicting");
    let imageData = "";

    if (activeSource === "camera") {
      imageData = captureCameraFrame();
    } else {
      if (!uploadedImageData) {
        throw new Error("Choose an image before predicting.");
      }
      imageData = uploadedImageData;
    }

    const result = await sendPrediction(imageData);
    if (activeSource === "camera") {
      handleCameraPrediction(result);
    } else {
      const qualityText =
        result.quality === null || result.quality === undefined
          ? ""
          : ` Landmarks ${Math.round(result.quality * 100)}%.`;

      setResult(
        result.label,
        `${result.mode === "landmark" ? "Landmark" : "YOLO"} confidence ${Math.round(result.confidence * 100)}%.${qualityText}`
      );
      setStatus("Done");
    }
  } catch (error) {
    console.error(error);
    if (activeSource === "camera") {
      handleCameraMiss(error.message || "Prediction failed.");
    } else {
      setResult("No prediction", error.message);
      setStatus("Try again");
    }
  }
}

document.getElementById("modeToggle").addEventListener("click", (event) => {
  const button = event.target.closest("[data-mode]");
  if (!button) {
    return;
  }
  activeMode = button.dataset.mode;
  resetCameraConsensus({ clearCommitted: false });
  setActiveToggle("modeToggle", activeMode, "mode");
});

document.getElementById("sourceToggle").addEventListener("click", (event) => {
  const button = event.target.closest("[data-source]");
  if (!button) {
    return;
  }
  activeSource = button.dataset.source;
  setActiveToggle("sourceToggle", activeSource, "source");
  cameraView.classList.toggle("hidden", activeSource !== "camera");
  uploadView.classList.toggle("hidden", activeSource !== "upload");
  resetCameraConsensus({ clearCommitted: activeSource !== "camera" });
  setStatus(activeSource === "camera" ? "Show a phrase" : "Upload ready");
  if (activeSource === "camera") {
    startAutoPredictLoop();
  } else {
    stopAutoPredictLoop();
  }
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files || [];
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    uploadedImageData = String(reader.result || "");
    uploadPreview.src = uploadedImageData;
    uploadPreview.classList.remove("hidden");
    setStatus("Image loaded");
  };
  reader.readAsDataURL(file);
});

predictButton.addEventListener("click", () => {
  runPrediction();
});

clearButton.addEventListener("click", () => {
  uploadedImageData = "";
  uploadPreview.removeAttribute("src");
  resetCameraConsensus({ clearCommitted: true });
  resultWord.textContent = "Waiting";
  resultMeta.textContent = "Point the camera at a phrase or upload an image.";
  setStatus("Idle");
});

setActiveToggle("modeToggle", activeMode, "mode");
setActiveToggle("sourceToggle", activeSource, "source");
startCamera();

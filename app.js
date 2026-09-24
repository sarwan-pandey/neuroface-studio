/**
 * ═══════════════════════════════════════════════════════════════════
 *  NEUROFACE STUDIO — REAL-TIME IN-BROWSER BIOMETRIC ENGINE
 * ═══════════════════════════════════════════════════════════════════
 */

// Global App State
const state = {
  modelsLoaded: false,
  cameraActive: false,
  enrollCamActive: false,
  liveStream: null,
  enrollStream: null,
  animFrameId: null,
  enrollAnimId: null,
  threshold: 0.45,
  showLandmarks: true,
  showBoxes: true,
  modelType: 'tiny', // 'tiny' or 'ssd'
  gallery: [], // Array of { id, name, descriptor, samples: [dataUrl, ...] }
  faceMatcher: null,
  capturedSnaps: [], // Array of dataUrls for current enrollment
  selectedPersonId: null,
  fpsCounter: {
    lastTime: performance.now(),
    frames: 0,
    fps: 0
  }
};

const MODEL_BASE_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';
const STORAGE_KEY = 'neuroface_biometric_gallery_v1';

// DOM Element References
const elements = {
  // Navigation
  navBtns: document.querySelectorAll('.nav-btn'),
  pages: document.querySelectorAll('.page'),
  navGalleryBadge: document.getElementById('navGalleryBadge'),
  sidebarStatusText: document.getElementById('sidebarStatusText'),

  // Status Badge
  modelStatusDot: document.getElementById('modelStatusDot'),
  modelStatusTitle: document.getElementById('modelStatusTitle'),
  modelStatusDetail: document.getElementById('modelStatusDetail'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnBuildGallery: document.getElementById('btnBuildGallery'),

  // Live Recognition View
  btnToggleCamera: document.getElementById('btnToggleCamera'),
  camBtnIcon: document.getElementById('camBtnIcon'),
  camBtnText: document.getElementById('camBtnText'),
  webcamVideo: document.getElementById('webcamVideo'),
  overlayCanvas: document.getElementById('overlayCanvas'),
  videoPlaceholder: document.getElementById('videoPlaceholder'),
  liveMatchBanner: document.getElementById('liveMatchBanner'),
  matchBannerName: document.getElementById('matchBannerName'),
  matchBannerScore: document.getElementById('matchBannerScore'),
  thresholdSlider: document.getElementById('thresholdSlider'),
  thresholdVal: document.getElementById('thresholdVal'),
  chkShowLandmarks: document.getElementById('chkShowLandmarks'),
  chkShowBoxes: document.getElementById('chkShowBoxes'),

  // Telemetry Cards
  metricFps: document.getElementById('metricFps'),
  metricFaces: document.getElementById('metricFaces'),
  metricIdentified: document.getElementById('metricIdentified'),
  metricStatus: document.getElementById('metricStatus'),
  metricStatusSub: document.getElementById('metricStatusSub'),

  // Enroll View
  enrollNameInput: document.getElementById('enrollNameInput'),
  btnStartEnrollCam: document.getElementById('btnStartEnrollCam'),
  btnStopEnrollCam: document.getElementById('btnStopEnrollCam'),
  btnSnapFace: document.getElementById('btnSnapFace'),
  btnUploadPhotos: document.getElementById('btnUploadPhotos'),
  fileUploadInput: document.getElementById('fileUploadInput'),
  enrollVideo: document.getElementById('enrollVideo'),
  enrollCanvas: document.getElementById('enrollCanvas'),
  enrollPlaceholder: document.getElementById('enrollPlaceholder'),
  thumbsContainer: document.getElementById('thumbsContainer'),
  snapCountBadge: document.getElementById('snapCountBadge'),
  btnSaveEnrollment: document.getElementById('btnSaveEnrollment'),
  btnClearSnaps: document.getElementById('btnClearSnaps'),

  // Gallery View
  personsList: document.getElementById('personsList'),
  totalIdentitiesBadge: document.getElementById('totalIdentitiesBadge'),
  photoGrid: document.getElementById('photoGrid'),
  selectedPersonTitle: document.getElementById('selectedPersonTitle'),
  selectedPersonSub: document.getElementById('selectedPersonSub'),
  btnRefreshGallery: document.getElementById('btnRefreshGallery'),
  btnDeleteSelectedPerson: document.getElementById('btnDeleteSelectedPerson'),

  // Settings View
  selectModelType: document.getElementById('selectModelType'),
  selectCameraDevice: document.getElementById('selectCameraDevice'),
  btnExportGallery: document.getElementById('btnExportGallery'),
  btnImportGallery: document.getElementById('btnImportGallery'),
  importFileInput: document.getElementById('importFileInput'),
  btnClearAllData: document.getElementById('btnClearAllData'),

  // Toast
  toast: document.getElementById('toast')
};

// ═══════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  loadGalleryFromStorage();
  setupEventListeners();
  populateCameraDevices();

  // Auto-load model on startup
  showToast('Initializing NeuroFace Neural Weights...', 'info');
  await loadAIModels();
});

// Toast Notifications
function showToast(msg, type = 'info') {
  const t = elements.toast;
  t.textContent = msg;
  t.className = 'toast';
  if (type === 'success') t.classList.add('toast-success');
  if (type === 'error') t.classList.add('toast-error');
  t.style.display = 'block';
  setTimeout(() => { t.style.display = 'none'; }, 3500);
}

// ═══════════════════════════════════════════════════════════════════
//  NAVIGATION CONTROLLER
// ═══════════════════════════════════════════════════════════════════
function setupNavigation() {
  elements.navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const pageId = btn.dataset.page;
      showPage(pageId);
    });
  });
}

function showPage(pageId) {
  elements.navBtns.forEach(b => b.classList.toggle('active', b.dataset.page === pageId));
  elements.pages.forEach(p => {
    p.classList.toggle('active', p.id === `page${capitalize(pageId)}`);
  });

  if (pageId === 'gallery') {
    renderGalleryList();
  }
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ═══════════════════════════════════════════════════════════════════
//  AI MODEL LOADER
// ═══════════════════════════════════════════════════════════════════
async function loadAIModels() {
  try {
    elements.modelStatusDot.className = 'status-dot dot-amber';
    elements.modelStatusTitle.textContent = 'Loading Weights...';
    elements.modelStatusDetail.textContent = 'Fetching ONNX/TensorFlow.js shards';
    elements.sidebarStatusText.textContent = 'Loading models...';

    // Load Tiny Face Detector, 68 Landmark Net, and 128D Face Recognition Net
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_BASE_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_BASE_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_BASE_URL)
    ]);

    state.modelsLoaded = true;
    elements.modelStatusDot.className = 'status-dot dot-green';
    elements.modelStatusTitle.textContent = 'Engine Active';
    elements.modelStatusDetail.textContent = 'WebGL / WASM Pipeline Ready';
    elements.sidebarStatusText.textContent = 'Ready • Models Loaded';

    buildFaceMatcher();
    showToast('NeuroFace AI Engine Loaded Successfully!', 'success');
  } catch (err) {
    console.error('Failed to load AI models:', err);
    elements.modelStatusDot.className = 'status-dot dot-red';
    elements.modelStatusTitle.textContent = 'Loading Failed';
    elements.modelStatusDetail.textContent = 'Check internet or CORS policy';
    elements.sidebarStatusText.textContent = 'Error loading weights';
    showToast('Failed to load models. Check network connectivity.', 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
//  GALLERY STORAGE & VECTOR MATCHER
// ═══════════════════════════════════════════════════════════════════
function loadGalleryFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      state.gallery = parsed.map(item => ({
        ...item,
        descriptor: new Float32Array(item.descriptor)
      }));
    } else {
      state.gallery = [];
    }
  } catch (e) {
    console.warn('Could not parse gallery storage:', e);
    state.gallery = [];
  }
  updateGalleryBadge();
  buildFaceMatcher();
}

function saveGalleryToStorage() {
  const serializable = state.gallery.map(item => ({
    id: item.id,
    name: item.name,
    descriptor: Array.from(item.descriptor),
    samples: item.samples || []
  }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
  updateGalleryBadge();
  buildFaceMatcher();
}

function updateGalleryBadge() {
  const count = state.gallery.length;
  elements.navGalleryBadge.textContent = count;
  elements.totalIdentitiesBadge.textContent = count;
}

function buildFaceMatcher() {
  if (!state.gallery.length) {
    state.faceMatcher = null;
    return;
  }
  const labeledDescriptors = state.gallery.map(person => {
    return new faceapi.LabeledFaceDescriptors(person.name, [person.descriptor]);
  });
  // Use current threshold
  state.faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, state.threshold);
}

// ═══════════════════════════════════════════════════════════════════
//  LIVE CAMERA RECOGNITION PIPELINE
// ═══════════════════════════════════════════════════════════════════
async function toggleCamera() {
  if (state.cameraActive) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  if (!state.modelsLoaded) {
    showToast('Please wait for models to finish loading', 'error');
    return;
  }

  try {
    const deviceId = elements.selectCameraDevice.value;
    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
        ...(deviceId && deviceId !== 'default' ? { deviceId: { exact: deviceId } } : {})
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.liveStream = stream;
    elements.webcamVideo.srcObject = stream;

    await new Promise(resolve => {
      elements.webcamVideo.onloadedmetadata = () => {
        elements.webcamVideo.play();
        resolve();
      };
    });

    state.cameraActive = true;
    elements.camBtnIcon.textContent = '⏹';
    elements.camBtnText.textContent = 'Stop Camera';
    elements.btnToggleCamera.className = 'btn btn-danger btn-lg';
    elements.videoPlaceholder.style.display = 'none';

    elements.metricStatus.textContent = 'Running';
    elements.metricStatusSub.textContent = 'Real-time inference';

    // Size overlay canvas to video
    elements.overlayCanvas.width = elements.webcamVideo.videoWidth;
    elements.overlayCanvas.height = elements.webcamVideo.videoHeight;

    runLiveInferenceLoop();
    showToast('Camera stream active • Biometric HUD started', 'success');
  } catch (err) {
    console.error('Camera access error:', err);
    showToast('Could not access webcam. Check browser permissions.', 'error');
  }
}

function stopCamera() {
  state.cameraActive = false;
  if (state.liveStream) {
    state.liveStream.getTracks().forEach(t => t.stop());
    state.liveStream = null;
  }
  if (state.animFrameId) {
    cancelAnimationFrame(state.animFrameId);
    state.animFrameId = null;
  }

  elements.webcamVideo.srcObject = null;
  elements.camBtnIcon.textContent = '▶';
  elements.camBtnText.textContent = 'Start Camera';
  elements.btnToggleCamera.className = 'btn btn-success btn-lg';
  elements.videoPlaceholder.style.display = 'flex';
  elements.liveMatchBanner.style.display = 'none';

  // Clear overlay canvas
  const ctx = elements.overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);

  // Reset telemetry
  elements.metricFps.textContent = '—';
  elements.metricFaces.textContent = '0';
  elements.metricIdentified.textContent = '0';
  elements.metricStatus.textContent = 'Idle';
  elements.metricStatusSub.textContent = 'Camera offline';
}

async function runLiveInferenceLoop() {
  if (!state.cameraActive) return;

  const video = elements.webcamVideo;
  const canvas = elements.overlayCanvas;
  const ctx = canvas.getContext('2d');

  if (video.readyState >= 2 && !video.paused && !video.ended) {
    // 1. Calculate FPS
    const now = performance.now();
    state.fpsCounter.frames++;
    if (now - state.fpsCounter.lastTime >= 1000) {
      state.fpsCounter.fps = Math.round((state.fpsCounter.frames * 1000) / (now - state.fpsCounter.lastTime));
      elements.metricFps.textContent = state.fpsCounter.fps;
      state.fpsCounter.frames = 0;
      state.fpsCounter.lastTime = now;
    }

    // 2. Perform Real-time Face Detection + Recognition
    const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.35 });
    const detections = await faceapi
      .detectAllFaces(video, options)
      .withFaceLandmarks()
      .withFaceDescriptors();

    // 3. Clear Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    elements.metricFaces.textContent = detections.length;
    let identifiedCount = 0;
    let topMatch = null;

    // 4. Render Bounding Boxes & Identity HUD
    detections.forEach(det => {
      const { box } = det.detection;
      let label = 'Unknown';
      let confidence = 0;
      let isMatch = false;

      if (state.faceMatcher && det.descriptor) {
        const bestMatch = state.faceMatcher.findBestMatch(det.descriptor);
        if (bestMatch.label !== 'unknown' && bestMatch.distance <= state.threshold) {
          label = bestMatch.label;
          isMatch = true;
          identifiedCount++;
          // Convert distance to similarity percentage
          confidence = Math.round(Math.max(0, Math.min(100, (1 - bestMatch.distance / state.threshold) * 100)));
          if (!topMatch || confidence > topMatch.confidence) {
            topMatch = { label, confidence };
          }
        }
      }

      // Draw Face Landmarks if enabled
      if (state.showLandmarks && det.landmarks) {
        drawFacialLandmarks(ctx, det.landmarks, isMatch);
      }

      // Draw Target Bounding Box
      if (state.showBoxes) {
        drawTargetBoundingBox(ctx, box, label, confidence, isMatch);
      }
    });

    elements.metricIdentified.textContent = identifiedCount;

    // 5. Update Live Match Notification Banner
    if (topMatch) {
      elements.matchBannerName.textContent = topMatch.label;
      elements.matchBannerScore.textContent = `${topMatch.confidence}% Match`;
      elements.liveMatchBanner.style.display = 'flex';
    } else {
      elements.liveMatchBanner.style.display = 'none';
    }
  }

  state.animFrameId = requestAnimationFrame(runLiveInferenceLoop);
}

// High-Tech Cyber Bounding Box Renderer
function drawTargetBoundingBox(ctx, box, label, confidence, isMatch) {
  const x = box.x;
  const y = box.y;
  const w = box.width;
  const h = box.height;

  const color = isMatch ? '#10b981' : '#ef4444'; // Emerald green vs Coral red
  const cornerLen = Math.min(22, w * 0.2);

  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = color;

  // Corner brackets
  ctx.beginPath();
  // Top-Left
  ctx.moveTo(x, y + cornerLen);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLen, y);
  // Top-Right
  ctx.moveTo(x + w - cornerLen, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + cornerLen);
  // Bottom-Left
  ctx.moveTo(x, y + h - cornerLen);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x + cornerLen, y + h);
  // Bottom-Right
  ctx.moveTo(x + w - cornerLen, y + h);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w, y + h - cornerLen);
  ctx.stroke();

  // Subtle translucent box fill
  ctx.fillStyle = isMatch ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)';
  ctx.fillRect(x, y, w, h);

  // Identity Tag Pill
  const tagText = isMatch ? `✓ ${label} (${confidence}%)` : `? Unknown Identity`;
  ctx.font = '600 13px Inter, sans-serif';
  const textWidth = ctx.measureText(tagText).width;
  const pillHeight = 26;
  const pillY = Math.max(8, y - pillHeight - 6);

  // Pill Background
  ctx.fillStyle = isMatch ? 'rgba(16, 185, 129, 0.95)' : 'rgba(239, 68, 68, 0.95)';
  ctx.beginPath();
  ctx.roundRect(x, pillY, textWidth + 20, pillHeight, 6);
  ctx.fill();

  // Pill Text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(tagText, x + 10, pillY + 17);

  ctx.restore();
}

// 68-Point Landmark Mesh Renderer
function drawFacialLandmarks(ctx, landmarks, isMatch) {
  const points = landmarks.positions;
  ctx.save();
  ctx.fillStyle = isMatch ? 'rgba(16, 185, 129, 0.6)' : 'rgba(59, 130, 246, 0.5)';
  points.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════════════
//  ENROLLMENT ENGINE
// ═══════════════════════════════════════════════════════════════════
async function startEnrollCamera() {
  if (state.enrollCamActive) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    state.enrollStream = stream;
    elements.enrollVideo.srcObject = stream;

    await new Promise(resolve => {
      elements.enrollVideo.onloadedmetadata = () => {
        elements.enrollVideo.play();
        resolve();
      };
    });

    state.enrollCamActive = true;
    elements.btnStartEnrollCam.disabled = true;
    elements.btnStopEnrollCam.disabled = false;
    elements.btnSnapFace.disabled = false;
    elements.enrollPlaceholder.style.display = 'none';

    elements.enrollCanvas.width = elements.enrollVideo.videoWidth;
    elements.enrollCanvas.height = elements.enrollVideo.videoHeight;

    runEnrollPreviewLoop();
  } catch (err) {
    console.error('Enroll camera error:', err);
    showToast('Failed to start enrollment camera', 'error');
  }
}

function stopEnrollCamera() {
  state.enrollCamActive = false;
  if (state.enrollStream) {
    state.enrollStream.getTracks().forEach(t => t.stop());
    state.enrollStream = null;
  }
  if (state.enrollAnimId) {
    cancelAnimationFrame(state.enrollAnimId);
    state.enrollAnimId = null;
  }

  elements.enrollVideo.srcObject = null;
  elements.btnStartEnrollCam.disabled = false;
  elements.btnStopEnrollCam.disabled = true;
  elements.btnSnapFace.disabled = true;
  elements.enrollPlaceholder.style.display = 'block';

  const ctx = elements.enrollCanvas.getContext('2d');
  ctx.clearRect(0, 0, elements.enrollCanvas.width, elements.enrollCanvas.height);
}

async function runEnrollPreviewLoop() {
  if (!state.enrollCamActive) return;

  const video = elements.enrollVideo;
  const canvas = elements.enrollCanvas;
  const ctx = canvas.getContext('2d');

  if (video.readyState >= 2 && !video.paused) {
    const detection = await faceapi.detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })
    );

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (detection) {
      const { box } = detection;
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
      ctx.font = '600 12px Inter, sans-serif';
      ctx.fillText('Face Detected — Ready to Snap', box.x, Math.max(16, box.y - 8));
    }
  }

  state.enrollAnimId = requestAnimationFrame(runEnrollPreviewLoop);
}

function snapCurrentFace() {
  if (!state.enrollCamActive) return;

  const video = elements.enrollVideo;
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const sCtx = snapCanvas.getContext('2d');
  // Draw without mirroring for storage
  sCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);

  const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.88);
  addSnapThumbnail(dataUrl);
  showToast(`Captured snapshot #${state.capturedSnaps.length}`, 'success');
}

function addSnapThumbnail(dataUrl) {
  state.capturedSnaps.push(dataUrl);
  renderCapturedThumbs();
}

function renderCapturedThumbs() {
  elements.thumbsContainer.innerHTML = '';
  elements.snapCountBadge.textContent = `${state.capturedSnaps.length} images`;
  elements.btnSaveEnrollment.disabled = state.capturedSnaps.length === 0;

  if (state.capturedSnaps.length === 0) {
    elements.thumbsContainer.innerHTML = '<div class="thumbs-empty"><span>No snapshots yet</span></div>';
    return;
  }

  state.capturedSnaps.forEach((dataUrl, idx) => {
    const item = document.createElement('div');
    item.className = 'thumb-item';

    const img = document.createElement('img');
    img.src = dataUrl;
    img.alt = `Snap ${idx + 1}`;

    const rmBtn = document.createElement('button');
    rmBtn.className = 'thumb-remove';
    rmBtn.innerHTML = '✕';
    rmBtn.title = 'Remove snapshot';
    rmBtn.onclick = (e) => {
      e.stopPropagation();
      state.capturedSnaps.splice(idx, 1);
      renderCapturedThumbs();
    };

    item.appendChild(img);
    item.appendChild(rmBtn);
    elements.thumbsContainer.appendChild(item);
  });
}

async function handleFileUpload(event) {
  const files = Array.from(event.target.files);
  if (!files.length) return;

  showToast(`Processing ${files.length} uploaded images...`, 'info');
  for (const file of files) {
    const dataUrl = await readFileAsDataURL(file);
    state.capturedSnaps.push(dataUrl);
  }
  renderCapturedThumbs();
  event.target.value = ''; // Reset input
  showToast(`Added ${files.length} images to queue`, 'success');
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Compute Biometric Centroid & Save Enrolled Identity
async function saveEnrollment() {
  const name = elements.enrollNameInput.value.trim();
  if (!name) {
    showToast('Please enter a person name', 'error');
    elements.enrollNameInput.focus();
    return;
  }

  if (state.capturedSnaps.length === 0) {
    showToast('Please capture or upload at least one image', 'error');
    return;
  }

  showToast(`Computing 128D embeddings for "${name}"...`, 'info');
  elements.btnSaveEnrollment.disabled = true;

  try {
    const descriptors = [];
    const validSamples = [];

    for (let i = 0; i < state.capturedSnaps.length; i++) {
      const img = new Image();
      img.src = state.capturedSnaps[i];
      await new Promise(r => { img.onload = r; });

      const detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.3 }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection && detection.descriptor) {
        descriptors.push(detection.descriptor);
        validSamples.push(state.capturedSnaps[i]);
      }
    }

    if (descriptors.length === 0) {
      showToast('No clear faces detected in captured images. Try again.', 'error');
      elements.btnSaveEnrollment.disabled = false;
      return;
    }

    // Compute Centroid (Mean Descriptor Vector)
    const avgDescriptor = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
      let sum = 0;
      for (let j = 0; j < descriptors.length; j++) {
        sum += descriptors[j][i];
      }
      avgDescriptor[i] = sum / descriptors.length;
    }

    // Check if person already exists (merge or overwrite)
    const existingIdx = state.gallery.findIndex(p => p.name.toLowerCase() === name.toLowerCase());
    if (existingIdx >= 0) {
      state.gallery[existingIdx].descriptor = avgDescriptor;
      state.gallery[existingIdx].samples = [
        ...state.gallery[existingIdx].samples,
        ...validSamples.slice(0, 8)
      ].slice(-16);
    } else {
      state.gallery.push({
        id: 'id_' + Date.now(),
        name: name,
        descriptor: avgDescriptor,
        samples: validSamples.slice(0, 12)
      });
    }

    saveGalleryToStorage();

    // Reset Form
    elements.enrollNameInput.value = '';
    state.capturedSnaps = [];
    renderCapturedThumbs();
    stopEnrollCamera();

    showToast(`Successfully enrolled "${name}" with ${descriptors.length} face descriptors!`, 'success');
  } catch (err) {
    console.error('Enrollment error:', err);
    showToast('Failed to enroll face profile', 'error');
    elements.btnSaveEnrollment.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  GALLERY MANAGER CONTROLLER
// ═══════════════════════════════════════════════════════════════════
function renderGalleryList() {
  const container = elements.personsList;
  container.innerHTML = '';

  if (state.gallery.length === 0) {
    container.innerHTML = `
      <div class="empty-list-notice">
        <span>No enrolled identities yet.<br>Use "Enroll Faces" to add users.</span>
      </div>
    `;
    elements.photoGrid.innerHTML = `
      <div class="empty-grid-notice">
        <span class="icon">🖼</span>
        <p>No identities enrolled in database</p>
      </div>
    `;
    elements.selectedPersonTitle.textContent = 'No identities';
    elements.selectedPersonSub.textContent = 'Enroll identities to view profiles';
    elements.btnDeleteSelectedPerson.disabled = true;
    return;
  }

  state.gallery.forEach(person => {
    const item = document.createElement('div');
    item.className = 'person-item';
    if (person.id === state.selectedPersonId) item.classList.add('active');

    const name = document.createElement('span');
    name.className = 'person-name';
    name.textContent = person.name;

    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = `${(person.samples || []).length} photos`;

    item.appendChild(name);
    item.appendChild(badge);

    item.addEventListener('click', () => {
      selectPerson(person.id);
    });

    container.appendChild(item);
  });

  // Default select first person if none selected
  if (!state.selectedPersonId && state.gallery.length > 0) {
    selectPerson(state.gallery[0].id);
  }
}

function selectPerson(id) {
  state.selectedPersonId = id;
  const person = state.gallery.find(p => p.id === id);
  if (!person) return;

  // Highlight list item
  document.querySelectorAll('.person-item').forEach(el => {
    el.classList.toggle('active', el.querySelector('.person-name').textContent === person.name);
  });

  elements.selectedPersonTitle.textContent = person.name;
  elements.selectedPersonSub.textContent = `${(person.samples || []).length} enrolled photo samples • 128D ArcFace Vector Active`;
  elements.btnDeleteSelectedPerson.disabled = false;

  // Render Photo Grid
  const grid = elements.photoGrid;
  grid.innerHTML = '';

  if (!person.samples || person.samples.length === 0) {
    grid.innerHTML = '<div class="empty-grid-notice"><p>No photo samples saved</p></div>';
    return;
  }

  person.samples.forEach(src => {
    const cell = document.createElement('div');
    cell.className = 'photo-cell';
    const img = document.createElement('img');
    img.src = src;
    img.alt = person.name;
    cell.appendChild(img);
    grid.appendChild(cell);
  });
}

function deleteSelectedPerson() {
  if (!state.selectedPersonId) return;
  const person = state.gallery.find(p => p.id === state.selectedPersonId);
  if (!person) return;

  if (confirm(`Are you sure you want to delete "${person.name}" from biometric database?`)) {
    state.gallery = state.gallery.filter(p => p.id !== state.selectedPersonId);
    state.selectedPersonId = null;
    saveGalleryToStorage();
    renderGalleryList();
    showToast(`Deleted "${person.name}" from gallery`, 'info');
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SETTINGS & BACKUP CONTROLLER
// ═══════════════════════════════════════════════════════════════════
async function populateCameraDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');
    const select = elements.selectCameraDevice;
    select.innerHTML = '<option value="default">Default Camera</option>';

    videoInputs.forEach((device, idx) => {
      const opt = document.createElement('option');
      opt.value = device.deviceId;
      opt.textContent = device.label || `Camera ${idx + 1}`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.warn('Could not enumerate cameras:', err);
  }
}

function exportGalleryDatabase() {
  if (state.gallery.length === 0) {
    showToast('Gallery is empty. Nothing to export.', 'error');
    return;
  }

  const exportData = state.gallery.map(p => ({
    id: p.id,
    name: p.name,
    descriptor: Array.from(p.descriptor),
    samples: p.samples || []
  }));

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `neuroface_biometric_backup_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported biometric database backup!', 'success');
}

function importGalleryDatabase(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    try {
      const imported = JSON.parse(e.target.result);
      if (!Array.isArray(imported)) throw new Error('Invalid format');

      state.gallery = imported.map(item => ({
        ...item,
        descriptor: new Float32Array(item.descriptor)
      }));

      saveGalleryToStorage();
      renderGalleryList();
      showToast(`Imported ${state.gallery.length} identities successfully!`, 'success');
    } catch (err) {
      console.error('Import error:', err);
      showToast('Failed to import database: Invalid JSON format', 'error');
    }
  };
  reader.readAsText(file);
  event.target.value = '';
}

function clearAllData() {
  if (confirm('⚠️ WARNING: This will delete ALL enrolled identities and face models. Continue?')) {
    state.gallery = [];
    state.selectedPersonId = null;
    saveGalleryToStorage();
    renderGalleryList();
    showToast('Biometric gallery wiped clean', 'info');
  }
}

// ═══════════════════════════════════════════════════════════════════
//  EVENT LISTENERS SETUP
// ═══════════════════════════════════════════════════════════════════
function setupEventListeners() {
  // Model loader buttons
  elements.btnLoadModel.addEventListener('click', loadAIModels);
  elements.btnBuildGallery.addEventListener('click', () => {
    buildFaceMatcher();
    showToast('Biometric vector matrix rebuilt!', 'success');
  });

  // Live Recognition controls
  elements.btnToggleCamera.addEventListener('click', toggleCamera);
  elements.thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    state.threshold = val;
    elements.thresholdVal.textContent = val.toFixed(2);
    buildFaceMatcher();
  });
  elements.chkShowLandmarks.addEventListener('change', (e) => {
    state.showLandmarks = e.target.checked;
  });
  elements.chkShowBoxes.addEventListener('change', (e) => {
    state.showBoxes = e.target.checked;
  });

  // Enroll controls
  elements.btnStartEnrollCam.addEventListener('click', startEnrollCamera);
  elements.btnStopEnrollCam.addEventListener('click', stopEnrollCamera);
  elements.btnSnapFace.addEventListener('click', snapCurrentFace);
  elements.btnUploadPhotos.addEventListener('click', () => elements.fileUploadInput.click());
  elements.fileUploadInput.addEventListener('change', handleFileUpload);
  elements.btnSaveEnrollment.addEventListener('click', saveEnrollment);
  elements.btnClearSnaps.addEventListener('click', () => {
    state.capturedSnaps = [];
    renderCapturedThumbs();
  });

  // Gallery controls
  elements.btnRefreshGallery.addEventListener('click', () => {
    loadGalleryFromStorage();
    renderGalleryList();
    showToast('Refreshed gallery identities', 'info');
  });
  elements.btnDeleteSelectedPerson.addEventListener('click', deleteSelectedPerson);

  // Settings controls
  elements.btnExportGallery.addEventListener('click', exportGalleryDatabase);
  elements.btnImportGallery.addEventListener('click', () => elements.importFileInput.click());
  elements.importFileInput.addEventListener('change', importGalleryDatabase);
  elements.btnClearAllData.addEventListener('click', clearAllData);
}

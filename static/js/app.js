document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const modelSelect = document.getElementById('modelSelect');
  const webcamBtn = document.getElementById('webcamBtn');
  const webcamVideo = document.getElementById('webcamVideo');
  const webcamCanvas = document.getElementById('webcamCanvas');
  
  const topClassName = document.getElementById('topClassName');
  const topConfidencePill = document.getElementById('topConfidencePill');
  const latencyBadge = document.getElementById('latencyBadge');
  const predictionsList = document.getElementById('predictionsList');

  const tabGradCam = document.getElementById('tabGradCam');
  const tabSaliency = document.getElementById('tabSaliency');
  const tabBenchmark = document.getElementById('tabBenchmark');
  const visualContainer = document.getElementById('visualContainer');
  const visualImage = document.getElementById('visualImage');
  const placeholderText = document.getElementById('placeholderText');
  const benchmarkContainer = document.getElementById('benchmarkContainer');
  const benchmarkBody = document.getElementById('benchmarkBody');

  let currentResult = null;
  let currentTab = 'gradcam';
  let currentImageBase64 = null;

  // File Upload Handlers
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      currentImageBase64 = event.target.result;
      classifyImage(currentImageBase64);
    };
    reader.readAsDataURL(file);
  }

  // Model Selection Change Listener
  modelSelect.addEventListener('change', () => {
    if (currentImageBase64) {
      classifyImage(currentImageBase64);
    }
  });

  // Sample Chips Handling - Load Photorealistic Sample Images
  document.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const label = chip.getAttribute('data-label');
      loadSampleImageFile(label);
    });
  });

  function loadSampleImageFile(label) {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      currentImageBase64 = canvas.toDataURL('image/jpeg');
      classifyImage(currentImageBase64);
    };
    img.onerror = () => {
      const b64 = generateFallbackCanvasImage(label);
      currentImageBase64 = b64;
      classifyImage(b64);
    };
    img.src = `/static/samples/${label}.jpg`;
  }

  function generateFallbackCanvasImage(label) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#ffffff';
    ctx.font = '64px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, 112, 112);
    return canvas.toDataURL('image/jpeg');
  }

  // Live Webcam Integration
  let stream = null;
  webcamBtn.addEventListener('click', async () => {
    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = stream;
        webcamVideo.style.display = 'block';
        webcamBtn.innerHTML = '<i class="fa-solid fa-camera"></i> Snap Photo & Classify';
      } catch (err) {
        alert('Webcam access error or permission denied.');
      }
    } else {
      webcamCanvas.width = webcamVideo.videoWidth;
      webcamCanvas.height = webcamVideo.videoHeight;
      const ctx = webcamCanvas.getContext('2d');
      ctx.drawImage(webcamVideo, 0, 0);
      currentImageBase64 = webcamCanvas.toDataURL('image/jpeg');
      
      stream.getTracks().forEach(track => track.stop());
      stream = null;
      webcamVideo.style.display = 'none';
      webcamBtn.innerHTML = '<i class="fa-solid fa-camera"></i> Capture from Live Webcam';

      classifyImage(currentImageBase64);
    }
  });

  // Main Classification API Call
  async function classifyImage(imageBase64) {
    const selectedModel = modelSelect.value;
    topClassName.textContent = 'Analyzing...';
    topConfidencePill.textContent = '--%';

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          image: imageBase64
        })
      });

      const data = await response.json();
      if (data.success) {
        currentResult = data;
        renderResults(data);
      } else {
        alert('Error: ' + data.error);
      }
    } catch (err) {
      console.error('Classification error:', err);
    }
  }

  function renderResults(data) {
    topClassName.textContent = data.top_class;
    topConfidencePill.textContent = `${data.top_confidence.toFixed(1)}%`;
    latencyBadge.innerHTML = `<i class="fa-solid fa-stopwatch"></i> ${data.latency_ms} ms`;

    // Render Progress Bars for Top-5 Predictions
    predictionsList.innerHTML = '';
    data.predictions.forEach(item => {
      const predElem = document.createElement('div');
      predElem.className = 'pred-item';
      predElem.innerHTML = `
        <div class="pred-meta">
          <span style="text-transform:capitalize; font-weight:500;">${item.class}</span>
          <span style="color:var(--text-muted); font-size:12px;">${item.confidence.toFixed(1)}%</span>
        </div>
        <div class="pred-bar-bg">
          <div class="pred-bar-fill" style="width: 0%;"></div>
        </div>
      `;
      predictionsList.appendChild(predElem);

      setTimeout(() => {
        predElem.querySelector('.pred-bar-fill').style.width = `${item.confidence}%`;
      }, 50);
    });

    updateVisualTab();
  }

  // Explainability Tabs Switching
  tabGradCam.addEventListener('click', () => {
    setActiveTab(tabGradCam, 'gradcam');
  });

  tabSaliency.addEventListener('click', () => {
    setActiveTab(tabSaliency, 'saliency');
  });

  tabBenchmark.addEventListener('click', () => {
    setActiveTab(tabBenchmark, 'benchmark');
    loadBenchmarks();
  });

  function setActiveTab(btn, tabName) {
    [tabGradCam, tabSaliency, tabBenchmark].forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTab = tabName;

    if (tabName === 'benchmark') {
      visualContainer.style.display = 'none';
      benchmarkContainer.style.display = 'block';
    } else {
      visualContainer.style.display = 'flex';
      benchmarkContainer.style.display = 'none';
      updateVisualTab();
    }
  }

  function updateVisualTab() {
    if (!currentResult) return;
    placeholderText.style.display = 'none';
    visualImage.style.display = 'block';

    if (currentTab === 'gradcam') {
      visualImage.src = currentResult.gradcam_image;
    } else if (currentTab === 'saliency') {
      visualImage.src = currentResult.saliency_image;
    }
  }

  async function loadBenchmarks() {
    try {
      const res = await fetch('/api/benchmark');
      const data = await res.json();
      benchmarkBody.innerHTML = '';
      data.benchmarks.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${item.model}</strong></td>
          <td>${item.top1_acc}</td>
          <td>${item.top3_acc}</td>
          <td>${item.latency_ms}</td>
          <td><span class="badge-tag">${item.type}</span></td>
        `;
        benchmarkBody.appendChild(row);
      });
    } catch (err) {
      console.error('Failed to load benchmarks:', err);
    }
  }

});

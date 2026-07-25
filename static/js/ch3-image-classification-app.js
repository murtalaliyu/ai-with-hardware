(() => {
  const form = document.getElementById('upload-form');
  const fileInput = document.getElementById('file-input');
  const dropTarget = document.getElementById('drop-target');
  const browseBtn = document.getElementById('browse-btn');
  const previewPanel = document.getElementById('preview-panel');
  const previewImage = document.getElementById('preview-image');
  const predictBtn = document.getElementById('predict-btn');
  const clearBtn = document.getElementById('clear-btn');
  const result = document.getElementById('result');
  const resultLabel = document.getElementById('result-label');
  const resultMeta = document.getElementById('result-meta');
  const resultNote = document.getElementById('result-note');
  const meterFill = document.getElementById('meter-fill');
  const modelStatus = document.getElementById('model-status');

  let selectedFile = null;
  let previewUrl = null;

  async function refreshStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.model_file_present) {
        modelStatus.textContent = data.model_loaded
          ? 'Model ready'
          : 'Model found — will load on first classify';
        modelStatus.className = 'status ready';
      } else {
        modelStatus.textContent = 'No model file — train with the CLI first';
        modelStatus.className = 'status error';
      }
    } catch {
      modelStatus.textContent = 'Could not reach server';
      modelStatus.className = 'status error';
    }
  }

  function setFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      return;
    }
    selectedFile = file;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    dropTarget.hidden = true;
    previewPanel.hidden = false;
    result.hidden = true;
    resultNote.hidden = true;
  }

  function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    previewImage.removeAttribute('src');
    previewPanel.hidden = true;
    dropTarget.hidden = false;
    result.hidden = true;
    resultNote.hidden = true;
    meterFill.style.width = '0%';
  }

  function showResult(data) {
    const isDog = data.label === 'Dog';
    result.hidden = false;
    resultLabel.textContent = data.label;
    resultLabel.className = `result-label ${isDog ? 'is-dog' : 'is-cat'}`;
    meterFill.className = `meter-fill ${isDog ? 'is-dog' : 'is-cat'}`;
    // Retrigger width transition
    meterFill.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        meterFill.style.width = `${Math.round(data.confidence * 100)}%`;
      });
    });
    resultMeta.textContent = `${Math.round(data.confidence * 100)}% confidence · ${data.filename || 'upload'}`;

    if (data.uncertain) {
      resultNote.hidden = false;
      resultNote.textContent =
        'Confidence is modest. This binary model always picks cat or dog — the photo may not be a pet.';
    } else {
      resultNote.hidden = true;
    }
  }

  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropTarget.addEventListener('click', () => fileInput.click());
  dropTarget.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) {
      setFile(fileInput.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    dropTarget.addEventListener(evt, (e) => {
      e.preventDefault();
      dropTarget.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((evt) => {
    dropTarget.addEventListener(evt, (e) => {
      e.preventDefault();
      dropTarget.classList.remove('dragover');
    });
  });

  dropTarget.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    setFile(file);
  });

  clearBtn.addEventListener('click', clearSelection);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = 'Classifying…';

    const body = new FormData();
    body.append('image', selectedFile);

    try {
      const res = await fetch('/api/predict', { method: 'POST', body });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Prediction failed');
      }
      showResult(data);
      refreshStatus();
    } catch (err) {
      result.hidden = false;
      resultLabel.textContent = 'Error';
      resultLabel.className = 'result-label';
      resultMeta.textContent = err.message;
      resultNote.hidden = true;
      meterFill.style.width = '0%';
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = 'Classify';
    }
  });

  refreshStatus();
})();

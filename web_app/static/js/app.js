/* ============================================================
   🚦 Vietnamese Traffic Sign Detection - Web App JavaScript
   ============================================================ */

// ============================================================
// Tab Switching
// ============================================================
function switchTab(tabName) {
    // Update tabs
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

// ============================================================
// Slider Sync
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');
    if (confSlider) {
        confSlider.addEventListener('input', function() {
            confValue.textContent = parseFloat(this.value).toFixed(2);
            this.style.background = `linear-gradient(to right, #1a73e8 0%, #1a73e8 ${(this.value - 0.05) / 0.9 * 100}%, #ddd ${(this.value - 0.05) / 0.9 * 100}%)`;
        });
    }

    const videoConfSlider = document.getElementById('videoConfSlider');
    const videoConfValue = document.getElementById('videoConfValue');
    if (videoConfSlider) {
        videoConfSlider.addEventListener('input', function() {
            videoConfValue.textContent = parseFloat(this.value).toFixed(2);
            this.style.background = `linear-gradient(to right, #1a73e8 0%, #1a73e8 ${(this.value - 0.05) / 0.9 * 100}%, #ddd ${(this.value - 0.05) / 0.9 * 100}%)`;
        });
    }

    // Drag & Drop
    setupDragDrop('uploadZone', 'imageInput');
    setupDragDrop('videoUploadZone', 'videoInput');
});

function setupDragDrop(zoneId, inputId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    
    if (!zone || !input) return;

    zone.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    zone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });

    zone.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }
    });
}

// ============================================================
// Image Detection
// ============================================================
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show original preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('originalPreview');
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Show loading
    document.getElementById('imageLoading').classList.remove('hidden');
    document.getElementById('imageResult').classList.add('hidden');

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', document.getElementById('confSlider').value);
    formData.append('enhance', document.getElementById('enhanceCheck').checked);
    formData.append('clahe', document.getElementById('claheCheck').checked);
    formData.append('sharpen', document.getElementById('sharpenCheck').checked);

    try {
        const response = await fetch('/api/detect/image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Lỗi xử lý ảnh');
        }

        const data = await response.json();

        // Show result
        document.getElementById('resultPreview').src = data.output_image + '?t=' + Date.now();
        document.getElementById('totalSigns').textContent = data.total_signs;

        // Render detection tags
        const list = document.getElementById('detectionList');
        list.innerHTML = '';
        
        const colors = ['#e74c3c', '#e67e22', '#3498db', '#27ae60', '#9b59b6', '#1abc9c', '#f39c12', '#2c3e50'];
        data.detections.forEach((det, i) => {
            const tag = document.createElement('span');
            tag.className = 'detection-tag';
            tag.style.background = colors[i % colors.length];
            tag.innerHTML = `${det.class_name_vi} <span class="conf">${(det.confidence * 100).toFixed(0)}%</span>`;
            list.appendChild(tag);
        });

        document.getElementById('imageLoading').classList.add('hidden');
        document.getElementById('imageResult').classList.remove('hidden');
        
        // Scroll to result
        document.getElementById('imageResult').scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        document.getElementById('imageLoading').classList.add('hidden');
        alert('❌ ' + error.message);
    }
}

// ============================================================
// Video Detection
// ============================================================
async function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show loading
    document.getElementById('videoLoading').classList.remove('hidden');
    document.getElementById('videoResult').classList.add('hidden');
    
    // Animate progress
    const progressFill = document.getElementById('videoProgress');
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 5, 90);
        progressFill.style.width = progress + '%';
    }, 2000);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', document.getElementById('videoConfSlider').value);
    formData.append('enhance', document.getElementById('videoEnhanceCheck').checked);

    try {
        const response = await fetch('/api/detect/video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Lỗi xử lý video');
        }

        const data = await response.json();

        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        // Show result
        setTimeout(() => {
            const video = document.getElementById('videoResultPlayer');
            video.src = data.output_video + '?t=' + Date.now();
            
            const stats = document.getElementById('videoStats');
            stats.innerHTML = `
                <span class="stat">📊 Tổng số frame: <strong>${data.total_frames}</strong></span>
                <span class="stat">⚡ Đã xử lý: <strong>${data.processed_frames}</strong> frame</span>
            `;

            document.getElementById('videoLoading').classList.add('hidden');
            document.getElementById('videoResult').classList.remove('hidden');
            
            // Scroll to result
            document.getElementById('videoResult').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        document.getElementById('videoLoading').classList.add('hidden');
        alert('❌ ' + error.message);
    }
}

// ============================================================
// Filter Classes
// ============================================================
function filterClasses() {
    const query = document.getElementById('classSearch').value.toLowerCase();
    const items = document.querySelectorAll('.class-item');
    
    items.forEach(item => {
        const name = item.dataset.name.toLowerCase();
        const code = item.dataset.code.toLowerCase();
        if (name.includes(query) || code.includes(query)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
}

/**
 * DeepFake Detector - Main JavaScript
 * ====================================
 * Handles file uploads, API communication, and UI interactions
 */

// ============================================
// Global State
// ============================================

const state = {
    selectedFile: null,
    selectedFiles: [],
    isProcessing: false,
    currentResult: null
};

// ============================================
// DOM Elements
// ============================================

const elements = {
    // Single Upload
    uploadZone: document.getElementById('upload-zone'),
    fileInput: document.getElementById('file-input'),
    previewSection: document.getElementById('preview-section'),
    previewMedia: document.getElementById('preview-media'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    removeFile: document.getElementById('remove-file'),
    optionsSection: document.getElementById('options-section'),
    videoOptions: document.getElementById('video-options'),
    generateHeatmap: document.getElementById('generate-heatmap'),
    sampleRate: document.getElementById('sample-rate'),
    analyzeBtn: document.getElementById('analyze-btn'),

    // Batch Upload
    batchUploadZone: document.getElementById('batch-upload-zone'),
    batchFileInput: document.getElementById('batch-file-input'),
    batchPreview: document.getElementById('batch-preview'),
    batchList: document.getElementById('batch-list'),
    batchCount: document.getElementById('batch-count'),
    clearBatch: document.getElementById('clear-batch'),
    batchAnalyzeBtn: document.getElementById('batch-analyze-btn'),

    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Processing
    processingOverlay: document.getElementById('processing-overlay'),
    processingStatus: document.getElementById('processing-status'),
    progressFill: document.getElementById('progress-fill'),
    processingDetails: document.getElementById('processing-details'),

    // Results
    resultsSection: document.getElementById('results-section')
};

// ============================================
// Utility Functions
// ============================================

/**
 * Format file size to human readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Check if file is an image
 */
function isImage(file) {
    return file.type.startsWith('image/');
}

/**
 * Check if file is a video
 */
function isVideo(file) {
    return file.type.startsWith('video/');
}

/**
 * Get file icon based on type
 */
function getFileIcon(file) {
    if (isImage(file)) return 'fa-image';
    if (isVideo(file)) return 'fa-video';
    return 'fa-file';
}

/**
 * Show/hide element
 */
function toggleElement(element, show) {
    if (element) {
        element.classList.toggle('hidden', !show);
    }
}

/**
 * Animate element entrance
 */
function animateIn(element) {
    element.style.opacity = '0';
    element.style.transform = 'translateY(20px)';
    toggleElement(element, true);
    
    requestAnimationFrame(() => {
        element.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
    });
}

// ============================================
// Tab Management
// ============================================

function initTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // Update active states
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            elements.tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
            
            // Reset state when switching tabs
            resetUpload();
        });
    });
}

// ============================================
// Single File Upload
// ============================================

function initSingleUpload() {
    // Click to upload
    elements.uploadZone.addEventListener('click', () => {
        elements.fileInput.click();
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop
    elements.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.add('drag-over');
    });

    elements.uploadZone.addEventListener('dragleave', () => {
        elements.uploadZone.classList.remove('drag-over');
    });

    elements.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Remove file
    elements.removeFile.addEventListener('click', resetUpload);

    // Analyze button
    elements.analyzeBtn.addEventListener('click', analyzeFile);
}

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    if (!isImage(file) && !isVideo(file)) {
        showNotification('Please select an image or video file', 'error');
        return;
    }

    state.selectedFile = file;
    
    // Update UI
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    
    // Show preview
    showPreview(file);
    
    // Show options
    animateIn(elements.optionsSection);
    
    // Show/hide video options
    toggleElement(elements.videoOptions, isVideo(file));
    
    // Show analyze button
    animateIn(elements.analyzeBtn);
}

/**
 * Show file preview
 */
function showPreview(file) {
    elements.previewMedia.innerHTML = '';
    
    if (isImage(file)) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = 'Preview';
        elements.previewMedia.appendChild(img);
    } else if (isVideo(file)) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.controls = true;
        video.muted = true;
        elements.previewMedia.appendChild(video);
    }
    
    animateIn(elements.previewSection);
}

/**
 * Reset upload state
 */
function resetUpload() {
    state.selectedFile = null;
    elements.fileInput.value = '';
    
    toggleElement(elements.previewSection, false);
    toggleElement(elements.optionsSection, false);
    toggleElement(elements.analyzeBtn, false);
    toggleElement(elements.resultsSection, false);
}

// ============================================
// Batch Upload
// ============================================

function initBatchUpload() {
    // Click to upload
    elements.batchUploadZone.addEventListener('click', () => {
        elements.batchFileInput.click();
    });

    // File input change
    elements.batchFileInput.addEventListener('change', (e) => {
        handleBatchSelect(Array.from(e.target.files));
    });

    // Drag and drop
    elements.batchUploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.batchUploadZone.classList.add('drag-over');
    });

    elements.batchUploadZone.addEventListener('dragleave', () => {
        elements.batchUploadZone.classList.remove('drag-over');
    });

    elements.batchUploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.batchUploadZone.classList.remove('drag-over');
        handleBatchSelect(Array.from(e.dataTransfer.files));
    });

    // Clear batch
    elements.clearBatch.addEventListener('click', resetBatch);

    // Batch analyze
    elements.batchAnalyzeBtn.addEventListener('click', analyzeBatch);
}

/**
 * Handle batch file selection
 */
function handleBatchSelect(files) {
    // Filter valid files
    const validFiles = files.filter(f => isImage(f) || isVideo(f));
    
    if (validFiles.length === 0) {
        showNotification('No valid image or video files selected', 'error');
        return;
    }
    
    // Limit to 10 files
    const limitedFiles = validFiles.slice(0, 10);
    if (validFiles.length > 10) {
        showNotification('Maximum 10 files allowed. Only first 10 will be processed.', 'warning');
    }
    
    state.selectedFiles = limitedFiles;
    updateBatchPreview();
}

/**
 * Update batch preview list
 */
function updateBatchPreview() {
    elements.batchList.innerHTML = '';
    
    state.selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'batch-item';
        item.innerHTML = `
            <i class="fas ${getFileIcon(file)}"></i>
            <span class="file-name">${file.name.substring(0, 20)}${file.name.length > 20 ? '...' : ''}</span>
            <button class="remove-item" data-index="${index}">
                <i class="fas fa-times"></i>
            </button>
        `;
        elements.batchList.appendChild(item);
    });
    
    // Add remove listeners
    document.querySelectorAll('.batch-item .remove-item').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.currentTarget.dataset.index);
            state.selectedFiles.splice(index, 1);
            updateBatchPreview();
        });
    });
    
    // Update count
    elements.batchCount.textContent = `${state.selectedFiles.length} file(s) selected`;
    
    // Show/hide elements
    toggleElement(elements.batchPreview, state.selectedFiles.length > 0);
    toggleElement(elements.batchAnalyzeBtn, state.selectedFiles.length > 0);
}

/**
 * Reset batch state
 */
function resetBatch() {
    state.selectedFiles = [];
    elements.batchFileInput.value = '';
    updateBatchPreview();
}

// ============================================
// Analysis Functions
// ============================================

/**
 * Analyze single file
 */
async function analyzeFile() {
    if (!state.selectedFile || state.isProcessing) return;
    
    state.isProcessing = true;
    showProcessing(true);
    
    try {
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        formData.append('generate_heatmap', elements.generateHeatmap.checked.toString());
        
        // Add video options if applicable
        if (isVideo(state.selectedFile)) {
            formData.append('sample_rate', elements.sampleRate.value);
        }
        
        // Determine endpoint
        const endpoint = isVideo(state.selectedFile) 
            ? '/api/analyze/video' 
            : '/api/analyze/image';
        
        // Update processing message
        updateProcessingStatus(
            isVideo(state.selectedFile) ? 'Analyzing video frames...' : 'Analyzing image...',
            'Running AI detection models...'
        );
        
        // Simulate progress
        simulateProgress();
        
        // Make request
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.currentResult = {
                type: isVideo(state.selectedFile) ? 'video' : 'image',
                filename: state.selectedFile.name,
                result: data.result
            };
            
            // Store in session storage for results page
            sessionStorage.setItem('analysisResult', JSON.stringify(state.currentResult));
            
            // Show results
            displayResults(state.currentResult);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(error.message || 'An error occurred during analysis', 'error');
    } finally {
        state.isProcessing = false;
        showProcessing(false);
    }
}

/**
 * Analyze batch files
 */
async function analyzeBatch() {
    if (state.selectedFiles.length === 0 || state.isProcessing) return;
    
    state.isProcessing = true;
    showProcessing(true);
    
    try {
        const formData = new FormData();
        state.selectedFiles.forEach(file => {
            formData.append('files[]', file);
        });
        
        updateProcessingStatus(
            `Analyzing ${state.selectedFiles.length} files...`,
            'This may take a few minutes...'
        );
        
        simulateProgress(60000); // Longer timeout for batch
        
        const response = await fetch('/api/analyze/batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data);
        } else {
            throw new Error(data.error || 'Batch analysis failed');
        }
    } catch (error) {
        console.error('Batch analysis error:', error);
        showNotification(error.message || 'An error occurred during batch analysis', 'error');
    } finally {
        state.isProcessing = false;
        showProcessing(false);
    }
}

// ============================================
// Results Display
// ============================================

/**
 * Display analysis results
 */
function displayResults(data) {
    const result = data.result;
    const isVideo = data.type === 'video';
    
    const prediction = isVideo ? result.overall_prediction : result.prediction;
    const confidence = isVideo ? result.overall_confidence : result.confidence;
    
    let html = `
        <div class="result-card ${prediction.toLowerCase().replace('-', '')}">
            <div class="result-header">
                <div class="result-badge ${getPredictionClass(prediction)}">
                    ${getResultIcon(prediction)}
                    <span>${prediction}</span>
                </div>
                <div class="result-confidence">
                    <span class="confidence-value">${confidence}%</span>
                    <span class="confidence-label">Confidence</span>
                </div>
            </div>

            <div class="result-details">
                <h3><i class="fas fa-file"></i> File Information</h3>
                <p><strong>Filename:</strong> ${data.filename}</p>
                <p><strong>Type:</strong> ${data.type.charAt(0).toUpperCase() + data.type.slice(1)}</p>
                <p><strong>Analyzed:</strong> ${new Date().toLocaleString()}</p>
            </div>
    `;

    if (isVideo) {
        html += `
            <div class="result-details">
                <h3><i class="fas fa-video"></i> Video Analysis</h3>
                <div class="stats-grid">
                    <div class="stat-box">
                        <span class="stat-number">${result.frames_analyzed}</span>
                        <span class="stat-text">Frames Analyzed</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">${result.suspicious_frames}</span>
                        <span class="stat-text">Suspicious Frames</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">${result.suspicious_percentage}%</span>
                        <span class="stat-text">Suspicious Rate</span>
                    </div>
                </div>
                <p class="summary-text">${result.summary}</p>
            </div>
        `;
    } else {
        html += `
            <div class="result-details">
                <h3><i class="fas fa-percentage"></i> Probability Scores</h3>
                <div class="probability-bars">
                    ${generateProbabilityBars(result.probabilities)}
                </div>
            </div>

            ${result.heatmap_path ? `
                <div class="result-details">
                    <h3><i class="fas fa-fire"></i> Attention Heatmap</h3>
                    <p class="heatmap-desc">Areas highlighted in red indicate regions the AI focused on for its decision.</p>
                    <img src="${result.heatmap_path}" alt="Heatmap" class="heatmap-image">
                </div>
            ` : ''}

            <div class="result-details">
                <h3><i class="fas fa-face-smile"></i> Face Detection</h3>
                <p>${result.face_detected ? `${result.face_regions.length} face(s) detected and analyzed` : 'No faces detected in image'}</p>
            </div>
        `;
    }

    html += `
        <div class="result-details">
            <h3><i class="fas fa-lightbulb"></i> Explanation</h3>
            <p class="explanation-text">${result.explanation || result.summary}</p>
        </div>

        ${result.artifacts && result.artifacts.length > 0 ? `
            <div class="result-details">
                <h3><i class="fas fa-search-plus"></i> Detected Artifacts</h3>
                <ul class="artifacts-list">
                    ${result.artifacts.map(a => `<li><i class="fas fa-check-circle"></i> ${a}</li>`).join('')}
                </ul>
            </div>
        ` : ''}

        <div class="result-actions">
            <button class="btn btn-primary" onclick="downloadReport()">
                <i class="fas fa-file-pdf"></i> Download Report
            </button>
            <button class="btn btn-outline" onclick="resetUpload(); window.scrollTo({top: 0, behavior: 'smooth'})">
                <i class="fas fa-redo"></i> Analyze Another
            </button>
        </div>
    </div>
    `;

    elements.resultsSection.innerHTML = html;
    animateIn(elements.resultsSection);
    
    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Display batch results
 */
function displayBatchResults(data) {
    let html = `
        <div class="result-card">
            <div class="result-header">
                <div class="result-badge">
                    <i class="fas fa-layer-group"></i>
                    <span>Batch Analysis Complete</span>
                </div>
                <div class="result-confidence">
                    <span class="confidence-value">${data.successful}/${data.total}</span>
                    <span class="confidence-label">Successful</span>
                </div>
            </div>

            <div class="result-details">
                <h3><i class="fas fa-list"></i> Results Summary</h3>
                <div class="batch-results-list">
    `;

    data.results.forEach(item => {
        const prediction = item.success 
            ? (item.result.prediction || item.result.overall_prediction)
            : 'Error';
        const confidence = item.success 
            ? (item.result.confidence || item.result.overall_confidence)
            : 0;

        html += `
            <div class="batch-result-item ${item.success ? getPredictionClass(prediction) : 'error'}">
                <div class="batch-result-info">
                    <i class="fas ${item.type === 'video' ? 'fa-video' : 'fa-image'}"></i>
                    <span class="batch-result-name">${item.filename}</span>
                </div>
                <div class="batch-result-status">
                    ${item.success 
                        ? `<span class="prediction">${prediction}</span>
                           <span class="confidence">${confidence}%</span>`
                        : `<span class="error-msg">${item.error}</span>`
                    }
                </div>
            </div>
        `;
    });

    html += `
                </div>
            </div>

            <div class="result-actions">
                <button class="btn btn-primary" onclick="downloadBatchReport()">
                    <i class="fas fa-file-pdf"></i> Download Batch Report
                </button>
                <button class="btn btn-outline" onclick="resetBatch(); window.scrollTo({top: 0, behavior: 'smooth'})">
                    <i class="fas fa-redo"></i> Analyze More Files
                </button>
            </div>
        </div>
    `;

    elements.resultsSection.innerHTML = html;
    animateIn(elements.resultsSection);
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ============================================
// Helper Functions for Results
// ============================================

function getPredictionClass(prediction) {
    switch(prediction.toLowerCase()) {
        case 'real': return 'real';
        case 'deepfake': return 'fake';
        case 'ai-generated': return 'ai-gen';
        default: return '';
    }
}

function getResultIcon(prediction) {
    switch(prediction.toLowerCase()) {
        case 'real': return '<i class="fas fa-check-circle"></i>';
        case 'deepfake': return '<i class="fas fa-mask"></i>';
        case 'ai-generated': return '<i class="fas fa-robot"></i>';
        default: return '<i class="fas fa-question-circle"></i>';
    }
}

function generateProbabilityBars(probabilities) {
    let html = '';
    for (const [label, value] of Object.entries(probabilities)) {
        const barClass = label.toLowerCase().includes('real') ? 'real' : 
                       label.toLowerCase().includes('deepfake') ? 'fake' : 'ai-gen';
        html += `
            <div class="prob-item">
                <div class="prob-label">${label}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar ${barClass}" style="width: ${value}%"></div>
                </div>
                <div class="prob-value">${value}%</div>
            </div>
        `;
    }
    return html;
}

// ============================================
// Processing UI
// ============================================

function showProcessing(show) {
    toggleElement(elements.processingOverlay, show);
    
    if (show) {
        document.body.style.overflow = 'hidden';
        elements.progressFill.style.width = '0%';
    } else {
        document.body.style.overflow = '';
    }
}

function updateProcessingStatus(status, details) {
    elements.processingStatus.textContent = status;
    elements.processingDetails.textContent = details;
}

function simulateProgress(duration = 30000) {
    let progress = 0;
    const interval = setInterval(() => {
        if (!state.isProcessing) {
            elements.progressFill.style.width = '100%';
            clearInterval(interval);
            return;
        }
        
        // Slow down as it approaches 90%
        const increment = Math.max(1, (90 - progress) / 20);
        progress = Math.min(90, progress + increment);
        elements.progressFill.style.width = `${progress}%`;
        
        // Update status messages
        if (progress > 30 && progress < 60) {
            updateProcessingStatus('Processing...', 'Detecting faces and analyzing features...');
        } else if (progress > 60) {
            updateProcessingStatus('Almost done...', 'Generating results and explanation...');
        }
    }, duration / 100);
}

// ============================================
// Report Download
// ============================================

async function downloadReport() {
    try {
        const response = await fetch('/api/report/generate', {
            method: 'POST'
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `deepfake_report_${Date.now()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            showNotification('Report downloaded successfully', 'success');
        } else {
            throw new Error('Failed to generate report');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error generating report', 'error');
    }
}

async function downloadBatchReport() {
    try {
        const response = await fetch('/api/report/batch', {
            method: 'POST'
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `batch_report_${Date.now()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            showNotification('Batch report downloaded successfully', 'success');
        } else {
            throw new Error('Failed to generate batch report');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error generating batch report', 'error');
    }
}

// ============================================
// Notifications
// ============================================

function showNotification(message, type = 'info') {
    // Remove existing notifications
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${getNotificationIcon(type)}"></i>
        <span>${message}</span>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;
    
    // Add styles
    Object.assign(notification.style, {
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        padding: '16px 24px',
        borderRadius: '12px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        zIndex: '9999',
        animation: 'slideIn 0.3s ease',
        background: type === 'success' ? 'rgba(16, 185, 129, 0.9)' :
                   type === 'error' ? 'rgba(239, 68, 68, 0.9)' :
                   type === 'warning' ? 'rgba(245, 158, 11, 0.9)' :
                   'rgba(99, 102, 241, 0.9)',
        color: 'white',
        boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)'
    });
    
    document.body.appendChild(notification);
    
    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });
    
    // Auto remove
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    switch(type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        default: return 'fa-info-circle';
    }
}

// ============================================
// API Status Check
// ============================================

async function checkAPIStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('API is healthy');
        }
    } catch (error) {
        console.warn('API health check failed:', error);
        showNotification('Warning: API may be unavailable', 'warning');
    }
}

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    initTabs();
    initSingleUpload();
    initBatchUpload();
    
    // Check API status
    checkAPIStatus();
    
    // Add notification animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(100px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        @keyframes slideOut {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100px);
            }
        }
        .notification-close {
            background: none;
            border: none;
            color: inherit;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity 0.2s;
        }
        .notification-close:hover {
            opacity: 1;
        }
        .batch-results-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .batch-result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            border-left: 4px solid;
        }
        .batch-result-item.real {
            border-color: var(--success);
        }
        .batch-result-item.fake {
            border-color: var(--danger);
        }
        .batch-result-item.ai-gen {
            border-color: var(--warning);
        }
        .batch-result-item.error {
            border-color: var(--gray-500);
        }
        .batch-result-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .batch-result-info i {
            color: var(--primary);
        }
        .batch-result-status {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .batch-result-status .prediction {
            font-weight: 600;
        }
        .batch-result-status .confidence {
            color: var(--text-secondary);
        }
        .batch-result-status .error-msg {
            color: var(--danger);
            font-size: 0.9rem;
        }
    `;
    document.head.appendChild(style);
    
    console.log('DeepFake Detector initialized');
});

// Make functions globally available
window.downloadReport = downloadReport;
window.downloadBatchReport = downloadBatchReport;
window.resetUpload = resetUpload;
window.resetBatch = resetBatch;

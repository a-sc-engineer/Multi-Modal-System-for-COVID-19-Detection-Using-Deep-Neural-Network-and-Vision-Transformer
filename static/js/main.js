document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const ctInput = document.getElementById('ct-input');
    const xrayInput = document.getElementById('xray-input');
    const ctPreview = document.getElementById('ct-preview');
    const xrayPreview = document.getElementById('xray-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('span');
    const btnLoader = document.getElementById('btn-loader');
    
    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    
    // Elements to update with results
    const predictionResult = document.getElementById('prediction-result');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');
    const probabilityDetails = document.getElementById('probability-details');

    // Handle file selection and preview
    function handleFileSelect(inputElement, previewElement) {
        inputElement.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewElement.src = e.target.result;
                    previewElement.classList.add('show');
                }
                reader.readAsDataURL(file);
            } else {
                previewElement.src = '';
                previewElement.classList.remove('show');
            }
        });
    }

    handleFileSelect(ctInput, ctPreview);
    handleFileSelect(xrayInput, xrayPreview);

    // Handle drag and drop visuals
    function setupDragAndDrop(dropzoneId) {
        const dropzone = document.getElementById(dropzoneId);
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => {
                dropzone.querySelector('.drop-area').classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => {
                dropzone.querySelector('.drop-area').classList.remove('dragover');
            }, false);
        });
    }

    setupDragAndDrop('ct-dropzone');
    setupDragAndDrop('xray-dropzone');

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!ctInput.files[0] || !xrayInput.files[0]) {
            showError("Please upload both CT Scan and X-Ray images.");
            return;
        }

        // Reset UI state
        hideError();
        resultsContainer.classList.add('hidden');
        setLoadingState(true);

        // Prepare data
        const formData = new FormData();
        formData.append('ct_image', ctInput.files[0]);
        formData.append('xray_image', xrayInput.files[0]);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "An error occurred during prediction.");
            }

            displayResults(data);
        } catch (error) {
            showError(error.message);
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        analyzeBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = 'Analyzing...';
            btnLoader.classList.remove('hidden');
        } else {
            btnText.textContent = 'Analyze Images';
            btnLoader.classList.add('hidden');
        }
    }

    function displayResults(data) {
        const { prediction, confidence, probabilities } = data;
        
        // Update Prediction Badge
        predictionResult.textContent = prediction;
        predictionResult.className = 'value badge'; // Reset classes
        if (prediction === 'COVID') {
            predictionResult.classList.add('covid');
        } else {
            predictionResult.classList.add('non-covid');
        }

        // Update Confidence Bar
        const confidencePercentage = (confidence * 100).toFixed(1) + '%';
        confidenceValue.textContent = confidencePercentage;
        
        // Small delay to allow CSS transition to play after un-hiding
        setTimeout(() => {
            confidenceFill.style.width = confidencePercentage;
        }, 100);

        // Update Probabilities breakdown
        probabilityDetails.innerHTML = `
            <span>COVID: ${(probabilities.COVID * 100).toFixed(1)}%</span>
            <span>NON-COVID: ${(probabilities.NON_COVID * 100).toFixed(1)}%</span>
        `;

        // Show results
        resultsContainer.classList.remove('hidden');
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('hidden');
    }

    function hideError() {
        errorContainer.classList.add('hidden');
    }
});

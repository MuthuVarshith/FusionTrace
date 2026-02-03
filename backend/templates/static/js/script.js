document.addEventListener('DOMContentLoaded', function() {
  // Configurable backend base URL
  const BACKEND_BASE_URL = 'http://localhost:8000'; // Adjust if backend runs on a different host/port

  // Navbar scroll effect
  const navbar = document.querySelector('.navbar');
  
  function handleScroll() {
    if (window.scrollY > 10) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }
  
  window.addEventListener('scroll', handleScroll);
  
  // Mobile menu functionality
  const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
  const navLinksContainer = document.querySelector('.nav-links-container');
  const navButtons = document.querySelector('.nav-buttons');
  
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', function() {
      navLinksContainer.classList.toggle('active');
      navButtons.classList.toggle('active');
    });
  }

  // Handle smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth'
        });
        
        // Close mobile menu if open
        navLinksContainer.classList.remove('active');
        navButtons.classList.remove('active');
      }
    });
  });
  
  // Login and Register button functionality
  const loginBtn = document.querySelector('.login-btn');
  const registerBtn = document.querySelector('.btn-primary');
  
  if (loginBtn) {
    loginBtn.addEventListener('click', function(e) {
      e.preventDefault();
      window.location.href = '/login';
      // You could also show a modal instead:
      // showLoginModal();
    });
  }
  
  if (registerBtn && registerBtn.parentElement.classList.contains('nav-buttons')) {
    registerBtn.addEventListener('click', function(e) {
      e.preventDefault();
      window.location.href = '/register';
      // You could also show a modal instead:
      // showRegisterModal();
    });
  }

  // Upload & Detect functionality
  const uploadArea = document.getElementById('upload-area');
  const fileInput = document.getElementById('file-input');
  const fileInfo = document.getElementById('file-info');
  const analysisResults = document.getElementById('analysis-results');
  const fileName = document.getElementById('file-name');
  const fileMeta = document.getElementById('file-meta');
  const resultsContainer = document.getElementById('results-container');
  
  // File selection via button
  const selectFileBtn = document.querySelector('.select-file-btn');
  if (selectFileBtn) {
    selectFileBtn.addEventListener('click', function(e) {
      e.preventDefault();
      if (fileInput) {
        fileInput.click();
        console.log('Select File button clicked, triggering file input');
      } else {
        console.error('File input element not found');
        showNotification('Error: File input not found. Please check the page setup.', 'error');
      }
    });
  } else {
    console.error('Select File button not found');
    showNotification('Error: Select File button not found. Please check the page setup.', 'error');
  }
  
  // File selection via input change
  if (fileInput) {
    fileInput.addEventListener('change', handleFileSelection);
  } else {
    console.error('File input element not found');
    showNotification('Error: File input not found. Please check the page setup.', 'error');
  }
  
  // Drag and drop functionality
  if (uploadArea) {
    uploadArea.addEventListener('dragover', function(e) {
      e.preventDefault();
      this.classList.add('dragging');
    });
    
    uploadArea.addEventListener('dragleave', function() {
      this.classList.remove('dragging');
    });
    
    uploadArea.addEventListener('drop', function(e) {
      e.preventDefault();
      this.classList.remove('dragging');
      
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelection();
      }
    });
  }
  
  // Handle file selection
  function handleFileSelection() {
    if (!fileInput || fileInput.files.length === 0) {
      console.log('No file selected or file input not available');
      return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'audio/mp3', 'audio/wav'];
    if (!validTypes.includes(file.type)) {
      showNotification('Invalid file type. Please upload JPG, PNG, MP3, or WAV files only.', 'error');
      return;
    }
    
    // Validate file size (max 20MB)
    if (file.size > 20 * 1024 * 1024) {
      showNotification('File too large. Maximum file size is 20MB.', 'error');
      return;
    }
    
    // Update UI to show file info
    fileName.textContent = file.name;
    fileMeta.textContent = `${formatFileSize(file.size)} • ${file.type}`;
    
    uploadArea.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    analysisResults.classList.add('hidden');
    console.log(`File selected: ${file.name}, type: ${file.type}, size: ${file.size}`);
  }
  
  // Detect button functionality
  const detectBtn = document.querySelector('.detect-btn');
  if (detectBtn) {
    detectBtn.addEventListener('click', async function() {
      this.disabled = true;
      this.textContent = 'Analyzing...';
      
      const file = fileInput.files[0];
      if (!file) {
        showNotification('No file selected.', 'error');
        this.disabled = false;
        this.textContent = 'Detect Now';
        return;
      }
      
      // Determine endpoint based on file type
      const audioTypes = ['audio/mp3', 'audio/wav'];
      const imageTypes = ['image/jpeg', 'image/png'];
      let endpoint = '';
      if (audioTypes.includes(file.type)) {
        endpoint = `${BACKEND_BASE_URL}/audio/detect`;
      } else if (imageTypes.includes(file.type)) {
        endpoint = `${BACKEND_BASE_URL}/image/detect`;
      } else {
        showNotification('Unsupported file type for detection.', 'error');
        this.disabled = false;
        this.textContent = 'Detect Now';
        return;
      }
      
      // Prepare form data
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        console.log('Sending request to:', endpoint);
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', [...response.headers.entries()]);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('Error response data:', errorData);
          throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Backend response:', data);
        
        // Update UI
        fileInfo.classList.add('hidden');
        analysisResults.classList.remove('hidden');
        
        // Process results
        const isFake = data.prediction.toLowerCase() === 'fake';
        // Parse confidence percentage string (e.g., "95.23%") to number for progress bar
        const confidenceValue = parseFloat(data.confidence) || 0; // Fallback to 0 if invalid
        const resultClass = isFake ? 'fake' : 'real';
        const resultText = isFake ? 'Deepfake Detected' : 'Authentic Content';
        
        // Generate anomalies list (based on prediction and endpoint)
        const anomalies = isFake ? [
          endpoint.includes('audio') ? "Inconsistent audio features detected" : "Visual manipulation detected",
          endpoint.includes('audio') ? "Voice pattern irregularities found" : "Unnatural pixel patterns found",
          endpoint.includes('audio') ? "Synthetic voice markers identified" : "AI-generated content markers identified"
        ] : [];
        
        let anomaliesHtml = '';
        if (anomalies.length) {
          anomaliesHtml = `
            <div class="anomalies">
              <div class="font-medium mb-2">Detected Anomalies:</div>
              <ul class="anomalies-list">
                ${anomalies.map(anomaly => `<li>${anomaly}</li>`).join('')}
              </ul>
            </div>
          `;
        }
        
        resultsContainer.innerHTML = `
          <div class="results-box ${resultClass === 'fake' ? 'bg-error-light' : 'bg-success-light'}">
            <div class="result ${resultClass}">
              <span class="result-icon">${resultClass === 'fake' ? '✕' : '✓'}</span>
              <span class="result-text">${resultText}</span>
            </div>
            <div class="file-info-result">
              File: ${data.saved_file_path}
            </div>
            
            <div class="confidence-score">
              <div class="score-label">
                <span>Confidence Score:</span>
                <span class="score-value">${data.confidence}</span>
              </div>
              <div class="progress-bar">
                <div class="progress ${resultClass}" style="width: ${confidenceValue}%"></div>
              </div>
            </div>
            
            ${anomaliesHtml}
          </div>
        `;
        
        this.disabled = false;
        this.textContent = 'Detect Now';
      } catch (error) {
        console.error('Fetch error details:', {
          message: error.message,
          name: error.name,
          stack: error.stack,
          endpoint: endpoint
        });
        showNotification(`Error: Unable to connect to the server. Please ensure the backend is running and try again. (${error.message})`, 'error');
        this.disabled = false;
        this.textContent = 'Detect Now';
      }
    });
  }
  
  // Try another file button
  const tryAnotherBtn = document.querySelector('.try-another-btn');
  if (tryAnotherBtn) {
    tryAnotherBtn.addEventListener('click', resetUploadArea);
  }
  
  // Remove file button
  const removeFileBtn = document.querySelector('.remove-file-btn');
  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', resetUploadArea);
  }
  
  function resetUploadArea() {
    if (!uploadArea || !fileInfo || !analysisResults || !fileInput) return;
    
    uploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analysisResults.classList.add('hidden');
    fileInput.value = '';
    console.log('Upload area reset');
  }
  
  // Contact form submission
  const contactForm = document.getElementById('contactForm');
  if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Get form data
      const name = document.getElementById('name').value;
      const email = document.getElementById('email').value;
      const organization = document.getElementById('organization').value;
      const message = document.getElementById('message').value;
      
      // Normally you would send this data to a server here
      // For demo purposes, we'll just show a success message
      showNotification(`Thanks ${name}! Your message has been sent.`, 'success');
      
      // Reset form
      contactForm.reset();
    });
  }
  
  // Helper: Format file size
  function formatFileSize(bytes) {
    if (bytes < 1024) {
      return bytes + ' bytes';
    } else if (bytes < 1024 * 1024) {
      return (bytes / 1024).toFixed(2) + ' KB';
    } else {
      return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }
  }
  
  // Helper: Show notification
  function showNotification(message, type = 'info') {
    // Check if notification container exists, if not create it
    let notificationContainer = document.querySelector('.notification-container');
    
    if (!notificationContainer) {
      notificationContainer = document.createElement('div');
      notificationContainer.className = 'notification-container';
      document.body.appendChild(notificationContainer);
      
      // Add styles for the notification container
      notificationContainer.style.position = 'fixed';
      notificationContainer.style.top = '20px';
      notificationContainer.style.right = '20px';
      notificationContainer.style.zIndex = '1000';
      notificationContainer.style.display = 'flex';
      notificationContainer.style.flexDirection = 'column';
      notificationContainer.style.gap = '10px';
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Style notification based on type
    notification.style.padding = '1rem';
    notification.style.borderRadius = '0.5rem';
    notification.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.1)';
    notification.style.marginBottom = '0.5rem';
    notification.style.opacity = '0';
    notification.style.transform = 'translateY(-10px)';
    notification.style.transition = 'opacity 0.3s, transform 0.3s';
    
    if (type === 'success') {
      notification.style.backgroundColor = '#10B981';
      notification.style.color = 'white';
    } else if (type === 'error') {
      notification.style.backgroundColor = '#EF4444';
      notification.style.color = 'white';
    } else {
      notification.style.backgroundColor = '#3B82F6';
      notification.style.color = 'white';
    }
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.marginLeft = '0.5rem';
    closeBtn.style.background = 'transparent';
    closeBtn.style.border = 'none';
    closeBtn.style.color = 'inherit';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '1.2rem';
    closeBtn.style.float = 'right';
    
    notification.appendChild(closeBtn);
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
      notification.style.opacity = '1';
      notification.style.transform = 'translateY(0)';
    }, 10);
    
    // Auto-remove after delay
    setTimeout(() => {
      notification.style.opacity = '0';
      notification.style.transform = 'translateY(-10px)';
      
      // Remove from DOM after animation completes
      setTimeout(() => {
        notification.remove();
      }, 300);
    }, 5000);
    
    // Close button functionality
    closeBtn.addEventListener('click', () => {
      notification.style.opacity = '0';
      notification.style.transform = 'translateY(-10px)';
      
      setTimeout(() => {
        notification.remove();
      }, 300);
    });
  }
});
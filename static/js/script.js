document.addEventListener('DOMContentLoaded', function() {
    // Set default date to today in dashboard form
    const dateInput = document.querySelector('input[name="date"]');
    if (dateInput) {
        const today = new Date().toISOString().split('T')[0];
        dateInput.value = today;
        dateInput.max = today;
    }

    // Login/Register form validation
    const authForms = document.querySelectorAll('.auth-form');
    authForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const email = form.querySelector('input[name="email"]');
            const password = form.querySelector('input[name="password"]');
            
            if (!email.value || !password.value) {
                e.preventDefault();
                alert('Please fill in all fields');
            }
        });
    });

    // Stock analysis form validation
    const analysisForm = document.getElementById('analysis-form');
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(e) {
            const symbol = analysisForm.querySelector('input[name="symbol"]');
            const date = analysisForm.querySelector('input[name="date"]');
            
            if (!symbol.value.match(/^[A-Za-z]{1,5}$/)) {
                e.preventDefault();
                alert('Please enter a valid stock symbol (1-5 letters)');
                return;
            }
            
            if (!date.value) {
                e.preventDefault();
                alert('Please select a date');
                return;
            }
        });
    }

    // Handle analysis progress updates
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) {
        const symbol = progressContainer.dataset.symbol;
        const date = progressContainer.dataset.date;
        const statusElement = document.getElementById('status-message');
        const progressBar = document.getElementById('progress-bar');
        
        let step = 0;
        let progress = 0;
        let isProcessing = false; // Flag to prevent multiple submissions

        function updateProgress() {
            // Prevent if already processing
            if (isProcessing) return;

            isProcessing = true; // Set the flag to indicate processing

            fetch('/process-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    date: date,
                    step: step
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    window.location.href = data.redirect;
                    return;
                }

                statusElement.textContent = data.status;
                progressBar.style.width = data.progress + '%';
                progressBar.textContent = data.progress + '%';
                
                if (data.completed) {
                    window.location.href = data.redirect;
                } else {
                    step = data.step;
                    setTimeout(updateProgress, 1500); // Check progress after 1.5 seconds
                }

                isProcessing = false; // Reset flag after processing completes
            })
            .catch(error => {
                console.error('Error:', error);
                statusElement.textContent = 'Analysis failed';
                progressBar.style.width = '100%';
                progressBar.classList.remove('bg-primary');
                progressBar.classList.add('bg-danger');
                setTimeout(() => window.location.href = '/dashboard', 3000);
                isProcessing = false; // Reset flag on error
            });
        }

        // Start progress updates after 1 second
        setTimeout(updateProgress, 1000);
    }
});

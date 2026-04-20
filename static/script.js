document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultSection = document.getElementById('result-section');
    const predictedTimeSpan = document.getElementById('predicted-time');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Loading State
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        
        // Hide previous result if any
        if (!resultSection.classList.contains('hidden')) {
            resultSection.style.opacity = '0';
            setTimeout(() => {
                resultSection.classList.add('hidden');
                resultSection.style.display = 'none';
            }, 300);
        }

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Cast numeric fields appropriately (API also handles this, but good practice)
        data.Distance_km = parseFloat(data.Distance_km);
        data.Preparation_Time_min = parseFloat(data.Preparation_Time_min);
        data.Courier_Experience_yrs = parseFloat(data.Courier_Experience_yrs);

        try {
            // Send to Flask Backend
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                alert("Error from server: " + result.error);
                return;
            }

            // Animate number counting up for cool effect
            animateValue(predictedTimeSpan, 0, result.predicted_time_min, 1500);

            // Show result section smoothly
            setTimeout(() => {
                resultSection.style.display = 'block';
                // Small delay to allow display:block to apply before animating opacity/transform
                setTimeout(() => {
                    resultSection.classList.remove('hidden');
                    resultSection.style.opacity = '1';
                }, 10);
            }, 300);

        } catch (error) {
            console.error('Error during prediction:', error);
            alert("Failed to connect to the prediction server. Ensure the Flask backend is running.");
        } finally {
            // Reset Loading State
            setTimeout(() => {
                submitBtn.classList.remove('loading');
                submitBtn.disabled = false;
            }, 500);
        }
    });

    // Helper function to animate numbers counting up
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // easeOutQuart
            const easeOut = 1 - Math.pow(1 - progress, 4);
            obj.innerHTML = Math.floor(easeOut * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end; // Ensure exact final value
            }
        };
        window.requestAnimationFrame(step);
    }
});

document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const loader = document.getElementById('loader');
    const statusText = document.getElementById('status-text');
    const metricsPanel = document.getElementById('metrics-panel');
    const verdict = document.getElementById('verdict-banner');
    const probBar = document.getElementById('prob-bar');
    const probVal = document.getElementById('prob-val');

    // UI Reset
    loader.classList.remove('hidden');
    metricsPanel.classList.add('hidden');
    statusText.innerText = "Analyzing transaction...";

    const model = document.getElementById('model-select').value;
    const time = parseFloat(document.getElementById('time').value);
    const amount = parseFloat(document.getElementById('amount').value);
    const v1 = parseFloat(document.getElementById('v1').value);

    // Mock features for demo - in production, we'd need all 31
    const features = Array(31).fill(0);
    features[0] = time;
    features[1] = v1;
    features[30] = amount;

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, features })
        });

        const data = await response.json();
        
        loader.classList.add('hidden');
        statusText.innerText = "Complete";
        metricsPanel.classList.remove('hidden');

        const percentage = (data.probability * 100).toFixed(1);
        probVal.innerText = `${percentage}%`;
        probBar.style.width = `${percentage}%`;

        if (data.is_fraud) {
            verdict.innerText = "FRAUD DETECTED";
            verdict.className = "verdict danger";
            probBar.style.backgroundColor = "#ef4444";
        } else {
            verdict.innerText = "LEGITIMATE";
            verdict.className = "verdict safe";
            probBar.style.backgroundColor = "#6366f1";
        }

    } catch (err) {
        loader.classList.add('hidden');
        statusText.innerText = "Error: API Offline";
        console.error(err);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    
    // UI Elements
    const emptyState = document.getElementById('results-empty');
    const loadingState = document.getElementById('results-loading');
    const contentState = document.getElementById('results-content');
    const insightsList = document.getElementById('insights-list');
    
    // Result Elements
    const probValue = document.getElementById('prob-value');
    const gaugeFill = document.getElementById('gauge-fill');
    
    const searchInput = document.getElementById('applicant-search');
    
    // View Management
    const navItems = document.querySelectorAll('.nav-item');
    const newAssessmentView = document.getElementById('new-assessment-view');
    const portfolioView = document.getElementById('portfolio-view');
    
    // Policy Threshold
    const thresholdSlider = document.getElementById('policy-threshold');
    const thresholdValSpan = document.getElementById('threshold-val');
    let currentThreshold = 0.40; // Set to 40% (Optimal balance of risk/benefit found in calibration)
    let currentApplicantData = null;

    thresholdSlider.addEventListener('input', (e) => {
        currentThreshold = e.target.value / 100;
        thresholdValSpan.textContent = e.target.value + '%';
        // If results are visible, update decision immediately
        if (!contentState.classList.contains('hidden')) {
            updateDecisionDisplay(lastProbability);
        }
    });

    let lastProbability = 0;

    // Reset Button
    const resetBtn = document.getElementById('reset-form-btn');
    resetBtn.addEventListener('click', () => {
        form.reset();
        searchInput.value = '';
        emptyState.classList.remove('hidden');
        contentState.classList.add('hidden');
        updateLiveMetrics();
        showToast('Form reset', 'info');
    });

    // Button state
    const submitBtn = document.getElementById('submit-btn');
    const submitBtnSpan = submitBtn.querySelector('span');
    const submitBtnIcon = submitBtn.querySelector('i');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Form Button Loading State
        submitBtn.disabled = true;
        submitBtnSpan.textContent = "Processing...";
        submitBtnIcon.className = "fa-solid fa-circle-notch fa-spin";
        
        // Change Results Panel State
        emptyState.classList.add('hidden');
        contentState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        // Scroll to results panel on mobile
        if(window.innerWidth <= 1100) {
            document.getElementById('results-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
        }

        // Collect form data
        const formData = new FormData(form);
        const requestData = {
            AMT_INCOME_TOTAL: parseFloat(formData.get('AMT_INCOME_TOTAL')),
            AMT_CREDIT: parseFloat(formData.get('AMT_CREDIT')),
            AMT_ANNUITY: parseFloat(formData.get('AMT_ANNUITY')),
            AMT_GOODS_PRICE: parseFloat(formData.get('AMT_GOODS_PRICE')),
            AGE_YEARS: parseFloat(formData.get('AGE_YEARS')),
            YEARS_EMPLOYED: parseFloat(formData.get('YEARS_EMPLOYED')),
            CODE_GENDER: parseInt(formData.get('CODE_GENDER')),
            FLAG_OWN_CAR: formData.get('FLAG_OWN_CAR') ? 1 : 0,
            FLAG_OWN_REALTY: formData.get('FLAG_OWN_REALTY') ? 1 : 0,
            NAME_EDUCATION_TYPE: parseInt(formData.get('NAME_EDUCATION_TYPE')),
            CNT_CHILDREN: parseInt(formData.get('CNT_CHILDREN')),
            EXT_SOURCE_1: parseFloat(formData.get('EXT_SOURCE_1')),
            EXT_SOURCE_2: parseFloat(formData.get('EXT_SOURCE_2')),
            EXT_SOURCE_3: parseFloat(formData.get('EXT_SOURCE_3')),
            CNT_FAM_MEMBERS: currentApplicantData ? currentApplicantData.CNT_FAM_MEMBERS : (parseInt(formData.get('CNT_CHILDREN')) + 2),
            DAYS_LAST_PHONE_CHANGE: currentApplicantData ? currentApplicantData.DAYS_LAST_PHONE_CHANGE : 0,
            app_id: searchInput.value || "Manual"
        };

        try {
            // Send request to API
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error('Prediction request failed');
            }

            const result = await response.json();
            
            // Artificial delay for UX
            setTimeout(() => {
                updateUIWithResults(result, requestData);
                showToast('Assessment completed successfully', 'success');
                
                // Reset Button
                submitBtn.disabled = false;
                submitBtnSpan.textContent = "Run AI Assessment";
                submitBtnIcon.className = "fa-solid fa-microchip";
            }, 1200);

        } catch (error) {
            console.error('Error:', error);
            showToast('Failed to run assessment', 'error');
            resetButtonState();
        }
    });

    function resetButtonState() {
        submitBtn.disabled = false;
        submitBtnSpan.textContent = "Run AI Assessment";
        submitBtnIcon.className = "fa-solid fa-microchip";
        loadingState.classList.add('hidden');
    }



    function updateDecisionDisplay(probability) {
        const decisionBox = document.querySelector('.decision-box');
        const decisionTitle = document.querySelector('.decision-title');
        const decisionSubtitle = document.querySelector('.decision-subtitle');
        const decisionIcon = document.querySelector('.decision-box i');
        const probValue = document.querySelector('.prob-value');
        
        const isDefault = probability > currentThreshold;

        if (isDefault) {
            decisionBox.classList.add('danger-mode');
            document.querySelector('.gauge-fill').style.stroke = 'var(--danger)';
            probValue.style.color = 'var(--danger)';
            decisionTitle.textContent = "DECLINED";
            decisionSubtitle.textContent = "Risk Exceeds Policy Threshold";
            decisionIcon.className = "fa-solid fa-triangle-exclamation";
        } else {
            decisionBox.classList.remove('danger-mode');
            document.querySelector('.gauge-fill').style.stroke = 'var(--success)';
            probValue.style.color = 'var(--success)';
            decisionTitle.textContent = "APPROVED";
            decisionSubtitle.textContent = "Low Risk Profile";
            decisionIcon.className = "fa-solid fa-shield-check";
        }
    }

    function updateUIWithResults(data, inputs) {
        lastProbability = data.probability;
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        contentState.classList.remove('hidden');

        const probPercent = (data.probability * 100).toFixed(1);

        // Reset gauge and classes
        gaugeFill.style.strokeDashoffset = '125.6'; // reset to 0
        
        // Animate numbers up
        animateValue(probValue, 0, parseFloat(probPercent), 1500);

        // Wait a tick before animating SVG
        setTimeout(() => {
            const offset = 125.6 - (125.6 * (Math.min(probPercent, 100) / 100));
            gaugeFill.style.strokeDashoffset = offset;
            updateDecisionDisplay(data.probability);
        }, 100);

        // Generate Insights based on input and prediction
        generateInsights(data, inputs);
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // easeOutQuart
            const ease = 1 - Math.pow(1 - progress, 4);
            const current = (start + ease * (end - start)).toFixed(1);
            obj.innerHTML = current + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    function generateInsights(data, inputs) {
        insightsList.innerHTML = ''; // clear old insights

        if (!data.explanations || data.explanations.length === 0) {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fa-solid fa-info-circle"></i> <span>No explainability data available.</span>`;
            insightsList.appendChild(li);
            return;
        }

        data.explanations.forEach(exp => {
            const li = document.createElement('li');
            li.className = 'insight-item';
            
            const featureName = exp.feature.replace(/_/g, ' ').toLowerCase();
            const formattedName = featureName.charAt(0).toUpperCase() + featureName.slice(1);
            
            const colorClass = exp.effect === "increases_risk" ? "text-red" : "text-green";
            const barClass = exp.effect === "increases_risk" ? "bar-red" : "bar-green";
            
            // Normalize bar width (SHAP values are typically small, e.g. 0.1 to 2.0)
            const barWidth = Math.min(exp.absolute_impact * 200, 100); 

            li.innerHTML = `
                <div class="insight-row">
                    <span class="insight-name">${formattedName}</span>
                    <span class="insight-value ${colorClass}">${exp.shap_value > 0 ? '+' : ''}${exp.shap_value.toFixed(3)}</span>
                </div>
                <div class="shap-bar-bg">
                    <div class="shap-bar ${barClass}" style="width: ${barWidth}%"></div>
                </div>
            `;
            insightsList.appendChild(li);
        });
    }

    // BULK UPLOAD LOGIC
    const bulkUploadBtn = document.getElementById('bulk-upload-btn');
    const csvUploadInput = document.getElementById('csv-upload-input');

    bulkUploadBtn.addEventListener('click', () => {
        csvUploadInput.click();
    });

    csvUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const uploadData = new FormData();
        uploadData.append('file', file);

        bulkUploadBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading...';
        bulkUploadBtn.disabled = true;

        try {
            const response = await fetch('/upload_csv', {
                method: 'POST',
                body: uploadData
            });
            const result = await response.json();
            
            if (response.ok) {
                showToast(`Loaded ${result.count} applicants successfully`, 'success');
                csvUploadInput.value = ''; // Reset
            } else {
                showToast(`Error: ${result.detail}`, 'error');
            }
        } catch (err) {
            console.error(err);
            showToast('Failed to process CSV file', 'error');
        } finally {
            bulkUploadBtn.innerHTML = '<i class="fa-solid fa-upload"></i> Bulk Upload CSV';
            bulkUploadBtn.disabled = false;
        }
    });

    // SEARCH LOGIC
    const searchContainer = document.getElementById('search-container');
    const searchIcon = searchContainer.querySelector('i');

    async function performSearch() {
        const appId = searchInput.value.trim();
        if (!appId) return;

        searchContainer.style.opacity = '0.5';

        try {
            const response = await fetch(`/applicant/${appId}`);
            if (!response.ok) {
                throw new Error(await response.text());
            }
            const data = await response.json();
            currentApplicantData = data;

            // Auto-fill form
            // Auto-fill form (Null-safe)
            form.elements['AMT_INCOME_TOTAL'].value = (data.AMT_INCOME_TOTAL || 0).toFixed(2);
            form.elements['AMT_CREDIT'].value = (data.AMT_CREDIT || 0).toFixed(2);
            form.elements['AMT_ANNUITY'].value = (data.AMT_ANNUITY || 0).toFixed(2);
            form.elements['AMT_GOODS_PRICE'].value = data.AMT_GOODS_PRICE ? data.AMT_GOODS_PRICE.toFixed(2) : (data.AMT_CREDIT || 0).toFixed(2);
            form.elements['AGE_YEARS'].value = data.AGE_YEARS ? Math.round(data.AGE_YEARS) : 30;
            form.elements['YEARS_EMPLOYED'].value = data.YEARS_EMPLOYED ? data.YEARS_EMPLOYED.toFixed(1) : 0;
            form.elements['CODE_GENDER'].value = data.CODE_GENDER !== undefined ? data.CODE_GENDER : "";
            form.elements['CNT_CHILDREN'].value = data.CNT_CHILDREN || 0;
            form.elements['NAME_EDUCATION_TYPE'].value = data.NAME_EDUCATION_TYPE !== undefined ? data.NAME_EDUCATION_TYPE : "";
            
            form.elements['FLAG_OWN_CAR'].checked = data.FLAG_OWN_CAR === 1;
            form.elements['FLAG_OWN_REALTY'].checked = data.FLAG_OWN_REALTY === 1;

            // Handle External Scores (map 0-1 to 0-100)
            const ext1 = data.EXT_SOURCE_1 !== null ? Math.round(data.EXT_SOURCE_1 * 100) : 50;
            const ext2 = data.EXT_SOURCE_2 !== null ? Math.round(data.EXT_SOURCE_2 * 100) : 50;
            const ext3 = data.EXT_SOURCE_3 !== null ? Math.round(data.EXT_SOURCE_3 * 100) : 50;

            form.elements['EXT_SOURCE_1'].value = ext1;
            document.getElementById('val-ext1').textContent = ext1;
            
            form.elements['EXT_SOURCE_2'].value = ext2;
            document.getElementById('val-ext2').textContent = ext2;
            
            form.elements['EXT_SOURCE_3'].value = ext3;
            document.getElementById('val-ext3').textContent = ext3;

            // Provide feedback and automatically trigger assessment
            searchInput.value = '';
            searchInput.placeholder = `Loaded ID: ${appId}`;
            
            // Switch back to assessment view if we are on portfolio
            navItems[0].click();
            
            // Trigger live metrics update
            updateLiveMetrics();
            
            showToast(`Applicant ${appId} loaded`, 'success');
            
            // Submit form programmatically
            submitBtn.click();

        } catch (err) {
            console.error(err);
            showToast(`Could not find Applicant ID ${appId}`, 'error');
        } finally {
            searchContainer.style.opacity = '1';
        }
    }

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    searchIcon.addEventListener('click', performSearch);

    // Slider value listeners
    const sliders = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'];
    const valueSpans = ['val-ext1', 'val-ext2', 'val-ext3'];

    sliders.forEach((sId, index) => {
        const slider = form.elements[sId];
        const span = document.getElementById(valueSpans[index]);
        slider.addEventListener('input', (e) => {
            span.textContent = e.target.value;
        });
    });

    // NAVIGATION LOGIC
    const historyTableBody = document.getElementById('history-table-body');
    const refreshHistoryBtn = document.getElementById('refresh-history');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // UI Toggle
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            if (item.textContent.includes('New Assessment')) {
                newAssessmentView.classList.remove('hidden');
                portfolioView.classList.add('hidden');
            } else if (item.textContent.includes('Portfolio')) {
                newAssessmentView.classList.add('hidden');
                portfolioView.classList.remove('hidden');
                loadHistory();
            }
        });
    });

    async function loadHistory() {
        historyTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 2rem;">Loading history...</td></tr>';
        
        try {
            const response = await fetch('/history');
            const data = await response.json();
            
            historyTableBody.innerHTML = '';
            
            if (data.length === 0) {
                historyTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 2rem;">No history found. Run an assessment first!</td></tr>';
                return;
            }

            data.forEach(item => {
                const tr = document.createElement('tr');
                const riskPercent = (item.probability * 100).toFixed(1);
                const decisionClass = item.is_default ? 'pill-danger' : 'pill-success';
                const decisionText = item.is_default ? 'Declined' : 'Approved';

                tr.innerHTML = `
                    <td>${item.timestamp}</td>
                    <td><code style="color: var(--accent-blue)">${item.app_id}</code></td>
                    <td>$${item.income.toLocaleString()}</td>
                    <td>$${item.credit.toLocaleString()}</td>
                    <td style="font-weight: 700;">${riskPercent}%</td>
                    <td><span class="decision-pill ${decisionClass}">${decisionText}</span></td>
                    <td><button class="view-btn" onclick="loadApplicantFromHistory('${item.app_id}')">Re-Load</button></td>
                `;
                historyTableBody.appendChild(tr);
            });

            updatePortfolioStats(data);

        } catch (err) {
            console.error(err);
            historyTableBody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 2rem; color: var(--danger);">Failed to load history.</td></tr>';
        }
    }

    function updatePortfolioStats(data) {
        if (!data || data.length === 0) return;
        
        const total = data.length;
        const approvedCount = data.filter(i => !i.is_default).length;
        const avgRisk = data.reduce((acc, i) => acc + i.probability, 0) / total;

        document.getElementById('stat-total').textContent = total;
        document.getElementById('stat-approval').textContent = ((approvedCount/total)*100).toFixed(1) + '%';
        document.getElementById('stat-risk').textContent = (avgRisk * 100).toFixed(1) + '%';
    }

    // LIVE METRICS LOGIC
    function updateLiveMetrics() {
        const income = parseFloat(form.elements['AMT_INCOME_TOTAL'].value) || 0;
        const credit = parseFloat(form.elements['AMT_CREDIT'].value) || 0;
        const annuity = parseFloat(form.elements['AMT_ANNUITY'].value) || 0;
        const goods = parseFloat(form.elements['AMT_GOODS_PRICE'].value) || 0;

        const dti = income > 0 ? (annuity / (income/12) * 100).toFixed(1) : '0.0';
        const ltv = goods > 0 ? (credit / goods * 100).toFixed(1) : '0.0';

        document.getElementById('live-dti').textContent = dti + '%';
        document.getElementById('live-ltv').textContent = ltv + '%';
        document.getElementById('live-annuity').textContent = '$' + annuity.toLocaleString();

        // Color DTI based on risk
        const dtiEl = document.getElementById('live-dti');
        if (parseFloat(dti) > 40) dtiEl.style.color = 'var(--danger)';
        else if (parseFloat(dti) > 30) dtiEl.style.color = 'var(--warning)';
        else dtiEl.style.color = 'var(--success)';
    }

    // Attach to all relevant inputs
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'].forEach(name => {
        form.elements[name].addEventListener('input', updateLiveMetrics);
    });

    refreshHistoryBtn.addEventListener('click', loadHistory);

    // Global function to re-load from history
    window.loadApplicantFromHistory = (appId) => {
        if (appId === "Manual") {
            alert("Manual assessments cannot be re-loaded by ID. Please use a specific Application Number.");
            return;
        }
        searchInput.value = appId;
        // Switch back to assessment view
        navItems[0].click();
        // Trigger search
        searchInput.dispatchEvent(new KeyboardEvent('keypress', {'key': 'Enter'}));
    };

    // TOAST NOTIFICATIONS
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let icon = 'fa-info-circle';
        if (type === 'success') icon = 'fa-circle-check';
        if (type === 'error') icon = 'fa-circle-exclamation';

        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);

        // Auto-remove
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

});

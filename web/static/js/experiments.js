/**
 * =====================================================
 * Experiments Dashboard JavaScript
 * =====================================================
 * จัดการ functionality ของหน้า Experiment Dashboard
 * - Fetch และแสดงข้อมูล Accuracy
 * - สร้าง Performance Chart
 * - แสดง Experiment Log
 * - คำนวณ Latency Stats
 * =====================================================
 */

// ========================
// Constants & Configuration
// ========================
const API_ENDPOINTS = {
    accuracy: '/api/experiments/accuracy',
    log: '/api/experiments/log'
};

const REFRESH_INTERVAL = 60000; // 60 seconds

// Chart instance
let performanceChart = null;

// ========================
// API Functions
// ========================

/**
 * ดึงข้อมูล Accuracy (MAE/RMSE) จาก API
 */
async function fetchAccuracyData() {
    try {
        const response = await fetch(API_ENDPOINTS.accuracy);
        const result = await response.json();

        if (result.success) {
            renderAccuracyTable(result.data);
            renderPerformanceChart(result.data);
        }
    } catch (error) {
        console.error('Error fetching accuracy data:', error);
        showError('accuracy-tbody', 'ไม่สามารถโหลดข้อมูล Accuracy ได้');
    }
}

/**
 * ดึงข้อมูล Experiment Log จาก API
 */
async function fetchExperimentLog() {
    try {
        const response = await fetch(API_ENDPOINTS.log);
        const result = await response.json();

        if (result.success) {
            renderExperimentLog(result.data.experiments);
            updateStats(result.data.stats);
            updateLatencyStats(result.data.latency);
        }
    } catch (error) {
        console.error('Error fetching experiment log:', error);
        showError('experiment-list', 'ไม่สามารถโหลดข้อมูล Experiment Log ได้');
    }
}

// ========================
// Render Functions
// ========================

/**
 * แสดงตาราง Accuracy
 * @param {Array} data - ข้อมูล accuracy ของ models
 */
function renderAccuracyTable(data) {
    const tbody = document.getElementById('accuracy-tbody');

    if (!data || data.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="loading">ไม่พบข้อมูล</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = data.map(row => {
        const maeClass = getMetricClass(row.mae, 'mae', row.coin);
        const rmseClass = getMetricClass(row.rmse, 'rmse', row.coin);

        return `
            <tr>
                <td>
                    <span class="coin-badge ${row.coin.toLowerCase()}">${row.coin}</span>
                </td>
                <td>
                    <span class="timeframe-badge">${row.timeframe}</span>
                </td>
                <td>
                    <span class="model-badge ${row.model.toLowerCase()}">${row.model}</span>
                </td>
                <td>
                    <span class="metric-value ${maeClass}">${row.mae.toFixed(2)}</span>
                </td>
                <td>
                    <span class="metric-value ${rmseClass}">${row.rmse.toFixed(2)}</span>
                </td>
            </tr>
        `;
    }).join('');
}

/**
 * กำหนด CSS class ตามค่า metric
 * @param {number} value - ค่า metric
 * @param {string} type - ประเภท (mae/rmse)
 * @param {string} coin - เหรียญ (BTC/ETH)
 * @returns {string} CSS class
 */
function getMetricClass(value, type, coin) {
    // Different thresholds for BTC vs ETH
    if (coin.toUpperCase() === 'BTC') {
        if (type === 'mae') {
            if (value < 300) return 'good';
            if (value < 500) return 'warn';
            return 'bad';
        } else {
            if (value < 500) return 'good';
            if (value < 800) return 'warn';
            return 'bad';
        }
    } else {
        if (type === 'mae') {
            if (value < 15) return 'good';
            if (value < 25) return 'warn';
            return 'bad';
        } else {
            if (value < 25) return 'good';
            if (value < 40) return 'warn';
            return 'bad';
        }
    }
}

/**
 * สร้าง Performance Chart
 * @param {Array} data - ข้อมูลสำหรับสร้าง chart
 */
function renderPerformanceChart(data) {
    const ctx = document.getElementById('performance-chart');
    if (!ctx) return;

    const context = ctx.getContext('2d');

    // Group data for chart
    const labels = data.map(d => `${d.coin}/${d.timeframe}/${d.model}`);
    const maeValues = data.map(d => d.mae);
    const rmseValues = data.map(d => d.rmse);

    // Destroy existing chart if exists
    if (performanceChart) {
        performanceChart.destroy();
    }

    performanceChart = new Chart(context, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'MAE',
                    data: maeValues,
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 1
                },
                {
                    label: 'RMSE',
                    data: rmseValues,
                    backgroundColor: 'rgba(168, 85, 247, 0.6)',
                    borderColor: 'rgba(168, 85, 247, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        font: {
                            family: "'Inter', sans-serif"
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        font: { size: 10 }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.5)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                }
            }
        }
    });
}

/**
 * แสดง Experiment Log
 * @param {Array} experiments - รายการ experiment
 */
function renderExperimentLog(experiments) {
    const container = document.getElementById('experiment-list');

    if (!experiments || experiments.length === 0) {
        container.innerHTML = `
            <div class="loading">
                <span>ไม่พบข้อมูล Experiment</span>
            </div>
        `;
        return;
    }

    container.innerHTML = experiments.map(exp => `
        <div class="experiment-item">
            <div class="experiment-info">
                <div class="experiment-icon ${exp.status}">
                    <i data-lucide="${exp.status === 'success' ? 'check-circle' : 'x-circle'}"></i>
                </div>
                <div class="experiment-details">
                    <h4>${exp.model_type.toUpperCase()} - ${exp.coin.toUpperCase()}/${exp.timeframe}</h4>
                    <span>${formatTimestamp(exp.timestamp)} • ID: ${exp.experiment_id}</span>
                </div>
            </div>
            <div class="experiment-metrics">
                <div class="metric-item">
                    <label>Accuracy</label>
                    <span style="color: var(--accent-green);">${exp.accuracy.toFixed(2)}%</span>
                </div>
                <div class="metric-item">
                    <label>MAE</label>
                    <span style="color: var(--accent-cyan);">${exp.mae.toFixed(2)}</span>
                </div>
                <div class="metric-item">
                    <label>RMSE</label>
                    <span style="color: var(--accent-purple);">${exp.rmse.toFixed(2)}</span>
                </div>
                <div class="metric-item">
                    <label>Latency</label>
                    <span style="color: var(--accent-orange);">${exp.latency_ms.toFixed(0)}ms</span>
                </div>
            </div>
        </div>
    `).join('');

    // Reinitialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}

// ========================
// Update Functions
// ========================

/**
 * อัพเดท Stats Summary
 * @param {Object} stats - ข้อมูลสถิติ
 */
function updateStats(stats) {
    updateElementText('stat-experiments', stats.total_experiments || 0);
    updateElementText('stat-avg-accuracy',
        stats.avg_accuracy ? `${stats.avg_accuracy.toFixed(2)}%` : '--'
    );
    updateElementText('stat-best-model', stats.best_model || '--');
}

/**
 * อัพเดท Latency Stats
 * @param {Object} latency - ข้อมูล latency
 */
function updateLatencyStats(latency) {
    updateElementText('latency-min',
        latency.min ? latency.min.toFixed(0) : '--'
    );
    updateElementText('latency-avg',
        latency.avg ? latency.avg.toFixed(0) : '--'
    );
    updateElementText('latency-max',
        latency.max ? latency.max.toFixed(0) : '--'
    );
    updateElementText('stat-avg-latency',
        latency.avg ? `${(latency.avg / 1000).toFixed(1)}s` : '--'
    );
}

// ========================
// Utility Functions
// ========================

/**
 * อัพเดทข้อความใน element
 * @param {string} id - Element ID
 * @param {string} text - ข้อความ
 */
function updateElementText(id, text) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
    }
}

/**
 * แสดง error message
 * @param {string} containerId - Container ID
 * @param {string} message - ข้อความ error
 */
function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="loading" style="color: var(--accent-red);">
                <i data-lucide="alert-circle" style="width: 40px; height: 40px;"></i>
                <span>${message}</span>
            </div>
        `;
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
}

/**
 * Format timestamp ให้อ่านง่าย
 * @param {string} timestamp - ISO timestamp
 * @returns {string} Formatted timestamp
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '--';

    try {
        const date = new Date(timestamp);
        return date.toLocaleString('th-TH', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return timestamp;
    }
}

/**
 * Refresh ข้อมูลทั้งหมด
 */
async function refreshData() {
    const btn = document.getElementById('refresh-btn');
    if (btn) btn.classList.add('spinning');

    console.log('🔄 Refreshing dashboard data...');

    try {
        await Promise.all([
            fetchAccuracyData(),
            fetchExperimentLog()
        ]);
    } catch (error) {
        console.error('Refresh failed:', error);
    } finally {
        if (btn) {
            // Add a small delay to ensure the animation is visible/smooth
            setTimeout(() => {
                btn.classList.remove('spinning');
            }, 500);
        }
    }
}

// ========================
// Initialization
// ========================

/**
 * Initialize dashboard เมื่อ DOM โหลดเสร็จ
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('📊 Experiment Dashboard initialized');

    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Fetch initial data
    fetchAccuracyData();
    fetchExperimentLog();

    // Setup auto refresh
    setInterval(refreshData, REFRESH_INTERVAL);
});

// Export functions for global access
window.refreshData = refreshData;

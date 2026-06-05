// Initial Data Presets from User Design
const initialPhysicalThicknesses = [
    123.756651, 217.102141, 122.323165, 214.986608, 121.736214, 214.513363, 
    121.474314, 214.857409, 121.558202, 215.174188, 121.848105, 214.361490, 
    121.467751, 214.157004, 121.509439, 214.345442, 121.107103, 213.806544, 
    120.635733, 213.625958, 120.579659, 213.453391, 120.052243, 212.675989, 
    119.865308, 212.276541, 119.639313, 212.153051, 119.221428, 211.582432, 
    119.424791, 211.581925, 119.263789, 211.261823, 118.904834, 210.961159, 
    118.850955, 210.725919, 118.670611, 211.257222, 118.840754, 212.337908, 
    121.091542, 291.088350
];

const initialMaterialLayers = [
    2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 
    2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1
];

const initialMaterialParams = {
    0: { name: "air", desc: "Air", n: 1.0, k: 0.0, a: null, alpha: null, beta: null, kappa: null, C: null, Y: null, prat: null, phiM: null },
    1: { name: "SiO2", desc: "Silica - Thin film Room Temperature", n: 1.45, k: 0.0, a: 0.0, alpha: 5.1e-7, beta: 0.000008, kappa: 1.38, C: 1641200, Y: 70000000000, prat: 0.19, phiM: 0.000023 },
    2: { name: "TiGermania", desc: "Titania doped Germania - Room Temperature LMA", n: 1.866, k: 2e-7, a: 1.0, alpha: 0.000001282, beta: 0.000024, kappa: 33.0, C: 2510000, Y: 92000000000, prat: 0.29, phiM: 0.00009013672 },
    999: { name: "air", desc: "Air", n: 1.0, k: 0.0, a: null, alpha: null, beta: null, kappa: null, C: null, Y: null, prat: null, phiM: null }
};

const materialColors = {
    "SiO2": "#1f77b4",
    "TiGermania": "#e377c2",
    "Substrate": "#7f7f7f"
};

// Global App State
let layersState = [];
let materialParamsState = {};
let selectedLayerIndex = null;
let debounceTimer = null;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    resetToDefault();
    setupEventListeners();
    setupTabSystem();
});

// Reset stack and params
function resetToDefault() {
    layersState = [];
    for (let i = 0; i < initialPhysicalThicknesses.length; i++) {
        layersState.push({
            thickness: initialPhysicalThicknesses[i],
            material: initialMaterialLayers[i]
        });
    }
    materialParamsState = JSON.parse(JSON.stringify(initialMaterialParams));
    selectedLayerIndex = 0;
    
    renderStackEditor();
    drawStackPlot();
    triggerSimulation();
}

// Render Stack Table rows
function renderStackEditor() {
    const tableBody = document.getElementById("stackTableBody");
    tableBody.innerHTML = "";
    
    const laserWavelength = parseFloat(document.getElementById("inputLambda").value) || 1064.0;
    
    const materialOptions = [
        { key: 1, name: "SiO2 (Material 1)" },
        { key: 2, name: "TiGermania (Material 2)" }
    ];
    
    const selectEditMaterial = document.getElementById("editMaterial");
    selectEditMaterial.innerHTML = materialOptions
        .map(o => `<option value="${o.key}">${o.name}</option>`)
        .join("");

    layersState.forEach((layer, idx) => {
        const n = materialParamsState[layer.material].n;
        const dOpt = (layer.thickness * n) / laserWavelength;
        
        const tr = document.createElement("tr");
        tr.setAttribute("data-index", idx);
        if (idx === selectedLayerIndex) {
            tr.classList.add("selected");
        }
        
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>
                <span class="subsystem-badge ${layer.material === 1 ? 'silica' : 'germania'}">
                    ${layer.material === 1 ? 'Silica (1)' : 'Germania (2)'}
                </span>
            </td>
            <td>
                <div class="numeric-input-group">
                    <input type="number" class="table-thick-input" step="0.01" value="${layer.thickness.toFixed(4)}">
                    <span class="unit-label">nm</span>
                </div>
            </td>
            <td class="mono">${dOpt.toFixed(5)}</td>
            <td class="actions-cell">
                <button class="icon-btn btn-up" title="Move Up">▲</button>
                <button class="icon-btn btn-down" title="Move Down">▼</button>
                <button class="icon-btn btn-delete" title="Delete Layer" style="color:#ff5252">✕</button>
            </td>
        `;
        
        tr.addEventListener("click", (e) => {
            if (e.target.tagName !== "INPUT" && e.target.tagName !== "BUTTON" && !e.target.classList.contains("icon-btn")) {
                selectLayer(idx);
            }
        });
        
        tr.addEventListener("dblclick", (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "BUTTON" || e.target.classList.contains("icon-btn")) {
                return;
            }
            const currentNum = idx + 1;
            const newNumStr = prompt(`Move Layer #${currentNum} to position (1 - ${layersState.length}):`, currentNum);
            if (newNumStr === null) return;
            const newNum = parseInt(newNumStr, 10);
            if (isNaN(newNum) || newNum < 1 || newNum > layersState.length) {
                alert(`Invalid layer number. Please enter a value between 1 and ${layersState.length}.`);
                return;
            }
            moveLayerToPosition(idx, newNum);
        });
        
        const thickInput = tr.querySelector(".table-thick-input");
        thickInput.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value) || 0;
            layersState[idx].thickness = val;
            updateOpticalThicknessDisplay();
            
            if (idx === selectedLayerIndex) {
                document.getElementById("editThicknessSlider").value = Math.min(val, 500);
                document.getElementById("editThicknessText").value = val;
                document.getElementById("editOpticalThickness").textContent = ((val * n) / laserWavelength).toFixed(5);
            }
            
            drawStackPlot();
            triggerSimulationDebounced();
        });
        
        tr.querySelector(".btn-up").addEventListener("click", (e) => {
            e.stopPropagation();
            moveLayer(idx, -1);
        });
        tr.querySelector(".btn-down").addEventListener("click", (e) => {
            e.stopPropagation();
            moveLayer(idx, 1);
        });
        tr.querySelector(".btn-delete").addEventListener("click", (e) => {
            e.stopPropagation();
            deleteLayer(idx);
        });
        
        tableBody.appendChild(tr);
    });
    
    updateActiveEditorDisplay();
}

// Draw the horizontal bar chart representing coating layer stack
function drawStackPlot() {
    const traces = [];
    let depthSoFar = 0.0;
    const legendShown = {};
    
    layersState.forEach((layer, idx) => {
        const matName = materialParamsState[layer.material].name;
        const thick = layer.thickness;
        
        let showLegend = false;
        if (!legendShown[matName]) {
            showLegend = true;
            legendShown[matName] = true;
        }
        
        const isSelected = (idx === selectedLayerIndex);
        
        traces.push({
            x: [depthSoFar + thick / 2.0],
            y: [thick],
            width: [thick],
            name: `${idx + 1}: ${matName}`,
            type: 'bar',
            marker: {
                color: materialColors[matName] || '#555555',
                line: {
                    width: isSelected ? 2.0 : 0.5,
                    color: isSelected ? '#ffffff' : '#000000'
                }
            },
            showlegend: showLegend,
            legendgroup: matName,
            hovertemplate: `Layer ${idx + 1}: ${matName}<br>Thickness: ${thick.toFixed(2)} nm<extra></extra>`
        });
        
        depthSoFar += thick;
    });
    
    // Substrate
    const subWidth = 150.0;
    let showSubLegend = false;
    if (!legendShown["Substrate"]) {
        showSubLegend = true;
        legendShown["Substrate"] = true;
    }
    
    const maxThick = Math.max(...layersState.map(l => l.thickness), 100);
    
    traces.push({
        x: [depthSoFar + subWidth / 2.0],
        y: [maxThick],
        width: [subWidth],
        name: "Substrate",
        type: 'bar',
        marker: {
            color: '#555555',
            line: { width: 0.5, color: '#000000' }
        },
        showlegend: showSubLegend,
        legendgroup: "Substrate",
        hovertemplate: "Substrate<br>Thickness: 150 nm<extra></extra>"
    });
    
    const layout = {
        paper_bgcolor: '#1e1e1e',
        plot_bgcolor: '#1e1e1e',
        margin: { l: 45, r: 15, t: 15, b: 35 },
        height: 325,
        legend: {
            font: { size: 9, color: '#e0e0e0' },
            orientation: 'h',
            y: -0.2
        },
        xaxis: {
            title: { text: "Coating Depth (nm)", font: { size: 9, color: '#888' } },
            tickfont: { size: 8, color: '#888' },
            gridcolor: 'rgba(255,255,255,0.04)',
            linecolor: '#2d2d2d'
        },
        yaxis: {
            title: { text: "Thick [nm]", font: { size: 9, color: '#888' } },
            tickfont: { size: 8, color: '#888' },
            gridcolor: 'rgba(255,255,255,0.04)',
            linecolor: '#2d2d2d'
        },
        hovermode: 'closest',
        barmode: 'overlay'
    };
    
    Plotly.newPlot('plotStack', traces, layout, { responsive: true, displayModeBar: false });
    
    const plotStackDiv = document.getElementById('plotStack');
    plotStackDiv.removeAllListeners('plotly_click');
    plotStackDiv.on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            const curveIdx = data.points[0].curveNumber;
            if (curveIdx >= 0 && curveIdx < layersState.length) {
                selectLayer(curveIdx);
            }
        }
    });
}

// Update optical thickness values in table cells
function updateOpticalThicknessDisplay() {
    const laserWavelength = parseFloat(document.getElementById("inputLambda").value) || 1064.0;
    const rows = document.querySelectorAll("#stackTableBody tr");
    rows.forEach((tr) => {
        const idx = parseInt(tr.getAttribute("data-index"));
        const layer = layersState[idx];
        const n = materialParamsState[layer.material].n;
        const dOpt = (layer.thickness * n) / laserWavelength;
        tr.cells[3].textContent = dOpt.toFixed(5);
    });
}

// Selected layer editor display
function updateActiveEditorDisplay() {
    const editorCard = document.getElementById("layerEditorCard");
    const editNum = document.getElementById("editLayerNumber");
    const editMat = document.getElementById("editMaterial");
    const editThickText = document.getElementById("editThicknessText");
    const editThickSlider = document.getElementById("editThicknessSlider");
    const editOpt = document.getElementById("editOpticalThickness");
    
    if (selectedLayerIndex === null || selectedLayerIndex >= layersState.length) {
        editorCard.style.opacity = "0.5";
        editorCard.style.pointerEvents = "none";
        editNum.textContent = "-";
        editOpt.textContent = "-";
        return;
    }
    
    editorCard.style.opacity = "1";
    editorCard.style.pointerEvents = "all";
    
    const layer = layersState[selectedLayerIndex];
    const n = materialParamsState[layer.material].n;
    const laserWavelength = parseFloat(document.getElementById("inputLambda").value) || 1064.0;
    
    editNum.textContent = selectedLayerIndex + 1;
    editMat.value = layer.material;
    editThickText.value = layer.thickness.toFixed(4);
    editThickSlider.value = Math.min(layer.thickness, 500);
    editOpt.textContent = ((layer.thickness * n) / laserWavelength).toFixed(5);
}

// Select Layer
function selectLayer(idx) {
    selectedLayerIndex = idx;
    
    const tableRows = document.querySelectorAll("#stackTableBody tr");
    tableRows.forEach((row, i) => {
        if (i === idx) {
            row.classList.add("selected");
        } else {
            row.classList.remove("selected");
        }
    });
    
    // Redraw stack plot to show selection border highlight on target curve
    drawStackPlot();
    updateActiveEditorDisplay();
}

// Move Layer Up/Down
function moveLayer(idx, direction) {
    const targetIdx = idx + direction;
    if (targetIdx < 0 || targetIdx >= layersState.length) return;
    
    const temp = layersState[idx];
    layersState[idx] = layersState[targetIdx];
    layersState[targetIdx] = temp;
    
    if (selectedLayerIndex === idx) {
        selectedLayerIndex = targetIdx;
    } else if (selectedLayerIndex === targetIdx) {
        selectedLayerIndex = idx;
    }
    
    renderStackEditor();
    drawStackPlot();
    triggerSimulation();
}

// Delete Layer
function deleteLayer(idx) {
    if (layersState.length <= 1) {
        alert("A coating stack must have at least one layer!");
        return;
    }
    
    layersState.splice(idx, 1);
    
    if (selectedLayerIndex === idx) {
        selectedLayerIndex = Math.max(0, idx - 1);
    } else if (selectedLayerIndex > idx) {
        selectedLayerIndex--;
    }
    
    renderStackEditor();
    drawStackPlot();
    triggerSimulation();
}

// Move layer to arbitrary 1-based position index
function moveLayerToPosition(fromIdx, toPosition1Based) {
    const toIdx = toPosition1Based - 1;
    if (isNaN(toIdx) || toIdx < 0 || toIdx >= layersState.length || toIdx === fromIdx) {
        return;
    }
    
    // Remove from old index and insert into new index
    const [layer] = layersState.splice(fromIdx, 1);
    layersState.splice(toIdx, 0, layer);
    
    selectedLayerIndex = toIdx;
    
    renderStackEditor();
    drawStackPlot();
    triggerSimulation();
    
    // Highlight and scroll the newly moved row into view
    setTimeout(() => {
        const row = document.querySelector(`#stackTableBody tr[data-index="${toIdx}"]`);
        if (row) {
            row.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
    }, 50);
}

// Populate Materials Dialog modal inputs dynamically
function renderMaterialConfigModal() {
    const modalBody = document.getElementById("materialsModalBody");
    modalBody.innerHTML = "";
    
    const editableMaterials = [
        { key: 1, label: "Material 1 (SiO2 Silica)", color: "#1f77b4" },
        { key: 2, label: "Material 2 (TiGermania)", color: "#e377c2" }
    ];
    
    const listContainer = document.createElement("div");
    listContainer.className = "modal-materials-list";
    
    editableMaterials.forEach(mat => {
        const p = materialParamsState[mat.key];
        const itemHtml = `
            <div class="modal-material-item" data-id="${mat.key}">
                <div class="modal-material-title-row">
                    <span class="modal-material-color" style="background-color: ${mat.color}"></span>
                    <span class="modal-material-name">${mat.label}</span>
                </div>
                <div class="grid-2-col">
                    <div class="input-field">
                        <label>Refractive Index (n)</label>
                        <input type="number" class="mat-input-n" step="0.001" value="${p.n}">
                    </div>
                    <div class="input-field">
                        <label>Extinction Coeff (k)</label>
                        <input type="number" class="mat-input-k" step="1e-8" value="${p.k}">
                    </div>
                </div>
            </div>
        `;
        listContainer.insertAdjacentHTML("beforeend", itemHtml);
    });
    
    modalBody.appendChild(listContainer);
    
    // Bind listeners to inputs
    listContainer.querySelectorAll(".modal-material-item").forEach(item => {
        const key = parseInt(item.getAttribute("data-id"));
        const nInput = item.querySelector(".mat-input-n");
        const kInput = item.querySelector(".mat-input-k");
        
        const updateParams = () => {
            materialParamsState[key].n = parseFloat(nInput.value) || 1.0;
            materialParamsState[key].k = parseFloat(kInput.value) || 0.0;
            
            updateOpticalThicknessDisplay();
            drawStackPlot();
            triggerSimulation();
        };
        
        nInput.addEventListener("change", updateParams);
        kInput.addEventListener("change", updateParams);
    });
}

// Set up UI Event listeners
function setupEventListeners() {
    // Open settings and materials modals
    document.getElementById("btnOpenSettings").addEventListener("click", () => {
        document.getElementById("settingsModal").showModal();
    });
    
    document.getElementById("btnOpenMaterials").addEventListener("click", () => {
        renderMaterialConfigModal();
        document.getElementById("materialsModal").showModal();
    });
    
    document.getElementById("btnReset").addEventListener("click", resetToDefault);
    
    // Layer editor inputs
    const editMat = document.getElementById("editMaterial");
    const editThickText = document.getElementById("editThicknessText");
    const editThickSlider = document.getElementById("editThicknessSlider");
    
    editMat.addEventListener("change", (e) => {
        if (selectedLayerIndex === null) return;
        const val = parseInt(e.target.value);
        layersState[selectedLayerIndex].material = val;
        
        renderStackEditor();
        drawStackPlot();
        triggerSimulation();
    });
    
    editThickText.addEventListener("input", (e) => {
        if (selectedLayerIndex === null) return;
        const val = parseFloat(e.target.value) || 0;
        layersState[selectedLayerIndex].thickness = val;
        editThickSlider.value = Math.min(val, 500);
        
        updateOpticalThicknessDisplay();
        const n = materialParamsState[layersState[selectedLayerIndex].material].n;
        const laserWavelength = parseFloat(document.getElementById("inputLambda").value) || 1064.0;
        document.getElementById("editOpticalThickness").textContent = ((val * n) / laserWavelength).toFixed(5);
        
        drawStackPlot();
        triggerSimulationDebounced();
    });
    
    editThickSlider.addEventListener("input", (e) => {
        if (selectedLayerIndex === null) return;
        const val = parseFloat(e.target.value);
        layersState[selectedLayerIndex].thickness = val;
        editThickText.value = val.toFixed(2);
        
        updateOpticalThicknessDisplay();
        const n = materialParamsState[layersState[selectedLayerIndex].material].n;
        const laserWavelength = parseFloat(document.getElementById("inputLambda").value) || 1064.0;
        document.getElementById("editOpticalThickness").textContent = ((val * n) / laserWavelength).toFixed(5);
        
        drawStackPlot();
        triggerSimulationDebounced();
    });
    
    // Add Layer Front button
    document.getElementById("btnAddLayerFront").addEventListener("click", () => {
        const firstLayer = layersState[0];
        const newMaterial = firstLayer ? (firstLayer.material === 1 ? 2 : 1) : 1;
        layersState.unshift({
            thickness: 100.0,
            material: newMaterial
        });
        selectedLayerIndex = 0;
        renderStackEditor();
        drawStackPlot();
        triggerSimulation();
        
        const tableContainer = document.querySelector(".table-container");
        tableContainer.scrollTop = 0;
    });
    
    // Add Layer Back button
    document.getElementById("btnAddLayerBack").addEventListener("click", () => {
        const lastLayer = layersState[layersState.length - 1];
        const newMaterial = lastLayer ? (lastLayer.material === 1 ? 2 : 1) : 1;
        layersState.push({
            thickness: 100.0,
            material: newMaterial
        });
        selectedLayerIndex = layersState.length - 1;
        renderStackEditor();
        drawStackPlot();
        triggerSimulation();
        
        const tableContainer = document.querySelector(".table-container");
        tableContainer.scrollTop = tableContainer.scrollHeight;
    });
    
    // Import/Export bindings
    document.getElementById("btnExportJson").addEventListener("click", exportStackJson);
    document.getElementById("btnImportJson").addEventListener("click", () => {
        document.getElementById("fileInputJson").click();
    });
    document.getElementById("fileInputJson").addEventListener("change", importStackJson);
    
    document.getElementById("btnExportCsv").addEventListener("click", exportStackCsv);
    document.getElementById("btnImportCsv").addEventListener("click", () => {
        document.getElementById("fileInputCsv").click();
    });
    document.getElementById("fileInputCsv").addEventListener("change", importStackCsv);
}

// Tabs click handler
function setupTabSystem() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
            
            // Critical Plotly fix: force resizing when a tab is made visible so charts render correctly
            const plotIds = ["plotSpectrum", "plotEFI", "plotNoise"];
            plotIds.forEach(id => {
                const p = document.getElementById(id);
                if (p && p.data) {
                    Plotly.Plots.resize(p);
                }
            });
        });
    });
}

// Debounce simulation requests
function triggerSimulationDebounced() {
    if (debounceTimer) {
        clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(() => {
        triggerSimulation();
    }, 150);
}

// Call FastAPI backend to simulate stack properties
async function triggerSimulation() {
    const indicator = document.querySelector(".status-indicator");
    const label = document.querySelector(".status-label");
    indicator.className = "status-indicator status-calculating";
    label.textContent = "Simulating...";
    
    const lambda = parseFloat(document.getElementById("inputLambda").value) || 1064.0;
    const targetLambda = parseFloat(document.getElementById("inputTargetLambda").value) || 532.0;
    const wBeam = parseFloat(document.getElementById("inputWBeam").value) || 0.062;
    const temp = parseFloat(document.getElementById("inputTemp").value) || 293.0;
    const polarisation = document.getElementById("selectPolarisation").value;
    const angle = parseFloat(document.getElementById("inputAngle").value) || 0.0;
    
    const payload = {
        layers: layersState,
        materialParams: materialParamsState,
        lambda_: lambda,
        wBeam: wBeam,
        Temp: temp,
        polarisation: polarisation,
        angle: angle,
        target_lambdas: [lambda, targetLambda]
    };
    
    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`Simulation failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        indicator.className = "status-indicator status-online";
        label.textContent = "Online";
        
        // Render updated Monospace text summary console box
        updateConsoleOutput(data, lambda, targetLambda);
        
        // Draw the plots
        drawPlots(data, lambda, targetLambda);
        
    } catch (err) {
        console.error("Simulation error:", err);
        indicator.className = "status-indicator status-offline";
        label.textContent = "Offline";
        document.getElementById("consoleOutput").textContent = `Error: ${err.message}\nEnsure the local FastAPI server is running.`;
    }
}

// Helper to look up target wavelength data, avoiding float representation mismatches (e.g. "1064.0" vs "1064")
function getTargetData(targets, wavelength) {
    if (!targets) return null;
    const targetWl = parseFloat(wavelength);
    for (const key in targets) {
        if (Math.abs(parseFloat(key) - targetWl) < 0.1) {
            return targets[key];
        }
    }
    return null;
}

// Format the verification console output string exactly as requested
function updateConsoleOutput(data, lambda, targetLambda) {
    const consoleEl = document.getElementById("consoleOutput");
    const mainTarget = getTargetData(data.targets, lambda);
    const secondaryTarget = getTargetData(data.targets, targetLambda);
    
    if (!mainTarget || !secondaryTarget) {
        consoleEl.textContent = "Error: Could not retrieve target wavelength data from simulation response.";
        return;
    }
    
    const m1 = data.materials["1"] || { name: "Silica (SiO2)", layers: 0, thickness: 0, n: 1.45, k: 0 };
    const m2 = data.materials["2"] || { name: "TiGermania", layers: 0, thickness: 0, n: 1.866, k: 2e-7 };
    
    const t_ppm = mainTarget.transmission * 1e6;
    const t_pct = secondaryTarget.transmission * 100;
    
    const report = `Coating Properties:

Laser Wavelength:               ${lambda.toFixed(2)} nm
Number of Materials:            2
Total Physical Thickness:       ${data.total_thickness.toFixed(2)} nm
absorption:                     ${data.absorption_ppm.toFixed(2)} ppm
CTN_at_100Hz:                   ${data.noise_100hz.brownian.toExponential(15)}
Reflectivity_${lambda.toFixed(0)}:              ${mainTarget.reflectivity.toFixed(5)}
Transmission_${lambda.toFixed(0)}:              ${t_ppm.toFixed(5)} ppm
stack_name:                     RL - Rank 1

--------- Material 1 -------------

No. Layers:\t\t\t${m1.layers}
Total Physical Thickness:\t${m1.thickness.toFixed(2)} nm
Refractive Index:\t\t${m1.n.toFixed(2)}

--------- Material 2 -------------

No. Layers:\t\t\t${m2.layers}
Total Physical Thickness:\t${m2.thickness.toFixed(2)} nm
Refractive Index:\t\t${m2.n.toFixed(2)}
Transmission at ${targetLambda.toFixed(0)} nm is ${t_pct.toFixed(2)} %`;

    consoleEl.textContent = report;
}

// Draw Plotly charts (Spectrum, EFI, Noise)
function drawPlots(data, lambda, targetLambda) {
    const layoutDefaults = {
        paper_bgcolor: "#1e1e1e",
        plot_bgcolor: "#1e1e1e",
        font: { color: "#e0e0e0", family: "Inter, sans-serif" },
        margin: { t: 30, r: 20, b: 45, l: 55 },
        xaxis: { 
            gridcolor: "rgba(255,255,255,0.06)", 
            linecolor: "#2d2d2d",
            tickfont: { size: 9 }
        },
        yaxis: { 
            gridcolor: "rgba(255,255,255,0.06)", 
            linecolor: "#2d2d2d",
            tickfont: { size: 9 }
        },
        showlegend: true,
        legend: { 
            font: { size: 9 },
            bgcolor: "rgba(11,12,16,0.85)",
            bordercolor: "#2d2d2d",
            borderwidth: 1
        }
    };
    
    // --- 1. Spectrum Plot ---
    const t_percent = data.charts.spectrum.transmission.map(t => t * 100);
    const traceSpectrum = {
        x: data.charts.spectrum.wavelengths,
        y: t_percent,
        mode: "lines",
        name: "Transmission",
        line: { color: "#00bcd4", width: 2 }
    };
    
    const layoutSpectrum = JSON.parse(JSON.stringify(layoutDefaults));
    layoutSpectrum.xaxis.title = "Wavelength (nm)";
    layoutSpectrum.yaxis.title = "Transmission (%)";
    layoutSpectrum.shapes = [
        {
            type: "line",
            x0: lambda, x1: lambda,
            y0: 0, y1: 100,
            line: { color: "#ff8c00", width: 1.5, dash: "dash" }
        },
        {
            type: "line",
            x0: targetLambda, x1: targetLambda,
            y0: 0, y1: 100,
            line: { color: "#ab7df8", width: 1.5, dash: "dash" }
        }
    ];
    layoutSpectrum.annotations = [
        { x: lambda, y: 80, text: `${lambda} nm`, showarrow: false, font: { color: "#ff8c00", size: 10 } },
        { x: targetLambda, y: 80, text: `${targetLambda} nm`, showarrow: false, font: { color: "#ab7df8", size: 10 } }
    ];
    
    Plotly.newPlot("plotSpectrum", [traceSpectrum], layoutSpectrum, { responsive: true, displayModeBar: false });
    
    // --- 2. EFI Profile Plot ---
    const traceEFI = {
        x: data.charts.efi.depths,
        y: data.charts.efi.intensity,
        mode: "lines",
        name: "|E/E₀|²",
        line: { color: "#2ec4b6", width: 2 }
    };
    
    const layoutEFI = JSON.parse(JSON.stringify(layoutDefaults));
    layoutEFI.xaxis.title = "Coating Depth (nm)";
    layoutEFI.yaxis.title = "Normalised Field Intensity";
    
    // Layer interfaces vertical bars
    const layerIdxArr = data.charts.efi.layer_idx;
    const depthsArr = data.charts.efi.depths;
    const shapes = [];
    
    let lastIdx = -999;
    for (let i = 0; i < layerIdxArr.length; i++) {
        const curIdx = layerIdxArr[i];
        if (curIdx !== lastIdx && lastIdx !== -999 && curIdx >= 0) {
            shapes.push({
                type: "line",
                x0: depthsArr[i], x1: depthsArr[i],
                y0: 0, y1: Math.max(...data.charts.efi.intensity) * 1.1,
                line: { color: "rgba(255,255,255,0.12)", width: 1, dash: "dot" }
            });
        }
        lastIdx = curIdx;
    }
    layoutEFI.shapes = shapes;
    
    Plotly.newPlot("plotEFI", [traceEFI], layoutEFI, { responsive: true, displayModeBar: false });
    
    // --- 3. Noise Spectrum Plot ---
    const traceNoiseBrownian = {
        x: data.charts.noise.frequencies,
        y: data.charts.noise.brownian,
        mode: "lines",
        name: "Brownian Noise",
        line: { color: "#ff5252", width: 2 }
    };
    
    const traceNoiseTO = {
        x: data.charts.noise.frequencies,
        y: data.charts.noise.thermo_optic,
        mode: "lines",
        name: "Thermo-Optic Noise",
        line: { color: "#ff8c00", width: 2 }
    };
    
    const traceNoiseTE = {
        x: data.charts.noise.frequencies,
        y: data.charts.noise.thermo_elastic,
        mode: "lines",
        name: "TE Component",
        line: { color: "#ab7df8", width: 1.5, dash: "dash" }
    };
    
    const traceNoiseTR = {
        x: data.charts.noise.frequencies,
        y: data.charts.noise.thermo_refractive,
        mode: "lines",
        name: "TR Component",
        line: { color: "#00f5d4", width: 1.5, dash: "dash" }
    };
    
    const layoutNoise = JSON.parse(JSON.stringify(layoutDefaults));
    layoutNoise.xaxis.title = "Frequency (Hz)";
    layoutNoise.xaxis.type = "log";
    layoutNoise.yaxis.title = "Spectral Density (m/√Hz)";
    layoutNoise.yaxis.type = "log";
    
    Plotly.newPlot("plotNoise", [
        traceNoiseBrownian,
        traceNoiseTO,
        traceNoiseTE,
        traceNoiseTR
    ], layoutNoise, { responsive: true, displayModeBar: false });
    
    // Force resize current visible tab plots to ensure correct layout display
    const activeTab = document.querySelector(".tab-btn.active").getAttribute("data-tab");
    const activePlotId = activeTab === "tabSpectrum" ? "plotSpectrum" : (activeTab === "tabEFI" ? "plotEFI" : "plotNoise");
    Plotly.Plots.resize(document.getElementById(activePlotId));
}

// Export / Import logic
function exportStackJson() {
    const payload = {
        stack_name: "Coating stack design configuration",
        layers: layersState,
        materialParams: materialParamsState,
        globals: {
            lambda: parseFloat(document.getElementById("inputLambda").value) || 1064.0,
            targetLambda: parseFloat(document.getElementById("inputTargetLambda").value) || 532.0,
            wBeam: parseFloat(document.getElementById("inputWBeam").value) || 0.062,
            temp: parseFloat(document.getElementById("inputTemp").value) || 293.0,
            polarisation: document.getElementById("selectPolarisation").value,
            angle: parseFloat(document.getElementById("inputAngle").value) || 0.0
        }
    };
    
    const blob = new Blob([JSON.stringify(payload, null, 4)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "coating_stack_design.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function importStackJson(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
        try {
            const data = JSON.parse(evt.target.result);
            if (!data.layers || !data.materialParams) {
                throw new Error("Invalid design JSON. Must contain 'layers' and 'materialParams'.");
            }
            
            layersState = data.layers;
            materialParamsState = data.materialParams;
            
            if (data.globals) {
                document.getElementById("inputLambda").value = data.globals.lambda || 1064.0;
                document.getElementById("inputTargetLambda").value = data.globals.targetLambda || 532.0;
                document.getElementById("inputWBeam").value = data.globals.wBeam || 0.062;
                document.getElementById("inputTemp").value = data.globals.temp || 293.0;
                document.getElementById("selectPolarisation").value = data.globals.polarisation || "p";
                document.getElementById("inputAngle").value = data.globals.angle || 0.0;
            }
            
            selectedLayerIndex = 0;
            renderStackEditor();
            drawStackPlot();
            triggerSimulation();
            
            alert("Design JSON imported successfully!");
        } catch (err) {
            alert(`Error reading JSON: ${err.message}`);
        }
    };
    reader.readAsText(file);
    e.target.value = "";
}

function exportStackCsv() {
    let csvContent = "Layer_Number,Material_Index,Physical_Thickness_nm\n";
    layersState.forEach((layer, idx) => {
        csvContent += `${idx + 1},${layer.material},${layer.thickness.toFixed(6)}\n`;
    });
    
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "coating_stack_layers.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function importStackCsv(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
        try {
            const text = evt.target.result;
            const lines = text.split("\n");
            
            const importedLayers = [];
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;
                
                const parts = line.split(",");
                if (parts.length < 3) continue;
                
                const mat = parseInt(parts[1]);
                const thick = parseFloat(parts[2]);
                
                if (isNaN(mat) || isNaN(thick)) {
                    throw new Error(`Invalid numbers on row ${i + 1}`);
                }
                
                importedLayers.push({
                    thickness: thick,
                    material: mat
                });
            }
            
            if (importedLayers.length === 0) {
                throw new Error("No valid layer rows found in CSV.");
            }
            
            layersState = importedLayers;
            selectedLayerIndex = 0;
            
            renderStackEditor();
            drawStackPlot();
            triggerSimulation();
            
            alert(`Imported ${layersState.length} layers from CSV!`);
        } catch (err) {
            alert(`Error reading CSV: ${err.message}`);
        }
    };
    reader.readAsText(file);
    e.target.value = "";
}

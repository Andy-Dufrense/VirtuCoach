/**
 * VirtuCoach - 手型检查模块
 * 和弦手型检查 + 技巧检查 + 视频录制
 * 依赖: app.js (DOM 引用、全局状态、showToast、resetUI)
 */

// ========== 手型检查子模式切换 ==========

function switchHandSubmode(submode) {
    handSubmode = submode;
    document.querySelectorAll(".submode-btn").forEach(b => b.classList.remove("active"));
    document.getElementById(submode === "chord" ? "submodeChordBtn" : "submodeTechniqueBtn").classList.add("active");

    if (submode === "chord") {
        handCheckTitle.textContent = "检查和弦手型";
        handCheckHint.textContent = "输入你要检查的和弦，上传3-5秒短视频";
        chordCheckArea.style.display = "block";
        techniqueCheckArea.style.display = "none";
        techniqueRefHint.style.display = "none";
        if (chordCheckState.selectedChord) {
            chordUploadArea.style.display = "block";
            var displayName = chordCheckState.chordName || chordCheckState.selectedChord;
            chordInput.value = displayName;
            updateRecordInstructions(chordCheckState.chordId || chordCheckState.selectedChord, displayName);
            if (chordCheckState.recordedBlob) {
                chordVideoReady.style.display = "block";
                analyzeChordBtn.style.display = "inline-block";
            } else {
                chordVideoReady.style.display = "none";
                analyzeChordBtn.style.display = "none";
            }
        } else {
            chordUploadArea.style.display = "none";
            chordVideoPreviewWrap.style.display = "none";
            chordVideoReady.style.display = "none";
            analyzeChordBtn.style.display = "none";
        }
    } else if (submode === "technique") {
        handCheckTitle.textContent = "检查技巧";
        handCheckHint.textContent = "选择你要检查的技巧，上传演示视频";
        chordCheckArea.style.display = "none";
        techniqueCheckArea.style.display = "block";
        chordMatchHint.style.display = "none";
        chordVideoPreviewWrap.style.display = "none";
        stopCameraPreview();
        if (techniqueCheckState.selectedTechnique) {
            chordUploadArea.style.display = "block";
            techniqueRefHint.style.display = "block";
            techniqueSelect.value = techniqueCheckState.selectedTechnique;
            var techName = techniqueSelect.options[techniqueSelect.selectedIndex] && techniqueSelect.options[techniqueSelect.selectedIndex].text;
            if (techName) {
                updateTechniqueInstructions(techniqueCheckState.selectedTechnique, techName);
            }
            if (techniqueCheckState.recordedBlob) {
                chordVideoReady.style.display = "block";
                analyzeChordBtn.style.display = "inline-block";
            } else {
                chordVideoReady.style.display = "none";
                analyzeChordBtn.style.display = "none";
            }
        } else {
            chordUploadArea.style.display = "none";
            techniqueRefHint.style.display = "none";
            chordVideoReady.style.display = "none";
            analyzeChordBtn.style.display = "none";
        }
    }
}

function updateTechniqueInstructions(techniqueId, techName) {
    var instrList = document.getElementById("recordInstructionsList");
    if (!instrList) return;
    var isLeftHand = ["basic-fretting", "barre-technique", "hammer-on", "pull-off", "bend", "vibrato", "slide", "slide-in", "slide-out", "natural-harmonics", "left-hand-tapping", "trill"].indexOf(techniqueId) >= 0;
    if (isLeftHand) {
        instrList.innerHTML = '<li>左手演示 <b id="chordNameInHint">' + techName + '</b> 技巧</li>' +
            '<li>确保左手手指和琴颈在镜头内清晰可见</li>' +
            '<li>动作连贯、发音清晰即可，不需要完美</li>' +
            '<li>手正对或略侧对镜头</li>';
    } else {
        instrList.innerHTML = '<li>右手演示 <b id="chordNameInHint">' + techName + '</b> 技巧</li>' +
            '<li>确保右手和琴弦在镜头内清晰可见</li>' +
            '<li>动作连贯、发音清晰即可，不需要完美</li>' +
            '<li>手正对或略侧对镜头</li>';
    }
}

function onTechniqueSelect() {
    var val = techniqueSelect.value;
    techniqueCheckState.selectedTechnique = val;
    if (val) {
        chordUploadArea.style.display = "block";
        techniqueRefHint.style.display = "block";
        chordVideoPreviewWrap.style.display = "none";
        var techName = techniqueSelect.options[techniqueSelect.selectedIndex].text;
        document.getElementById("chordNameInHint").textContent = techName;
        updateTechniqueInstructions(val, techName);
        if (techniqueCheckState.recordedBlob) {
            chordVideoReady.style.display = "block";
            analyzeChordBtn.style.display = "inline-block";
        }
    } else {
        chordUploadArea.style.display = "none";
        techniqueRefHint.style.display = "none";
        chordVideoPreviewWrap.style.display = "none";
        chordVideoReady.style.display = "none";
        analyzeChordBtn.style.display = "none";
    }
}

// ========== 和弦手型检查 ==========

async function loadChordList() {
    try {
        const resp = await fetch(`${API_BASE}/api/chords`, {
            headers: window.VirtuCoach ? window.VirtuCoach.getAuthHeaders() : {}
        });
        const data = await resp.json();
        availableChords = data.chords || [];
        if (availableChords.length === 0) {
            availableChords = FALLBACK_CHORDS;
            console.log("后端和弦列表为空，使用本地降级列表");
        }
    } catch (e) {
        console.log("和弦列表加载失败，使用本地降级列表:", e.message);
        availableChords = FALLBACK_CHORDS;
    }
    chordDatalist.innerHTML = availableChords.map(c =>
        `<option value="${c.name}">${c.name}</option>`
    ).join("");
}

var CHORD_STRING_PATTERNS = {
    "C": "⑤→④→③→②→①", "C-major": "⑤→④→③→②→①", "C7": "⑤→④→③→②→①",
    "D": "④→③→②→①", "D-major": "④→③→②→①", "D7": "④→③→②→①", "Dm": "④→③→②→①", "Dm7": "④→③→②→①",
    "E": "⑥→⑤→④→③→②→①", "E-major": "⑥→⑤→④→③→②→①", "Em": "⑥→⑤→④→③→②→①", "E7": "⑥→⑤→④→③→②→①",
    "F": "⑥→⑤→④→③→②→①", "Fmaj7": "⑥→⑤→④→③→②→①",
    "G": "⑥→⑤→④→③→②→①", "G-major": "⑥→⑤→④→③→②→①", "G7": "⑥→⑤→④→③→②→①",
    "A": "⑤→④→③→②→①", "A-major": "⑤→④→③→②→①", "Am": "⑤→④→③→②→①", "A7": "⑤→④→③→②→①",
    "B": "⑤→④→③→②→①", "B-major": "⑤→④→③→②→①", "Bm": "⑤→④→③→②→①", "B7": "⑤→④→③→②→①",
    "Cmaj7": "⑤→④→③→②→①", "Am7": "⑤→④→③→②→①",
    "D-over-Fsharp": "⑥→④→③→②→①",
};

function updateRecordInstructions(chordId, chordName) {
    var instrList = document.getElementById("recordInstructionsList");
    if (!instrList) return;
    var pattern = CHORD_STRING_PATTERNS[chordId] || CHORD_STRING_PATTERNS[chordName] || "逐根拨弦";
    instrList.innerHTML = '<li>按好 <b id="chordNameInHint">' + (chordName || chordId) + '</b> 和弦</li>' +
        '<li>从低音弦逐根拨到高音弦（' + pattern + '）</li>' +
        '<li>每根弦尽量弹饱满、发音清晰即可</li>' +
        '<li>左手正对或略侧对镜头，保证手指和琴弦清晰可见</li>';
}

function onChordInput() {
    const input = chordInput.value.trim();
    if (!input) {
        chordUploadArea.style.display = "none";
        chordVideoPreviewWrap.style.display = "none";
        chordVideoReady.style.display = "none";
        analyzeChordBtn.style.display = "none";
        chordCheckState.selectedChord = "";
        chordCheckState.chordId = "";
        chordCheckState.chordName = "";
        chordCheckState.hasKnowledge = false;
        chordMatchHint.style.display = "none";
        return;
    }

    var normInput = input.replace(/\s+/g, '');
    var matched = null;
    var bestScore = 0;
    for (var i = 0; i < availableChords.length; i++) {
        var c = availableChords[i];
        var score = 0;
        if (c.id === input) score = 100;
        else if (c.name === input) score = 90;
        else if (c.id && c.name && input.indexOf(c.id) === 0) score = 80;
        else if (c.name && c.name.replace(/\s+/g, '') === normInput) score = 70;
        if (score > bestScore) { bestScore = score; matched = c; }
    }

    if (matched) {
        if (chordCheckState.chordId && chordCheckState.chordId !== matched.id) {
            chordCheckState.recordedBlob = null;
            chordCheckState.recordedUrl = null;
            chordVideoReady.style.display = "none";
            analyzeChordBtn.style.display = "none";
        }
        chordCheckState.selectedChord = matched.id;
        chordCheckState.chordId = matched.id;
        chordCheckState.chordName = matched.name || matched.id;
        chordCheckState.hasKnowledge = true;
        chordMatchHint.style.display = "none";
        updateRecordInstructions(matched.id, chordCheckState.chordName);
    } else {
        chordCheckState.selectedChord = input;
        chordCheckState.chordId = "";
        chordCheckState.chordName = input;
        chordCheckState.hasKnowledge = false;
        chordMatchHint.style.display = "block";
        chordMatchHint.textContent = `⚠️ "${input}" 暂无知识库数据，分析将使用通用手型标准`;
        updateRecordInstructions("", input);
    }

    chordUploadArea.style.display = "block";
    chordVideoPreviewWrap.style.display = "none";
}

async function startCameraPreview() {
    if (chordCheckState.cameraStream) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: true
        });
        chordCheckState.cameraStream = stream;
        chordPreview.srcObject = stream;
        chordPreview.style.display = "block";
        document.getElementById("chordVideoPlaceholder").style.display = "none";
    } catch (e) {
        console.log("摄像头不可用:", e.message);
        document.getElementById("chordVideoPlaceholder").innerHTML = "<span>⚠️ 摄像头不可用<br>请上传已录好的视频文件</span>";
    }
}

function startChordRecord() {
    var isTechniqueMode = (handSubmode === "technique");
    var checkTarget = isTechniqueMode ? techniqueCheckState.selectedTechnique : chordCheckState.selectedChord;

    if (!checkTarget) {
        showToast(isTechniqueMode ? "请先选择要检查的技巧" : "请先输入要检查的和弦", "error");
        return;
    }

    if (isTechniqueMode) {
        var techName = techniqueSelect.options[techniqueSelect.selectedIndex].text;
        document.getElementById("chordNameInHint").textContent = techName;
        document.getElementById("chordRecordInstructions").innerHTML =
            '<p><strong>录制要求：</strong></p><ol>' +
            '<li>演示 <b>' + techName + '</b> 技巧</li>' +
            '<li>右手在镜头内清晰可见</li>' +
            '<li>保持自然手型，正常速度弹奏</li>' +
            '<li>录制3-5秒即可</li></ol>';
    }

    if (!chordCheckState.cameraStream) {
        chordVideoPreviewWrap.style.display = "block";
        startCameraPreview().then(() => {
            if (chordCheckState.cameraStream) {
                startChordRecord();
            }
        });
        return;
    }

    const stream = chordCheckState.cameraStream;
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9" : "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        if (handSubmode === "technique") {
            techniqueCheckState.recordedBlob = blob;
            if (techniqueCheckState.recordedUrl) URL.revokeObjectURL(techniqueCheckState.recordedUrl);
            techniqueCheckState.recordedUrl = URL.createObjectURL(blob);
        } else {
            chordCheckState.recordedBlob = blob;
            if (chordCheckState.recordedUrl) URL.revokeObjectURL(chordCheckState.recordedUrl);
            chordCheckState.recordedUrl = URL.createObjectURL(blob);
        }
        chordVideoResult.src = (handSubmode === "technique" ? techniqueCheckState.recordedUrl : chordCheckState.recordedUrl);
        chordVideoReady.style.display = "block";
        analyzeChordBtn.style.display = "inline-block";
        startRecordBtn.style.display = "inline-block";
        startRecordBtn.textContent = "🔴 重新录制";
        recordTimer.style.display = "none";
    };

    chordCheckState.mediaRecorder = recorder;
    chordCheckState.recordingStartTime = Date.now();
    recorder.start();
    startRecordBtn.disabled = true;
    startRecordBtn.textContent = "⏺ 录制中...";
    recordTimer.style.display = "inline-block";
    chordVideoReady.style.display = "none";

    updateRecordTimer();
}

function updateRecordTimer() {
    const elapsed = (Date.now() - chordCheckState.recordingStartTime) / 1000;
    const remaining = Math.max(0, chordCheckState.maxRecordSec - elapsed);
    recordTimer.textContent = `录制中 ${elapsed.toFixed(1)}s / ${chordCheckState.maxRecordSec}s`;
    if (remaining <= 0) {
        stopChordRecord();
        return;
    }
    chordCheckState.recordingTimerId = setTimeout(updateRecordTimer, 100);
}

function stopChordRecord() {
    if (chordCheckState.recordingTimerId) {
        clearTimeout(chordCheckState.recordingTimerId);
        chordCheckState.recordingTimerId = null;
    }
    if (chordCheckState.mediaRecorder && chordCheckState.mediaRecorder.state === "recording") {
        chordCheckState.mediaRecorder.stop();
    }
    startRecordBtn.disabled = false;
}

// ========== 视频分析模式录制 ==========

async function startVideoRecordCamera() {
    if (videoRecordState.cameraStream) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: true
        });
        videoRecordState.cameraStream = stream;
        videoRecordPreview.srcObject = stream;
        videoRecordPreview.style.display = "block";
        videoRecordPlaceholder.style.display = "none";
    } catch (e) {
        console.log("摄像头不可用:", e.message);
        videoRecordPlaceholder.innerHTML = "<span>⚠️ 摄像头不可用<br>请上传已录好的视频文件</span>";
    }
}

function startVideoRecord() {
    if (!videoRecordState.cameraStream) {
        videoRecordPreviewWrap.style.display = "block";
        startVideoRecordCamera().then(() => {
            if (videoRecordState.cameraStream) startVideoRecord();
        });
        return;
    }

    const stream = videoRecordState.cameraStream;
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9" : "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        videoRecordState.recordedBlob = blob;
        if (videoRecordState.recordedUrl) URL.revokeObjectURL(videoRecordState.recordedUrl);
        videoRecordState.recordedUrl = URL.createObjectURL(blob);
        videoRecordResult.src = videoRecordState.recordedUrl;
        videoRecordReady.style.display = "block";
        startVideoRecordBtn.style.display = "inline-block";
        startVideoRecordBtn.textContent = "🔴 重新录制";
        videoRecordTimer.style.display = "none";
        uploadRecordedVideo(blob);
    };

    videoRecordState.mediaRecorder = recorder;
    videoRecordState.recordingStartTime = Date.now();
    recorder.start();
    startVideoRecordBtn.disabled = true;
    startVideoRecordBtn.textContent = "⏺ 录制中...";
    videoRecordTimer.style.display = "inline-block";
    videoRecordReady.style.display = "none";

    updateVideoRecordTimer();
}

function updateVideoRecordTimer() {
    const elapsed = (Date.now() - videoRecordState.recordingStartTime) / 1000;
    const remaining = Math.max(0, videoRecordState.maxRecordSec - elapsed);
    videoRecordTimer.textContent = `录制中 ${elapsed.toFixed(0)}s / ${videoRecordState.maxRecordSec}s`;
    if (remaining <= 0) {
        stopVideoRecord();
        return;
    }
    videoRecordState.recordingTimerId = setTimeout(updateVideoRecordTimer, 500);
}

function stopVideoRecord() {
    if (videoRecordState.recordingTimerId) {
        clearTimeout(videoRecordState.recordingTimerId);
        videoRecordState.recordingTimerId = null;
    }
    if (videoRecordState.mediaRecorder && videoRecordState.mediaRecorder.state === "recording") {
        videoRecordState.mediaRecorder.stop();
    }
    startVideoRecordBtn.disabled = false;
}

function uploadRecordedVideo(blob) {
    const file = new File([blob], "recording.webm", { type: blob.type || "video/webm" });
    const dt = new DataTransfer();
    dt.items.add(file);
    videoInput.files = dt.files;
    uploadVideo(file);
}

function resetVideoRecord() {
    if (videoRecordState.mediaRecorder && videoRecordState.mediaRecorder.state === "recording") {
        videoRecordState.mediaRecorder.stop();
    }
    if (videoRecordState.recordingTimerId) {
        clearTimeout(videoRecordState.recordingTimerId);
    }
    if (videoRecordState.cameraStream) {
        videoRecordState.cameraStream.getTracks().forEach(t => t.stop());
        videoRecordState.cameraStream = null;
    }
    if (videoRecordState.recordedUrl) {
        URL.revokeObjectURL(videoRecordState.recordedUrl);
    }
    videoRecordState = {
        recordedBlob: null,
        recordedUrl: null,
        mediaRecorder: null,
        cameraStream: null,
        recordingStartTime: 0,
        recordingTimerId: null,
        maxRecordSec: 30,
    };
    videoRecordPreview.style.display = "none";
    videoRecordPreview.srcObject = null;
    videoRecordPlaceholder.style.display = "flex";
    videoRecordReady.style.display = "none";
    videoRecordPreviewWrap.style.display = "none";
    startVideoRecordBtn.style.display = "inline-block";
    startVideoRecordBtn.textContent = "📷 录制视频";
    startVideoRecordBtn.disabled = false;
    videoRecordTimer.style.display = "none";
}

function handleChordVideoUpload() {
    const file = chordVideoInput.files[0];
    if (!file) return;
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (![".mp4", ".mov", ".webm", ".avi", ".mkv"].includes(ext)) {
        showToast("仅支持视频格式 (MP4/MOV/WEBM等)", "error");
        return;
    }
    if (file.size > 100 * 1024 * 1024) {
        showToast("视频不能超过 100MB", "error");
        return;
    }
    if (handSubmode === "technique") {
        techniqueCheckState.recordedBlob = file;
        if (techniqueCheckState.recordedUrl) URL.revokeObjectURL(techniqueCheckState.recordedUrl);
        techniqueCheckState.recordedUrl = URL.createObjectURL(file);
    } else {
        chordCheckState.recordedBlob = file;
        if (chordCheckState.recordedUrl) URL.revokeObjectURL(chordCheckState.recordedUrl);
        chordCheckState.recordedUrl = URL.createObjectURL(file);
    }
    chordVideoResult.src = (handSubmode === "technique" ? techniqueCheckState.recordedUrl : chordCheckState.recordedUrl);
    chordVideoReady.style.display = "block";
    analyzeChordBtn.style.display = "inline-block";
    if (handSubmode !== "technique" && !chordInput.value.trim()) {
        showToast("⚠️ 请先输入要检查的和弦", "");
    }
    if (handSubmode === "technique" && !techniqueSelect.value) {
        showToast("⚠️ 请先选择要检查的技巧", "");
    }
}

async function analyzeChordHand() {
    var isTechniqueMode = (handSubmode === "technique");
    var checkTarget = isTechniqueMode ? techniqueCheckState.selectedTechnique : chordCheckState.selectedChord;

    if (!checkTarget) {
        showToast(isTechniqueMode ? "请先选择要检查的技巧" : "请先选择要检查的和弦", "error");
        return;
    }
    if (!chordCheckState.recordedBlob && !techniqueCheckState.recordedBlob) {
        showToast("请先录制或上传视频", "error");
        return;
    }

    var blob = chordCheckState.recordedBlob || techniqueCheckState.recordedBlob;
    var formData = new FormData();
    formData.append("video", blob, blob.name || "recording.webm");

    if (isTechniqueMode) {
        formData.append("technique", checkTarget);
        formData.append("mode", "technique");
    } else {
        formData.append("chord", checkTarget);
        formData.append("mode", "chord");
    }
    formData.append("instrument", "guitar");

    window._handAnalyzing = true;
    uploadSection.style.display = "none";
    handResultSection.style.display = "block";

    // Show progress card, hide result card
    var progressCard = document.getElementById("handProgressCard");
    var resultCard = document.getElementById("handResultCard");
    progressCard.style.display = "block";
    resultCard.style.display = "none";

    // Reset progress
    var hProgressBar = document.getElementById("handProgressBar");
    var hProgressMsg = document.getElementById("handProgressMessage");
    hProgressBar.style.width = "0%";
    hProgressMsg.textContent = "准备中...";
    resetHandSteps();

    // Start progress animation
    var handProgressTimer = startHandProgress(hProgressBar, hProgressMsg);
    window._handProgressTimer = handProgressTimer;

    try {
        const resp = await fetch(`${API_BASE}/api/check-chord`, {
            method: "POST",
            headers: window.VirtuCoach ? window.VirtuCoach.getAuthHeaders() : {},
            body: formData,
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `请求失败: ${resp.status}`);
        }
        const data = await resp.json();
        // Complete progress
        stopHandProgress(handProgressTimer);
        hProgressBar.style.width = "100%";
        hProgressMsg.textContent = "分析完成！";
        updateHandStep("hStep4", "done");
        // Short delay to show 100% before showing results
        await new Promise(r => setTimeout(r, 400));
        progressCard.style.display = "none";
        resultCard.style.display = "block";
        renderChordResult(data);
    } catch (e) {
        stopHandProgress(handProgressTimer);
        progressCard.style.display = "none";
        showToast(`分析失败: ${e.message}`, "error");
        resetChordUI();
    } finally {
        window._handAnalyzing = false;
    }
}

function renderChordResult(data) {
    window._lastHandResult = data;

    if (currentMode !== "image") {
        handResultSection.style.display = "none";
        showToast(`${data.chord_name || ""} 手型分析完成！切换回「手型检查」查看结果`, "success");
        return;
    }

    uploadSection.style.display = "none";
    handResultSection.style.display = "block";

    const headerEl = document.querySelector(".hand-result-card h3");
    if (headerEl) {
        headerEl.textContent = `🔍 ${data.chord_name || ""} 和弦手型检查`;
    }

    const score = data.overall_score;
    document.getElementById("handScore").textContent = score !== undefined ? score : "--";

    const badge = document.getElementById("handStatusBadge");
    if (!data.hands_detected) {
        badge.textContent = "⚠️ 未检测到手部";
        badge.style.background = "var(--warning-light)";
        badge.style.color = "var(--warning)";
    } else if (!data.issues || data.issues.length === 0) {
        badge.textContent = "✅ 和弦手型规范";
        badge.style.background = "var(--success-light)";
        badge.style.color = "var(--success)";
    } else {
        const severeCount = data.issues.filter(i => i.severity === "severe").length;
        badge.textContent = severeCount > 0 ? `⚠️ ${data.issues.length} 个问题` : `💡 ${data.issues.length} 个建议`;
        badge.style.background = severeCount > 0 ? "var(--danger-light)" : "var(--warning-light)";
        badge.style.color = severeCount > 0 ? "var(--danger)" : "var(--warning)";
    }

    const qualityNote = data.quality_note || "";
    if (qualityNote) {
        const noteBg = data.audio_quality === "poor" ? "var(--danger-light)" : "var(--warning-light)";
        const noteColor = data.audio_quality === "poor" ? "var(--danger)" : "var(--warning)";
        const noteIcon = data.audio_quality === "poor" ? "⚠️" : "💡";
        const qualityNoteHtml = `<div style="margin-top:12px;padding:10px 12px;background:${noteBg};border-radius:8px;border-left:3px solid ${noteColor};">
            <span style="color:${noteColor};font-size:14px;line-height:1.5;">${noteIcon} ${qualityNote}</span>
        </div>`;
        const issuesContainerRef = document.getElementById("handIssueItems");
        const existingNote = document.getElementById("audioQualityNote");
        if (existingNote) existingNote.remove();
        issuesContainerRef.insertAdjacentHTML("beforebegin", `<div id="audioQualityNote">${qualityNoteHtml}</div>`);
    }

    const stringResults = data.string_results || [];
    let stringResultsHtml = "";
    if (stringResults.length > 0) {
        stringResultsHtml = '<div style="margin-top:12px;padding:12px;background:var(--bg);border-radius:8px;">';
        stringResultsHtml += '<h4 style="margin:0 0 8px 0;">🎵 逐弦音色检测</h4>';
        stringResults.forEach(sr => {
            const statusIcon = sr.ok ? "✅" : (sr.status_text === "信号不足以判断" ? "⚠️" : "❌");
            const statusColor = sr.ok ? "var(--success)" : (sr.status_text === "信号不足以判断" ? "var(--warning)" : "var(--danger)");
            const statusText = sr.status_text || (sr.ok ? "清晰 ✓" : (!sr.has_signal ? "未检测到拨弦" : "信号弱"));
            stringResultsHtml += `<div style="display:flex;align-items:center;gap:8px;padding:4px 0;color:${statusColor};font-size:14px;">
                ${statusIcon} <b>${sr.string_name}</b> ${sr.note || ""} — ${statusText}
            </div>`;
        });
        stringResultsHtml += "</div>";
    }

    document.getElementById("userHandImage").src = data.user_image_url || "";
    const refs = data.reference_images || [];
    if (refs.length > 0) {
        document.getElementById("refHandImage").src = API_BASE + refs[0].image_url;
        document.getElementById("refHandDesc").textContent = "";
    } else {
        document.getElementById("refHandImage").src = "";
        document.getElementById("refHandDesc").textContent = "";
    }

    const issuesContainer = document.getElementById("handIssueItems");
    const noIssuesMsg = document.getElementById("noIssuesMsg");
    issuesContainer.innerHTML = "";

    const issues = data.issues || [];
    if (issues.length === 0) {
        noIssuesMsg.style.display = "block";
    } else {
        noIssuesMsg.style.display = "none";
        const sevLabels = { "mild": "轻微", "moderate": "中等", "severe": "重要" };
        const sevCls = { "mild": "severity-cosmetic", "moderate": "severity-minor", "severe": "severity-critical" };
        let issuesHtml = '<ul>' + issues.map(issue => {
            const sev = issue.severity || "moderate";
            return `<li class="${sevCls[sev] || 'severity-minor'}">
                <b>[${sevLabels[sev] || sev}]</b> <b>${issue.body_part || ''}</b>: ${issue.description || ''}
                ${issue.suggestion ? '<br><span style="color:var(--success);">💡 ' + issue.suggestion + '</span>' : ''}
            </li>`;
        }).join('') + '</ul>';
        issuesContainer.innerHTML += issuesHtml;
    }

    if (stringResultsHtml) {
        issuesContainer.innerHTML += stringResultsHtml;
    }

    // 逐指对比表
    const fingerComparison = data.finger_comparison || [];
    if (fingerComparison.length > 0) {
        const statusIcon = { ok: "✅", deviation: "⚠️", error: "❌" };
        const statusLabel = { ok: "正确", deviation: "偏差", error: "错误" };
        let compareHtml = '<div style="margin-top:12px;padding:12px;background:var(--bg);border-radius:8px;">';
        compareHtml += '<h4 style="margin:0 0 8px 0;">🔍 逐指对比诊断</h4>';
        compareHtml += '<table style="width:100%;font-size:13px;border-collapse:collapse;">';
        compareHtml += '<tr style="border-bottom:1px solid var(--card-border);color:var(--text-dim);"><th style="padding:6px 4px;text-align:left;">手指</th><th style="padding:6px 4px;text-align:left;">标准要求</th><th style="padding:6px 4px;text-align:left;">实际观察</th><th style="padding:6px 4px;text-align:center;">状态</th></tr>';
        fingerComparison.forEach(fc => {
            const icon = statusIcon[fc.status] || "➖";
            const label = statusLabel[fc.status] || fc.status;
            const rowColor = fc.status === "error" ? "var(--danger-light)" : (fc.status === "deviation" ? "var(--warning-light)" : "");
            compareHtml += `<tr style="border-bottom:1px solid var(--card-border);background:${rowColor};">
                <td style="padding:6px 4px;font-weight:600;">${fc.finger || ""}</td>
                <td style="padding:6px 4px;color:var(--text-dim);">${fc.kb_expect || ""}</td>
                <td style="padding:6px 4px;">${fc.actual || ""}</td>
                <td style="padding:6px 4px;text-align:center;">${icon} ${label}</td>
            </tr>`;
            if (fc.note) {
                compareHtml += `<tr style="border-bottom:1px solid var(--card-border);background:${rowColor};"><td></td><td colspan="3" style="padding:2px 4px 6px 4px;font-size:12px;color:var(--text-dim);">💬 ${fc.note}</td></tr>`;
            }
        });
        compareHtml += '</table></div>';
        issuesContainer.innerHTML += compareHtml;
    }

    const summary = data.summary || "";
    if (summary) {
        issuesContainer.innerHTML += `<div style="margin-top:12px;padding:12px;background:var(--card-bg);border-radius:8px;border-left:3px solid var(--primary);">
            <h4 style="margin:0 0 6px 0;">📋 AI 诊断总结</h4>
            <p style="font-size:14px;line-height:1.7;color:var(--text);margin:0;">${summary}</p>
        </div>`;
    }

    const tips = data.practice_tips || [];
    if (tips.length > 0) {
        let tipsHtml = '<div style="margin-top:12px;padding:12px;background:var(--primary-glow);border-radius:8px;">';
        tipsHtml += '<h4 style="margin:0 0 8px 0;color:var(--primary);">🎯 后续练习建议</h4>';
        tips.forEach((tip, idx) => {
            tipsHtml += `<div style="padding:4px 0;font-size:14px;line-height:1.6;">${idx+1}. ${tip}</div>`;
        });
        tipsHtml += "</div>";
        issuesContainer.innerHTML += tipsHtml;
    }
}

function resetChordUI() {
    stopHandProgress(window._handProgressTimer);
    window._handProgressTimer = null;
    if (chordCheckState.mediaRecorder && chordCheckState.mediaRecorder.state === "recording") {
        chordCheckState.mediaRecorder.stop();
    }
    if (chordCheckState.recordingTimerId) {
        clearTimeout(chordCheckState.recordingTimerId);
    }
    if (chordCheckState.cameraStream) {
        chordCheckState.cameraStream.getTracks().forEach(t => t.stop());
        chordCheckState.cameraStream = null;
    }
    if (chordCheckState.recordedUrl) {
        URL.revokeObjectURL(chordCheckState.recordedUrl);
    }
    if (techniqueCheckState.recordedUrl) {
        URL.revokeObjectURL(techniqueCheckState.recordedUrl);
    }
    const savedChord = chordCheckState.selectedChord;
    const savedChordId = chordCheckState.chordId;
    const savedHasKnowledge = chordCheckState.hasKnowledge;
    chordCheckState = {
        selectedChord: savedChord,
        chordId: savedChordId,
        hasKnowledge: savedHasKnowledge,
        recordedBlob: null,
        recordedUrl: null,
        mediaRecorder: null,
        cameraStream: null,
        recordingStartTime: 0,
        recordingTimerId: null,
        maxRecordSec: 5,
    };
    techniqueCheckState = {
        selectedTechnique: techniqueCheckState.selectedTechnique,
        recordedBlob: null,
        recordedUrl: null,
    };
    chordPreview.style.display = "none";
    chordPreview.srcObject = null;
    document.getElementById("chordVideoPlaceholder").style.display = "flex";
    chordVideoReady.style.display = "none";
    analyzeChordBtn.style.display = "none";
    startRecordBtn.style.display = "inline-block";
    startRecordBtn.textContent = "🔴 开始录制";
    startRecordBtn.disabled = false;
    recordTimer.style.display = "none";
    uploadSection.style.display = "block";
    handResultSection.style.display = "none";
    window._lastHandResult = null;
    window._handAnalyzing = false;
    if (chordCheckState.selectedChord || techniqueCheckState.selectedTechnique) {
        startCameraPreview();
    }
}

// ── Hand Check Progress Animation ──

function startHandProgress(bar, msg) {
    var start = Date.now();
    var timer = setInterval(function () {
        var elapsed = (Date.now() - start) / 1000;
        // Simulate progress with diminishing speed
        var progress = Math.min(95, 5 + elapsed * 6 * (1 - elapsed / 25));
        progress = Math.max(progress, 5);
        bar.style.width = progress + "%";

        if (progress < 20) {
            msg.textContent = "正在提取视频关键帧...";
            updateHandStep("hStep1", "active");
        } else if (progress < 50) {
            updateHandStep("hStep1", "done");
            updateHandStep("hStep2", "active");
            msg.textContent = "正在检测手部关键点...";
        } else if (progress < 80) {
            updateHandStep("hStep2", "done");
            updateHandStep("hStep3", "active");
            msg.textContent = "正在分析手型与参考图对比...";
        } else {
            updateHandStep("hStep3", "done");
            updateHandStep("hStep4", "active");
            msg.textContent = "正在生成诊断报告...";
        }
    }, 250);
    return timer;
}

function stopHandProgress(timer) {
    if (timer) clearInterval(timer);
}

function resetHandSteps() {
    ["hStep1","hStep2","hStep3","hStep4"].forEach(function(id) {
        var el = document.getElementById(id);
        if (el) { el.className = "step"; }
    });
    ["hCheck1","hCheck2","hCheck3","hCheck4"].forEach(function(id) {
        var el = document.getElementById(id);
        if (el) { el.textContent = "⌛"; }
    });
}

function updateHandStep(stepId, status) {
    var el = document.getElementById(stepId);
    if (!el) return;
    var num = stepId.replace("hStep", "");
    var checkEl = document.getElementById("hCheck" + num);
    if (status === "active") {
        el.className = "step active";
        if (checkEl) checkEl.textContent = "⌛";
    } else if (status === "done") {
        el.className = "step done";
        if (checkEl) checkEl.textContent = "✅";
    }
}

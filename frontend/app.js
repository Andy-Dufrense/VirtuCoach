/**
 * VirtuCoach - 前端应用逻辑
 */

const API_BASE = window.location.origin;
const POLL_INTERVAL = 1500; // 1.5秒轮询一次，更流畅

let currentTaskId = null;
let pollTimer = null;

// ========== DOM 引用 ==========
const uploadSection = document.getElementById("uploadSection");
const progressSection = document.getElementById("progressSection");
const resultSection = document.getElementById("resultSection");

const uploadBox = document.getElementById("uploadBox");
const videoInput = document.getElementById("videoInput");
const selectBtn = document.getElementById("selectBtn");
const instrumentSelect = document.getElementById("instrumentSelect");
const levelSelect = document.getElementById("levelSelect");
const modelStatus = document.getElementById("modelStatus");
const progressBar = document.getElementById("progressBar");
const progressMessage = document.getElementById("progressMessage");
const progressTitle = document.getElementById("progressTitle");

const toast = document.getElementById("toast");
let toastTimer = null;

// 和弦手型检查 DOM
const modeVideoBtn = document.getElementById("modeVideoBtn");
const modeImageBtn = document.getElementById("modeImageBtn");
const uploadVideoMode = document.getElementById("uploadVideoMode");
const uploadImageMode = document.getElementById("uploadImageMode");
const chordInput = document.getElementById("chordInput");
const chordDatalist = document.getElementById("chordDatalist");
const chordMatchHint = document.getElementById("chordMatchHint");
const chordUploadArea = document.getElementById("chordUploadArea");
const chordRecordInstructions = document.getElementById("chordRecordInstructions");
const chordVideoPreviewWrap = document.getElementById("chordVideoPreviewWrap");
const chordPreview = document.getElementById("chordPreview");
const startRecordBtn = document.getElementById("startRecordBtn");
const recordTimer = document.getElementById("recordTimer");
const chordVideoInput = document.getElementById("chordVideoInput");
const uploadChordVideoBtn = document.getElementById("uploadChordVideoBtn");
const chordVideoReady = document.getElementById("chordVideoReady");
const chordVideoResult = document.getElementById("chordVideoResult");
const analyzeChordBtn = document.getElementById("analyzeChordBtn");
const handResultSection = document.getElementById("handResultSection");

// 技巧检查 DOM
const submodeChordBtn = document.getElementById("submodeChordBtn");
const submodeTechniqueBtn = document.getElementById("submodeTechniqueBtn");
const chordCheckArea = document.getElementById("chordCheckArea");
const techniqueCheckArea = document.getElementById("techniqueCheckArea");
const techniqueSelect = document.getElementById("techniqueSelect");
const techniqueMatchHint = document.getElementById("techniqueMatchHint");
const techniqueRefHint = document.getElementById("techniqueRefHint");
const handCheckTitle = document.getElementById("handCheckTitle");
const handCheckHint = document.getElementById("handCheckHint");

// 视频分析模式录制 DOM
const startVideoRecordBtn = document.getElementById("startVideoRecordBtn");
const videoRecordTimer = document.getElementById("videoRecordTimer");
const videoRecordPreviewWrap = document.getElementById("videoRecordPreviewWrap");
const videoRecordPreview = document.getElementById("videoRecordPreview");
const videoRecordPlaceholder = document.getElementById("videoRecordPlaceholder");
const videoRecordReady = document.getElementById("videoRecordReady");
const videoRecordResult = document.getElementById("videoRecordResult");
let currentMode = "video";

// 硬编码降级和弦列表（与 knowledge/chords/ 保持同步，后端不可用时使用）
const FALLBACK_CHORDS = [
    { id: "C", name: "C和弦", difficulty: "beginner" },
    { id: "Am", name: "Am和弦", difficulty: "beginner" },
    { id: "G", name: "G和弦", difficulty: "beginner" },
    { id: "D", name: "D和弦", difficulty: "beginner" },
    { id: "Em", name: "Em和弦", difficulty: "beginner" },
    { id: "E", name: "E和弦", difficulty: "beginner" },
    { id: "A", name: "A和弦", difficulty: "beginner" },
    { id: "Dm", name: "Dm和弦", difficulty: "beginner" },
    { id: "C7", name: "C7和弦", difficulty: "beginner" },
    { id: "G7", name: "G7和弦", difficulty: "beginner" },
    { id: "E7", name: "E7和弦", difficulty: "beginner" },
    { id: "A7", name: "A7和弦", difficulty: "beginner" },
    { id: "B7", name: "B7和弦", difficulty: "beginner" },
    { id: "D7", name: "D7和弦", difficulty: "beginner" },
    { id: "fmaj7", name: "Fmaj7和弦", difficulty: "beginner" },
    { id: "Dm7", name: "Dm7和弦", difficulty: "beginner" },
    { id: "Bm", name: "Bm和弦", difficulty: "intermediate" },
    { id: "F", name: "F和弦", difficulty: "intermediate" },
    { id: "B", name: "B和弦", difficulty: "intermediate" },
    { id: "D-over-Fsharp", name: "D/F#和弦", difficulty: "intermediate" },
];

// 知识库和弦列表（从后端加载，失败时使用硬编码降级）
let availableChords = FALLBACK_CHORDS.slice();

// 和弦检查状态
let chordCheckState = {
    selectedChord: "",
    chordId: "",  // 对应知识库 id（如 chord-C-major → C-major）
    chordName: "",  // 显示名称（如 "C大三和弦"）
    hasKnowledge: false,
    recordedBlob: null,
    recordedUrl: null,
    mediaRecorder: null,
    cameraStream: null,
    recordingStartTime: 0,
    recordingTimerId: null,
    maxRecordSec: 5,
};

// 手型检查子模式: "chord" | "technique"
let handSubmode = "chord";

// 技巧检查状态
let techniqueCheckState = {
    selectedTechnique: "",
    recordedBlob: null,
    recordedUrl: null,
};

// 视频分析模式录制状态
let videoRecordState = {
    recordedBlob: null,
    recordedUrl: null,
    mediaRecorder: null,
    cameraStream: null,
    recordingStartTime: 0,
    recordingTimerId: null,
    maxRecordSec: 30,
};

// ========== 初始化 ==========
document.addEventListener("DOMContentLoaded", () => {
    checkModelStatus();
    setupEventListeners();
});

function setupEventListeners() {
    try { selectBtn.addEventListener("click", () => videoInput.click()); } catch(e) { console.error("setup: selectBtn", e); }
    try { videoInput.addEventListener("change", handleFileSelect); } catch(e) { console.error("setup: videoInput", e); }

    try {
        uploadBox.addEventListener("dragover", (e) => {
            e.preventDefault();
            uploadBox.classList.add("dragover");
        });
        uploadBox.addEventListener("dragleave", () => {
            uploadBox.classList.remove("dragover");
        });
        uploadBox.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadBox.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                videoInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });
    } catch(e) { console.error("setup: uploadBox", e); }

    try {
        document.querySelectorAll(".tab").forEach(tab => {
            tab.addEventListener("click", () => {
                document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                const tabName = tab.dataset.tab;
                document.getElementById("audioTab").style.display = tabName === "audio" ? "block" : "none";
                document.getElementById("handTab").style.display = tabName === "hand" ? "block" : "none";
            });
        });
    } catch(e) { console.error("setup: tabs", e); }

    try { document.getElementById("newAnalysisBtn").addEventListener("click", resetUI); } catch(e) { console.error("setup: newAnalysisBtn", e); }
    try { document.getElementById("copyReportBtn").addEventListener("click", copyReport); } catch(e) { console.error("setup: copyReportBtn", e); }

    // 模式切换
    try { modeVideoBtn.addEventListener("click", () => switchMode("video")); } catch(e) { console.error("setup: modeVideoBtn", e); }
    try { modeImageBtn.addEventListener("click", () => switchMode("image")); } catch(e) { console.error("setup: modeImageBtn", e); }

    // 和弦检查
    loadChordList();
    try { chordInput.addEventListener("input", onChordInput); } catch(e) { console.error("setup: chordInput input", e); }
    try { chordInput.addEventListener("change", onChordInput); } catch(e) { console.error("setup: chordInput change", e); }
    try { startRecordBtn.addEventListener("click", startChordRecord); } catch(e) { console.error("setup: startRecordBtn", e); }
    try { uploadChordVideoBtn.addEventListener("click", () => chordVideoInput.click()); } catch(e) { console.error("setup: uploadChordVideoBtn", e); }
    try { chordVideoInput.addEventListener("change", handleChordVideoUpload); } catch(e) { console.error("setup: chordVideoInput", e); }
    try { analyzeChordBtn.addEventListener("click", analyzeChordHand); } catch(e) { console.error("setup: analyzeChordBtn", e); }
    try { document.getElementById("newHandAnalysisBtn").addEventListener("click", resetChordUI); } catch(e) { console.error("setup: newHandAnalysisBtn", e); }

    // 子模式切换：和弦检查 / 技巧检查
    try { submodeChordBtn.addEventListener("click", () => switchHandSubmode("chord")); } catch(e) { console.error("setup: submodeChordBtn", e); }
    try { submodeTechniqueBtn.addEventListener("click", () => switchHandSubmode("technique")); } catch(e) { console.error("setup: submodeTechniqueBtn", e); }
    try { techniqueSelect.addEventListener("change", onTechniqueSelect); } catch(e) { console.error("setup: techniqueSelect", e); }

    // 视频分析模式录制
    try { startVideoRecordBtn.addEventListener("click", startVideoRecord); } catch(e) { console.error("setup: startVideoRecordBtn", e); }

    // 图片拖放
    try {
        const imageUploadBox = document.getElementById("imageUploadBox");
        if (imageUploadBox) {
            imageUploadBox.addEventListener("dragover", (e) => { e.preventDefault(); imageUploadBox.classList.add("dragover"); });
            imageUploadBox.addEventListener("dragleave", () => { imageUploadBox.classList.remove("dragover"); });
            imageUploadBox.addEventListener("drop", (e) => {
                e.preventDefault();
                imageUploadBox.classList.remove("dragover");
                if (e.dataTransfer.files.length > 0) {
                    chordVideoInput.files = e.dataTransfer.files;
                    handleChordVideoUpload();
                }
            });
        }
    } catch(e) { console.error("setup: imageUploadBox", e); }
}

// ========== 模型状态检查 ==========
async function checkModelStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/models/status`);
        const data = await resp.json();
        const bpOk = data.basic_pitch;
        const mpOk = data.mediapipe_hands;
        const visOk = data.vision_ai;
        const allOk = bpOk && mpOk;
        if (allOk) {
            var visName = "";
            if (visOk) {
                visName = data.vision_provider === "qwen" ? " + 通义千问 VL" : " + Claude Vision";
            }
            modelStatus.textContent = "✅ 所有模型就绪" + visName;
            modelStatus.className = "status-badge ready";
        } else {
            const missing = [];
            if (!bpOk) missing.push("basic-pitch");
            if (!mpOk) missing.push("MediaPipe");
            modelStatus.textContent = `⚠️ 未加载: ${missing.join(", ")}`;
            modelStatus.className = "status-badge error";
        }
    } catch (e) {
        modelStatus.textContent = "⚠️ 无法连接后端服务";
        modelStatus.className = "status-badge error";
    }
}

// ========== 文件选择 ==========
function handleFileSelect() {
    const file = videoInput.files[0];
    if (!file) return;

    // MIME type 在 Windows 上不可靠，扩展名兜底
    const validTypes = ["video/mp4", "video/quicktime", "video/x-msvideo", ""];
    const validExts = [".mp4", ".mov", ".avi", ".webm", ".mkv"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
        showToast("请选择 MP4 / MOV / AVI / WebM / MKV 格式的视频文件", "error");
        return;
    }
    if (file.size > 500 * 1024 * 1024) {
        showToast("文件过大，请选择小于 500MB 的视频", "error");
        return;
    }
    uploadVideo(file);
}

// ========== 上传视频 ==========
async function uploadVideo(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("instrument", instrumentSelect.value);
    formData.append("level", levelSelect.value);
    formData.append("title", file.name.replace(/\.[^/.]+$/, ""));
    var capoVal = document.getElementById("capoSelect")?.value;
    if (!capoVal || capoVal === "") {
        alert("请选择变调夹位置");
        uploadSection.style.display = "block";
        return;
    }
    formData.append("capo", capoVal);

    uploadSection.style.display = "none";
    progressSection.style.display = "block";
    resultSection.style.display = "none";

    resetProgress();
    progressTitle.textContent = `正在分析: ${file.name}`;
    updateStep("step1", "active");

    // 标记上传进行中，确保切 Tab 回来能恢复进度页
    window._uploadingVideo = true;

    try {
        const resp = await fetch(`${API_BASE}/api/analyze`, {
            method: "POST",
            body: formData,
        });
        if (!resp.ok) throw new Error(`上传失败: ${resp.status}`);
        const data = await resp.json();
        currentTaskId = data.task_id;
        window.currentTaskId = data.task_id;
        window._uploadingVideo = false;
        updateStep("step1", "done");
        updateStep("step2", "active");
        progressMessage.textContent = "正在提取音频并分析音符...";
        startPolling(data.task_id);
    } catch (e) {
        window._uploadingVideo = false;
        showToast(`上传失败: ${e.message}`, "error");
        resetUI();
    }
}

// ========== 轮询 ==========
function startPolling(taskId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/task/${taskId}`);
            const data = await resp.json();

            const p = data.progress || 0;
            progressBar.style.width = `${p}%`;

            // 动态进度消息
            if (p < 5) {
                progressMessage.textContent = "正在提取音频并分析音符...";
            } else if (p < 35) {
                progressMessage.textContent = "音频分析中，检测音符和节奏...";
            } else if (p < 60) {
                progressMessage.textContent = "视频关键帧提取 + 手型检测中...";
            } else if (p < 90) {
                progressMessage.textContent = "AI 综合评估中，生成老师报告...";
            } else {
                progressMessage.textContent = "报告整理中，即将完成...";
            }

            // 动态步骤
            if (p >= 100) {
                updateStep("step2", "done");
                updateStep("step3", "done");
                updateStep("step4", "done");
                progressBar.style.width = "100%";
                progressMessage.textContent = "分析完成！";
            } else if (p >= 60) {
                updateStep("step2", "done");
                updateStep("step3", "done");
                updateStep("step4", "active");
                progressMessage.textContent = "AI 综合评估中，生成老师报告...";
            } else if (p >= 35) {
                updateStep("step2", "done");
                updateStep("step3", "active");
            } else if (p >= 10) {
                updateStep("step2", "active");
            }

            // 更新耗时显示
            if (data.timing) {
                var timingDiv = document.getElementById("progressTiming");
                if (timingDiv) {
                    timingDiv.style.display = "flex";
                    if (data.timing.audio > 0) document.getElementById("audioElapsed").textContent = data.timing.audio + "s";
                    if (data.timing.hand > 0) document.getElementById("handElapsed").textContent = data.timing.hand + "s";
                    if (data.timing.ai > 0) document.getElementById("aiElapsed").textContent = data.timing.ai + "s";
                }
            }
            if (data.status === "completed") {
                clearInterval(pollTimer);
                pollTimer = null;
                showResult(data);
            }
            if (data.status === "failed") {
                clearInterval(pollTimer);
                pollTimer = null;
                showToast(`分析失败: ${data.message}`, "error");
                resetUI();
            }
        } catch (e) {}
    }, POLL_INTERVAL);
}


// ========== 显示结果 ==========
function showResult(data) {
    // 保存结果到全局缓存（始终保存，无论当前模式）
    window._uploadingVideo = false;
    window._hasVideoResult = true;
    window.currentTaskId = data.id;
    window.currentResult = data.result || {};
    window._rawHandIssues = null;
    window._lastVideoResult = data;

    // 如果用户已切换到其他模式，静默保存结果，不切换显示
    if (currentMode !== "video") {
        progressSection.style.display = "none";
        showToast("视频分析完成！切换回「视频分析」查看结果", "success");
        return;
    }

    // 清除旧内容并渲染
    document.getElementById("audioErrorList").innerHTML = "";
    document.getElementById("handIssueList").innerHTML = "";
    document.getElementById("reportContent").innerHTML = "";
    document.getElementById("summaryCard").innerHTML = "";

    progressSection.style.display = "none";
    resultSection.style.display = "block";

    // 把最终耗时也显示出来
    if (data.timing && data.timing.total > 0) {
        var timingDiv = document.getElementById("progressTiming");
        if (timingDiv) {
            timingDiv.style.display = "flex";
            document.getElementById("audioElapsed").textContent = (data.timing.audio || 0) + "s";
            document.getElementById("handElapsed").textContent = (data.timing.hand || 0) + "s";
            document.getElementById("aiElapsed").textContent = (data.timing.ai || 0) + "s";
        }
    }

    var result = data.result || {};
    var score = result.score || {};
    var referenceImages = result.reference_images || [];

    // 无音频等边缘情况，所有分数可能为 null
    if (score.overall != null) {
        animateScore("overallScore", score.overall);
        animateScore("pitchScore", score.pitch != null ? score.pitch : "--");
        animateScore("rhythmScore", score.rhythm != null ? score.rhythm : "--");
    } else {
        document.getElementById("overallScore").textContent = "--";
        document.getElementById("pitchScore").textContent = "--";
        document.getElementById("rhythmScore").textContent = "--";
    }
    if (score.technique != null) {
        animateScore("techniqueScore", score.technique);
    } else {
        document.getElementById("techniqueScore").textContent = "--";
    }

    // 预处理手型数据（必须在摘要徽章之前，否则 realIssues 未定义）
    var handIssues = result.hand_issues || [];
    var flatIssues = [];
    var handSeen = new Set();
    handIssues.forEach(function(h) {
        var who = h["哪只手"] || h["handedness"] || h.handedness || "";
        var iss = h["问题"] || h["issue"] || "";
        var t = h["时间"] || h["timestamp"] || h.timestamp || 0;
        if (Array.isArray(h.issues) || Array.isArray(h["issues"])) {
            var subList = Array.isArray(h.issues) ? h.issues : h["issues"];
            subList.forEach(function(subIss) {
                var key = who + "|" + subIss;
                if (!handSeen.has(key)) { handSeen.add(key); flatIssues.push({who:who, issue:subIss, time:t, snapUrl: h["截图"] || ""}); }
            });
        } else if (iss) {
            var key = who + "|" + iss;
            if (!handSeen.has(key)) { handSeen.add(key); flatIssues.push({who:who, issue:iss, time:t, snapUrl: h["截图"] || ""}); }
        }
    });
    window._rawHandIssues = handIssues;
    var realIssues = flatIssues.filter(function(u) {
        return u.issue.indexOf("正常") < 0 && u.issue.indexOf("无问题") < 0;
    });

    // 摘要（含问题计数徽章）
    var audioErrCount = (result.audio_errors || []).length;
    var handIssCount = realIssues ? realIssues.length : 0;
    var summaryText = result.summary || "分析完成，请查看下方详细报告。";
    document.getElementById("summaryCard").innerHTML = "💡 " + summaryText;

    // ===== 检测详情 =====
    var severityMap = {
        "wrong_note": "critical", "extra_note": "critical", "missed_note": "critical",
        "rhythm_fault": "minor", "overlap": "cosmetic", "pause": "cosmetic"
    };
    // 后端返回的 severity 映射到前端样式
    var sevLevelMap = { "high": "critical", "medium": "minor", "low": "cosmetic" };
    var severityCn = { "critical": "重要错误", "minor": "轻微错误", "cosmetic": "小问题" };

    // 音频错误
    var audioErrors = result.audio_errors || [];
    var audioStatus = result.audio_status || "";
    var audioList = document.getElementById("audioErrorList");

    // 内容类型提示：单音旋律 → 说明跳过了和弦分析
    var contentType = result.content_type || "";
    var contentTypeNote = "";
    if (contentType === "melody") {
        contentTypeNote = '<li style="border-left-color:var(--primary);background:rgba(9,132,227,0.05);">🎵 检测到<b>单音旋律</b>，已自动跳过和弦识别分析（和弦识别仅适用于同时弹奏多音的场景）</li>';
    }

    if (audioStatus === "nSound") {
        audioList.innerHTML = contentTypeNote + '<li>🔇 未检测到有效音频信号，请确认录制的视频包含清晰的乐器声音</li>';
    } else if (audioErrors.length > 0) {
        audioList.innerHTML = contentTypeNote + audioErrors.map(function(e) {
            // 优先用后端返回的 severity，其次用类型映射
            var sev = sevLevelMap[e.severity] || severityMap[e.type] || "cosmetic";
            var icon = e.type === "wrong_note" ? "🎵" :
                       e.type === "rhythm_fault" ? "⏱" :
                       e.type === "extra_note" ? "➕" :
                       e.type === "missed_note" ? "➖" :
                       e.type === "overlap" ? "🔀" :
                       e.type === "pause" ? "⏸" : "🔍";
            var detail = e.detail || '';
            // 节奏错误标记偏快/偏慢方向
            if (e.type === 'rhythm_fault' && detail) {
                if (/偏快|抢拍|往前赶|快了/.test(detail)) detail = '🏃 偏快 — ' + detail;
                else if (/偏慢|拖拍|慢了/.test(detail)) detail = '🐢 偏慢 — ' + detail;
                else if (/不均匀|不稳/.test(detail)) detail = '📊 ' + detail;
            }
            return '<li class="severity-' + sev + '">' +
                icon + ' <b>[' + (severityCn[sev] || '') + ']</b> ' + detail +
                '<span class="err-time">(' + formatTime(e.time) + ')</span></li>';
        }).join("");
    } else {
        audioList.innerHTML = contentTypeNote + '<li style="border-left-color:var(--success)">✅ 未检测到明显音频问题</li>';
    }

    // 手型问题渲染（数据预处理已在摘要徽章前完成）
    var handList = document.getElementById("handIssueList");
    var handCompareSection = document.getElementById("handCompareSection");

    if (handIssues.length > 0) {
        if (realIssues.length > 0) {
            // 按时点分组：同一秒内的问题合并为一条
            var timeGroups = {};
            realIssues.forEach(function(item) {
                var tKey = Math.round(item.time);
                if (!timeGroups[tKey]) timeGroups[tKey] = { who: item.who, time: item.time, issues: [] };
                timeGroups[tKey].issues.push(item.issue);
            });
            // 按时点升序排列
            var sortedTimes = Object.keys(timeGroups).map(Number).sort(function(a, b) { return a - b; });
            handList.innerHTML = sortedTimes.map(function(tKey) {
                var g = timeGroups[tKey];
                // 同一时点的问题合并，去重
                var merged = g.issues.filter(function(v, i, a) { return a.indexOf(v) === i; }).join("；");
                var sev = "critical";
                if (merged.indexOf("偏高") >= 0 || merged.indexOf("偏低") >= 0 || merged.indexOf("角度") >= 0) sev = "minor";
                if (merged.indexOf("紧张") >= 0 || merged.indexOf("略微") >= 0 || merged.indexOf("太高") >= 0) sev = "cosmetic";
                return '<li class="severity-' + sev + '">' +
                    '✋ <b>' + (g.who || '未知') + '</b> (' + formatTime(g.time) + ')<br>' +
                    merged + '</li>';
            }).join("");
            // 显示正确手型对比
            renderHandCompare(realIssues, referenceImages);
        } else {
            handList.innerHTML = '<li style="border-left-color:var(--success)">✅ 手型正常</li>';
            if (handCompareSection) handCompareSection.style.display = "none";
        }
    } else {
        var hs = result.hand_status || "";
        if (hs === "nHand") {
            handList.innerHTML = '<li>📷 本次未捕捉到手部画面，下次录制时请确保左右双手都在镜头内清晰可见</li>';
        } else if (hs === "ok") {
            handList.innerHTML = '<li style="border-left-color:var(--success)">✅ 手型正常</li>';
        } else {
            // hand_status 未知或缺失 → 诚实展示
            handList.innerHTML = '<li>⚠️ 手型检测状态未知，请确认手部在镜头内清晰可见</li>';
        }
        if (handCompareSection) handCompareSection.style.display = "none";
    }

    // ===== 截图嵌入 AI 报告中 =====
    var snapshots = result.snapshots || [];
    var reportContent = document.getElementById("reportContent");
    if (result.report_markdown) {
        var reportHtml = renderMarkdown(result.report_markdown);
        reportContent.innerHTML = reportHtml;
    } else {
        reportContent.innerHTML = "<p>报告生成中...</p>";
    }

    // ===== 截图对比 ─ 独立区域（仅当有手型问题和截图时显示） =====
    // （正确手型对比放在检测详情的手型tab里，这里放截图对比）
    var snapInline = document.getElementById("snapshotInline");
    if (snapInline) {
        snapInline.style.display = "none"; // 截图已嵌入报告，独立区域隐藏
    }

    var practiceTips = result.practice_tips || [];
    var oldPractice = document.getElementById("practiceCard");
    if (oldPractice) oldPractice.remove();
    if (practiceTips.length > 0) {
        var tipsHtml = '<div class="practice-card" id="practiceCard"><h3>后续练的方向</h3><ul>';
        practiceTips.forEach(function(tip) {
            tipsHtml += "<li>" + tip + "</li>";
        });
        tipsHtml += "</ul></div>";
        var actionsDiv = document.querySelector(".actions");
        if (actionsDiv) {
            actionsDiv.insertAdjacentHTML("beforebegin", tipsHtml);
        }
    }

    // 追问面板
    try { renderChatPanel(); showChatPanel(); } catch(e) {}
}

// ========== 正确手型对比渲染（以后端 referenceImages 分组为准）==========

function renderHandCompare(realIssues, referenceImages) {
    var section = document.getElementById("handCompareSection");
    var grid = document.getElementById("handCompareGrid");
    if (!section || !grid) return;

    var refs = referenceImages || [];
    if (refs.length === 0) {
        section.style.display = "none";
        return;
    }
    section.style.display = "block";

    // 预建 rawIssues 索引 (who|time→截图URL)，避免每次 O(n) 扫描
    var rawIssueSnaps = {};
    var rawIssues = window._rawHandIssues || [];
    rawIssues.forEach(function(rh) {
        var key = (rh["哪只手"] || "") + "|" + (Math.round((rh["时间"] || rh["timestamp"] || 0) * 10) / 10);
        if (rh["截图"]) rawIssueSnaps[key] = rh["截图"];
    });

    var techniqueMap = {
        "basic_fretting": "基本按弦", "natural_harmonics": "自然泛音", "pm": "PM闷音",
        "am": "AM指弹", "tapping": "点弦", "muting": "左手闷音", "bend": "推弦",
        "slide": "滑音", "vibrato": "揉弦", "hammer_on": "击弦", "pull_off": "勾弦"
    };

    var cardsHtml = "";

    refs.forEach(function(ref) {
        var gtimes = ref.group_times || [];
        var refTime = ref.issue_time || 0;

        // 匹配问题
        var matchedIssues = realIssues.filter(function(item) {
            if (item.issue.indexOf('💡 AI 观察') >= 0) return false;
            for (var gi = 0; gi < gtimes.length; gi++) {
                if (Math.abs(gtimes[gi] - item.time) < 0.01) return true;
            }
            if (Math.abs(refTime - item.time) < 0.01) return true;
            return false;
        });

        if (matchedIssues.length === 0) {
            matchedIssues = [{ issue: "未定位到具体问题", time: refTime, who: "", snapUrl: "" }];
        }

        // 收集用户截图（同0.1s去重，用预建索引O(1)查找）
        var userSnaps = [];
        var shownTimes = {};
        matchedIssues.forEach(function(item) {
            var tKey = Math.round(item.time * 10) / 10;
            if (shownTimes[tKey]) return;
            shownTimes[tKey] = true;

            var snapUrl = item.snapUrl || "";
            if (!snapUrl) {
                var idxKey = (item.who || "") + "|" + tKey;
                snapUrl = rawIssueSnaps[idxKey] || "";
            }
            // 索引兜底：同时间任意截图
            if (!snapUrl) {
                for (var ri = 0; ri < rawIssues.length; ri++) {
                    var rt = rawIssues[ri]["时间"] || rawIssues[ri]["timestamp"] || 0;
                    if (Math.abs(rt - item.time) < 0.01 && (rawIssues[ri]["截图"] || "")) {
                        snapUrl = rawIssues[ri]["截图"];
                        break;
                    }
                }
            }
            if (snapUrl) {
                userSnaps.push({ snapUrl: snapUrl, time: item.time });
            }
        });

        // 参考图渲染：直接用后端字段
        var refUrl = API_BASE + ref.image_url;
        var ch = ref.channel || "";

        // 判断参考图类型：wrist/thumb vs finger
        // channel 格式如 "wrist:collapsed; thumb:too-high" 或 "finger:too-curved"
        var isWristRef = ch.indexOf("wrist") >= 0 || ch.indexOf("thumb") >= 0;

        // 合并问题文本：wrist/thumb 参考图只显示手腕拇指问题，
        // finger 参考图只显示手指问题，避免两个 div 显示相同内容
        var allIssuesText = matchedIssues.map(function(iss) { return iss.issue; })
            .filter(function(v, i, a) { return a.indexOf(v) === i; })
            .filter(function(txt) {
                var hasWrist = txt.indexOf("手腕") >= 0;
                var hasThumb = txt.indexOf("拇指") >= 0;
                if (isWristRef) return hasWrist || hasThumb;
                else return !hasWrist && !hasThumb;
            }).join("；");
        if (!allIssuesText) allIssuesText = isWristRef ? "手腕与拇指问题" : "手指问题";
        var issueWho = matchedIssues[0].who || "";

        var chordLabel = "";
        // 手腕参考图标签显示"手腕与大拇指"，手指参考图显示和弦名
        if (isWristRef) {
            chordLabel = " · 手腕与大拇指";
        } else if (ref.chord_name) {
            chordLabel = " · " + ref.chord_name;
        }
        var tLabel = techniqueMap[ref.technique] || ref.technique || "";
        var viewHint = "";
        if (ref.is_topdown) viewHint = "俯视";
        else if (ref.is_front) viewHint = "正视";
        var labelText = (viewHint || tLabel || "参考手型") + chordLabel;
        if (ref.chord_matched === false) {
            labelText += "（通用参考）";
        }

        var refHtml = '<div class="compare-side">' +
            '<span class="compare-label">✅ 正确手型' + chordLabel + '</span>' +
            '<a href="' + refUrl + '" target="_blank" class="snap-link">' +
            '<img src="' + refUrl + '" alt="正确手型参考" loading="lazy">' +
            '</a></div>';

        var userHtml = "";
        if (userSnaps.length > 0) {
            var firstSnap = userSnaps[0];
            var timeLabel = matchedIssues.length === 1
                ? formatTime(matchedIssues[0].time)
                : formatTime(matchedIssues[0].time) + " 等" + matchedIssues.length + "处";
            userHtml = '<div class="compare-side">' +
                '<span class="compare-label">❌ 你的手型 (' + (issueWho || '') + ' · ' + timeLabel + ')</span>' +
                '<a href="' + firstSnap.snapUrl + '" target="_blank" class="snap-link">' +
                '<img src="' + firstSnap.snapUrl + '" alt="你的手型" loading="lazy">' +
                '</a><span class="compare-desc">' + allIssuesText + '</span></div>';
        } else {
            userHtml = '<div class="compare-side">' +
                '<span class="compare-label">❌ 你的手型 (' + (issueWho || '') + ')</span>' +
                '<div class="compare-placeholder"><span>🖐️</span><br>未匹配到<br>对应截图</div></div>';
        }

        cardsHtml += '<div class="hand-compare-card">' +
            '<div class="compare-row">' + userHtml + refHtml + '</div></div>';
    });

    grid.innerHTML = cardsHtml;
}// ========== 追问 Agent ==========
let resultCache = null;

function renderChatPanel() {
    const existing = document.getElementById("chatCard");
    if (existing) existing.remove();

    const chatHtml = `
        <div class="chat-card" id="chatCard">
            <div class="chat-header">
                <h3>💬 追问 AI 老师</h3>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="chat-msg ai">
                    <div class="msg-bubble">你好！对报告有什么想问的吗？</div>
                </div>
            </div>
            <div class="chat-input-area">
                <input type="text" id="chatInput" placeholder="输入你的问题..." class="chat-input">
                <button id="chatSend" class="btn btn-primary btn-small">发送</button>
            </div>
        </div>
    `;
    document.querySelector('.actions').insertAdjacentHTML('afterend', chatHtml);

    document.getElementById('chatSend').addEventListener('click', sendQuestion);
    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendQuestion();
    });
}

function showChatPanel() {
    const el = document.getElementById("chatCard");
    if (el) el.style.display = "block";
}

async function sendQuestion() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question) return;

    const msgs = document.getElementById('chatMessages');
    msgs.innerHTML += `<div class="chat-msg user"><div class="msg-bubble">${escapeHtml(question)}</div></div>`;
    input.value = '';
    msgs.scrollTop = msgs.scrollHeight;

    document.getElementById('chatSend').disabled = true;
    // 创建流式消息气泡，用于逐字追加内容
    const streamId = 'streamMsg_' + Date.now();
    msgs.innerHTML += `<div class="chat-msg ai" id="${streamId}"><div class="msg-bubble"></div></div>`;

    const result = window.currentResult || {};
    try {
        const resp = await fetch(`${API_BASE}/api/ask/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: window.currentTaskId, question, context: result })
        });
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let buffer = '';
        const bubble = document.querySelector(`#${streamId} .msg-bubble`);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // 解析 SSE 事件
            const lines = buffer.split('\n');
            buffer = '';
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.chunk) {
                            fullText += data.chunk;
                            bubble.innerHTML = renderMarkdown(fullText);
                            msgs.scrollTop = msgs.scrollHeight;
                        }
                    } catch (e) {
                        // 不完整的 JSON，放回 buffer
                        buffer = line + '\n';
                    }
                }
            }
        }
        if (fullText) {
            bubble.innerHTML = renderMarkdown(fullText);
        } else {
            bubble.innerHTML = renderMarkdown('抱歉，老师暂时无法回答。');
        }
    } catch (e) {
        document.getElementById(streamId).querySelector('.msg-bubble').innerHTML = '⚠️ 网络开小差了，稍后再问吧！';
    } finally {
        document.getElementById('chatSend').disabled = false;
        msgs.scrollTop = msgs.scrollHeight;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========== 工具函数 ==========
function resetProgress() {
    progressBar.style.width = "0%";
    progressMessage.textContent = "准备中...";
    ["step1", "step2", "step3", "step4"].forEach(id => {
        document.getElementById(id).className = "step";
        document.getElementById(id.replace("step", "check")).textContent = "⌛";
    });
    // 清空上轮耗时
    var timingDiv = document.getElementById("progressTiming");
    if (timingDiv) timingDiv.style.display = "none";
    ["audioElapsed", "handElapsed", "aiElapsed"].forEach(function(id) {
        var el = document.getElementById(id);
        if (el) el.textContent = "--";
    });
}

function updateStep(stepId, status) {
    const step = document.getElementById(stepId);
    const check = document.getElementById(stepId.replace("step", "check"));
    step.className = "step";
    if (status === "active") {
        step.classList.add("active");
        check.textContent = "⌛";
    } else if (status === "done") {
        step.classList.add("done");
        check.textContent = "✅";
    }
}

function animateScore(elementId, target) {
    const el = document.getElementById(elementId);
    if (target === "--" || target == null) {
        el.textContent = "--";
        return;
    }
    let current = 0;
    const stepSize = Math.ceil(target / 25);
    const timer = setInterval(() => {
        current += stepSize;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        el.textContent = Math.round(current);
    }, 40);
}

function formatTime(seconds) {
    if (seconds === undefined || seconds === null) return "--";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}

function renderMarkdown(md) {
    if (!md) return "";
    var lines = md.split("\n");
    var html = "";
    var inList = false, inQuote = false;

    function closeList() { if (inList) { html += "</ul>"; inList = false; } }
    function closeQuote() { if (inQuote) { html += "</blockquote>"; inQuote = false; } }

    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        var trimmed = line.trim();
        if (!trimmed) { closeList(); closeQuote(); continue; }

        // 标题
        if (trimmed.match(/^## /)) {
            closeList(); closeQuote();
            html += "<h2>" + formatInline(trimmed.replace(/^## /, "")) + "</h2>";
        }
        else if (trimmed.match(/^### /)) {
            closeList(); closeQuote();
            html += "<h3>" + formatInline(trimmed.replace(/^### /, "")) + "</h3>";
        }
        // 引用块
        else if (trimmed.match(/^> /)) {
            closeList();
            if (!inQuote) {
                var quoteContent = trimmed.replace(/^> /, "");
                var nextLine = (lines[i+1] || "").trim();
                var fullQuote = quoteContent;
                // Peek ahead to classify
                var j = i + 1;
                while (j < lines.length && (lines[j].trim().match(/^> /))) {
                    fullQuote += " " + lines[j].trim().replace(/^> /, "");
                    j++;
                }
                var isIssue = fullQuote.indexOf("⚠️") >= 0 || fullQuote.indexOf("⚠") >= 0;
                var isGood = fullQuote.indexOf("✅") >= 0;
                var cls = isIssue ? "issue-quote" : (isGood ? "good-quote" : "");
                html += "<blockquote class=\"" + cls + "\">";
                inQuote = true;
            }
            html += formatInline(trimmed.replace(/^> /, ""));
            var nextTrimmed = (lines[i+1] || "").trim();
            if (nextTrimmed.match(/^> /)) html += "<br>";
        }
        // 列表
        else if (trimmed.match(/^- /)) {
            closeQuote();
            if (!inList) { html += "<ul>"; inList = true; }
            html += "<li>" + formatInline(trimmed.replace(/^- /, "")) + "</li>";
        }
        // 分割线
        else if (trimmed.match(/^---+/)) {
            closeList(); closeQuote();
            html += "<hr>";
        }
        // 普通段落
        else {
            closeList(); closeQuote();
            html += "<p>" + formatInline(trimmed) + "</p>";
        }
    }
    closeList(); closeQuote();
    return html;
}

function formatInline(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/✅/g, '<span style="color:var(--success)">✅</span>')
        .replace(/⚠️/g, '<span style="color:var(--warning)">⚠️</span>')
        .replace(/📷/g, '<span style="color:var(--text-dim)">📷</span>');
}

// ========== 模式切换 ==========

function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    document.getElementById(mode === "video" ? "modeVideoBtn" : "modeImageBtn").classList.add("active");

    // 检测是否有后台运行的分析任务
    var videoUploading = !!window._uploadingVideo;
    var videoAnalyzing = !!pollTimer;
    var videoHasResult = !!window._hasVideoResult;
    var imageAnalyzing = !!window._handAnalyzing;

    // 隐藏所有 section
    uploadSection.style.display = "none";
    progressSection.style.display = "none";
    resultSection.style.display = "none";
    // 手型检查有结果或正在分析时不隐藏，保持状态
    if (!(mode === "image" && (window._lastHandResult || imageAnalyzing))) {
        handResultSection.style.display = "none";
    }

    if (mode === "video") {
        uploadVideoMode.style.display = "block";
        uploadImageMode.style.display = "none";
        if (videoHasResult && window._lastVideoResult) {
            showResult(window._lastVideoResult);
        } else if (videoAnalyzing || videoUploading) {
            progressSection.style.display = "block";
        } else {
            uploadSection.style.display = "block";
        }
    } else if (mode === "image") {
        uploadVideoMode.style.display = "none";
        uploadImageMode.style.display = "block";
        if (window._lastHandResult) {
            // 有已保存的结果 → 直接渲染
            renderChordResult(window._lastHandResult);
        } else if (imageAnalyzing) {
            // 分析进行中 → 显示进度状态，保持结果区域可见
            uploadSection.style.display = "none";
            handResultSection.style.display = "block";
            document.getElementById("handStatusBadge") && (document.getElementById("handStatusBadge").textContent = "分析中...");
            document.getElementById("handScore") && (document.getElementById("handScore").textContent = "...");
        } else {
            uploadSection.style.display = "block";
        }
    }
}

function copyReport() {
    const content = document.getElementById("reportContent").innerText;
    navigator.clipboard.writeText(content).then(() => {
        showToast("报告已复制到剪贴板", "success");
    }).catch(() => {
        showToast("复制失败", "error");
    });
}

function resetUI() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
    currentTaskId = null;
    window.currentTaskId = null;
    window.currentResult = null;
    window._rawHandIssues = null;
    window._lastVideoResult = null;
    window._uploadingVideo = false;
    window._hasVideoResult = false;
    videoInput.value = "";
    uploadSection.style.display = "block";
    progressSection.style.display = "none";
    resultSection.style.display = "none";
    handResultSection.style.display = "none";
    resetProgress();
    resetVideoRecord();
}

function showToast(message, type = "") {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add("show");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}

// ═══════════════════════════════════════════
// Feedback Widget
// ═══════════════════════════════════════════

(function() {
    const fab = document.getElementById('feedbackFab');
    const modal = document.getElementById('feedbackModal');
    const closeBtn = document.getElementById('feedbackClose');
    const form = document.getElementById('feedbackForm');
    const dropZone = document.getElementById('fbDropZone');
    const fileInput = document.getElementById('fbFileInput');
    const preview = document.getElementById('fbScreenshotPreview');

    if (!fab || !modal) return;

    fab.addEventListener('click', () => modal.classList.add('active'));
    closeBtn.addEventListener('click', () => modal.classList.remove('active'));
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });

    // Screenshot: click / drag / paste
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => fbShowPreview(fileInput.files[0]));

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const f = e.dataTransfer.files[0];
        if (f) {
            const dt = new DataTransfer();
            dt.items.add(f);
            fileInput.files = dt.files;
            fbShowPreview(f);
        }
    });

    document.addEventListener('paste', e => {
        if (!modal.classList.contains('active')) return;
        if (!e.clipboardData) return;
        const items = e.clipboardData.items;
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const blob = item.getAsFile();
                const dt = new DataTransfer();
                dt.items.add(new File([blob], 'paste.png', { type: blob.type }));
                fileInput.files = dt.files;
                fbShowPreview(blob);
                break;
            }
        }
    });

    function fbShowPreview(file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<div class="preview-wrap">
                <img src="${e.target.result}" alt="截图预览">
                <button class="remove-btn" onclick="document.getElementById('fbScreenshotPreview').innerHTML='';document.getElementById('fbFileInput').value='';">×</button>
            </div>`;
        };
        reader.readAsDataURL(file);
    }

    form.addEventListener('submit', async e => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = '提交中...';
        submitBtn.disabled = true;

        const fd = new FormData(form);
        if (fileInput.files[0]) fd.set('screenshot', fileInput.files[0]);
        fd.set('browser_info', navigator.userAgent);

        try {
            const res = await fetch('/api/feedbacks', { method: 'POST', body: fd });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || '提交失败');
            }
            showToast('反馈提交成功！', 'success');
            form.reset();
            preview.innerHTML = '';
            fileInput.value = '';
            modal.classList.remove('active');
        } catch (err) {
            showToast(err.message || '提交失败，请重试', 'error');
        } finally {
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        }
    });

    // Esc to close
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            modal.classList.remove('active');
        }
    });
})();

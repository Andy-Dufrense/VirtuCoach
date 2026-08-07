/**
 * VirtuCoach - 参考图管理器模块
 * 参考图的上传、AI分析、保存、浏览和删除
 * 依赖: app.js (DOM 引用、全局状态、showToast、API_BASE)
 */

// ========== 参考图管理器 ==========
var refSelectedFile = null;

function initRefManager() {
    var toggleBtn = document.getElementById("refManagerBtn");
    var panel = document.getElementById("refManagerSection");
    var closeBtn = document.getElementById("refManagerClose");
    var selectBtn = document.getElementById("refSelectBtn");
    var imageInput = document.getElementById("refImageInput");
    var analyzeBtn = document.getElementById("refAnalyzeBtn");
    var saveBtn = document.getElementById("refSaveBtn");

    if (!toggleBtn) return;

    // 只对管理员显示按钮（URL 带 ?admin 参数）
    var isAdmin = window.location.search.indexOf("admin") >= 0;
    if (isAdmin) {
        toggleBtn.style.display = "";
    } else {
        toggleBtn.style.display = "none";
    }

    toggleBtn.addEventListener("click", function() {
        if (panel.style.display === "none" || !panel.style.display) {
            panel.style.display = "block";
            document.getElementById("uploadSection").style.display = "none";
            document.getElementById("progressSection").style.display = "none";
            document.getElementById("resultSection").style.display = "none";
            loadRefGallery();
        } else {
            panel.style.display = "none";
        }
    });

    closeBtn.addEventListener("click", function() {
        panel.style.display = "none";
    });

    selectBtn.addEventListener("click", function() { imageInput.click(); });

    imageInput.addEventListener("change", function() {
        refSelectedFile = imageInput.files[0];
        if (refSelectedFile) {
            var reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById("refPreviewBox").innerHTML =
                    '<img src="' + e.target.result + '" alt="预览" style="max-width:100%;max-height:200px;border-radius:8px;">';
            };
            reader.readAsDataURL(refSelectedFile);
            analyzeBtn.disabled = false;
            saveBtn.disabled = true;
            // 清空之前的分析结果
            document.getElementById("refDescription").value = "";
            document.getElementById("refTags").value = "";
            document.getElementById("refHand").value = "双手";
            document.getElementById("refTechnique").value = "basic_fretting";
            document.getElementById("refBodyPart").value = "full_hand";
        }
    });

    // AI 分析按钮
    analyzeBtn.addEventListener("click", function() {
        if (!refSelectedFile) return;
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "⏳ 分析中...";

        var fd = new FormData();
        fd.append("image", refSelectedFile);

        fetch(API_BASE + "/api/references/analyze", { method: "POST", body: fd })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.error) { showToast(data.error, "error"); return; }
                if (!data.suggested) { showToast("AI 分析失败，请手动填写", "error"); return; }
                var s = data.suggested;
                document.getElementById("refHand").value = s.hand || "双手";
                document.getElementById("refTechnique").value = s.technique || "basic_fretting";
                document.getElementById("refBodyPart").value = s.body_part || "full_hand";
                document.getElementById("refView").value = s.view_direction || "前";
                document.getElementById("refDescription").value = s.description || "";
                document.getElementById("refTags").value = Array.isArray(s.tags) ? s.tags.join(",") : (s.tags || "");
                saveBtn.disabled = false;
                showToast("AI 分析完成，请确认后保存", "success");
            })
            .catch(function() { showToast("分析失败，请重试", "error"); })
            .finally(function() {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = "🤖 AI 分析";
            });
    });

    // 保存按钮
    saveBtn.addEventListener("click", function() {
        if (!refSelectedFile) return;
        var desc = document.getElementById("refDescription").value.trim();
        if (!desc) { showToast("请填写描述", "error"); return; }

        saveBtn.disabled = true;
        saveBtn.textContent = "⏳ 保存中...";

        var fd = new FormData();
        fd.append("image", refSelectedFile);
        fd.append("hand", document.getElementById("refHand").value);
        fd.append("technique", document.getElementById("refTechnique").value);
        fd.append("body_part", document.getElementById("refBodyPart").value);
        fd.append("view_direction", document.getElementById("refView").value);
        fd.append("description", desc);
        fd.append("tags", document.getElementById("refTags").value);

        fetch(API_BASE + "/api/references", { method: "POST", body: fd })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.id) {
                    showToast("参考图已保存！", "success");
                    // 重置
                    refSelectedFile = null;
                    document.getElementById("refImageInput").value = "";
                    document.getElementById("refPreviewBox").innerHTML = '<span>🖼️</span><p>选择图片</p>';
                    document.getElementById("refDescription").value = "";
                    document.getElementById("refTags").value = "";
                    document.getElementById("refAnalyzeBtn").disabled = true;
                    saveBtn.disabled = true;
                    loadRefGallery();
                }
            })
            .catch(function() { showToast("保存失败", "error"); })
            .finally(function() {
                saveBtn.disabled = false;
                saveBtn.textContent = "💾 保存参考图";
            });
    });
}

function loadRefGallery() {
    var grid = document.getElementById("refGalleryGrid");
    var countEl = document.getElementById("refCount");
    if (!grid) return;

    grid.innerHTML = '<p style="color:var(--text-dim);">加载中...</p>';

    fetch(API_BASE + "/api/references")
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var refs = data.references || [];
            countEl.textContent = "(" + refs.length + " 张)";
            if (refs.length === 0) {
                grid.innerHTML = '<p style="color:var(--text-dim);">还没有参考图，请上传</p>';
                return;
            }
            grid.innerHTML = refs.map(function(ref) {
                var techniqueMap = {
                    "basic_fretting":"基本按弦","natural_harmonics":"自然泛音","pm":"PM闷音","am":"AM指弹",
                    "tapping":"点弦","muting":"左手闷音","bend":"推弦","slide":"滑音","vibrato":"揉弦",
                    "hammer_on":"击弦","pull_off":"勾弦"
                };
                var tLabel = techniqueMap[ref.technique] || ref.technique;
                return '<div class="ref-gallery-item">' +
                    '<img src="' + API_BASE + ref.image_url + '" alt="' + (ref.description||'') + '" loading="lazy">' +
                    '<div class="ref-gallery-info">' +
                    '<span class="ref-gallery-tag">' + (tLabel||'') + '</span>' +
                    '<span class="ref-gallery-tag">' + (ref.body_part||'') + '</span>' +
                    '<span class="ref-gallery-tag">' + (ref.hand||'') + '</span>' +
                    '</div>' +
                    '<p style="font-size:12px;margin:4px 0;color:var(--text-dim);">' + (ref.description||'') + '</p>' +
                    '<button class="btn btn-small btn-danger ref-delete-btn" data-id="' + ref.id + '">删除</button>' +
                    '</div>';
            }).join("");

            // 绑定删除事件
            grid.querySelectorAll(".ref-delete-btn").forEach(function(btn) {
                btn.addEventListener("click", function() {
                    var rid = this.dataset.id;
                    if (!confirm("确定删除这张参考图吗？")) return;
                    fetch(API_BASE + "/api/references/" + rid, { method: "DELETE" })
                        .then(function() { loadRefGallery(); })
                        .catch(function() { showToast("删除失败", "error"); });
                });
            });
        })
        .catch(function() { grid.innerHTML = '<p style="color:red;">加载失败</p>'; });
}

// 在页面加载时初始化
document.addEventListener("DOMContentLoaded", function() {
    initRefManager();
});

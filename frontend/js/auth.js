/**
 * VirtuCoach shared auth module.
 * Token stored in sessionStorage key: virtucoach_token
 */
(function () {
  var AUTH_KEY = "virtucoach_token";
  var USER_KEY = "virtucoach_user";

  var VC = (window.VirtuCoach = window.VirtuCoach || {});

  VC.getToken = function () {
    return sessionStorage.getItem(AUTH_KEY);
  };

  VC.setToken = function (token) {
    sessionStorage.setItem(AUTH_KEY, token);
  };

  VC.clearToken = function () {
    sessionStorage.removeItem(AUTH_KEY);
    sessionStorage.removeItem(USER_KEY);
  };

  VC.isLoggedIn = function () {
    return !!sessionStorage.getItem(AUTH_KEY);
  };

  VC.getAuthHeaders = function () {
    var token = sessionStorage.getItem(AUTH_KEY);
    if (token) return { Authorization: "Bearer " + token };
    return {};
  };

  var HEARTBEAT_MS = 30000;
  var heartbeatTimer = null;

  VC.startHeartbeat = function () {
    VC.stopHeartbeat();
    function beat() {
      var token = VC.getToken();
      if (!token) { VC.stopHeartbeat(); return; }
      fetch("/api/auth/heartbeat", {
        method: "POST",
        headers: { Authorization: "Bearer " + token },
        keepalive: true,
      }).then(function (r) {
        if (r.status === 401) VC.sessionExpired();
      }).catch(function () {});
    }
    beat(); // 立即标记在线，避免刷新/导航后出现 online=0 的空窗期
    heartbeatTimer = setInterval(beat, HEARTBEAT_MS);
  };

  VC.stopHeartbeat = function () {
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  };

  /** 页面关闭/刷新信标：仅释放 online 占用，不使 token 失效 */
  VC.markOffline = function () {
    var token = VC.getToken();
    if (!token) return;
    try {
      fetch("/api/auth/offline", {
        method: "POST",
        headers: { Authorization: "Bearer " + token },
        keepalive: true,
      });
    } catch (e) {}
  };

  VC.logout = async function () {
    VC.stopHeartbeat();
    var token = VC.getToken();
    if (token) {
      // 先等后端释放单会话占用（清 last_activity + online=0），再跳转，避免退出后仍被判定"已在其他设备登录"
      try {
        await fetch("/api/auth/logout", {
          method: "POST",
          headers: { Authorization: "Bearer " + token },
          keepalive: true,
        });
      } catch (e) {}
    }
    VC.clearToken();
    window.location.href = "/login.html";
  };

  /** 登录态失效：清掉本地 token 并回登录页（避免过期 token 静默降级为游客） */
  VC.sessionExpired = function () {
    VC.stopHeartbeat();
    VC.clearToken();
    if (!/login\.html$/.test(window.location.pathname)) {
      window.location.href = "/login.html";
    }
  };

  VC.fetchUser = async function () {
    var token = VC.getToken();
    if (!token) return null;
    try {
      var resp = await fetch("/api/auth/me", {
        headers: { Authorization: "Bearer " + token },
      });
      if (!resp.ok) return null;
      var data = await resp.json();
      sessionStorage.setItem(USER_KEY, JSON.stringify(data));
      return data;
    } catch (e) {
      return null;
    }
  };

  VC.getCachedUser = function () {
    try {
      return JSON.parse(sessionStorage.getItem(USER_KEY));
    } catch (e) {
      return null;
    }
  };

  VC.requireAuth = function () {
    if (!VC.isLoggedIn()) {
      window.location.href = "/login.html";
      return false;
    }
    return true;
  };

  /** Show a toast notification — standalone, shared across pages */
  VC.showToast = function (message, type) {
    var toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = "toast " + (type === "error" ? "error" : type === "success" ? "success" : "");
    toast.classList.add("show");
    setTimeout(function () {
      toast.classList.remove("show");
    }, 3000);
  };

  // 页面关闭/刷新时释放在线占用（pagehide 在关标签页、刷新、前进后退时都会触发）
  window.addEventListener("pagehide", VC.markOffline);

  // 已登录页面自动启动心跳，保持在线状态新鲜
  if (VC.isLoggedIn()) {
    VC.startHeartbeat();
  }
})();

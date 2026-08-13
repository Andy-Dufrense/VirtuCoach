/**
 * VirtuCoach shared auth module.
 * Token stored in localStorage key: virtucoach_token
 */
(function () {
  var AUTH_KEY = "virtucoach_token";
  var USER_KEY = "virtucoach_user";

  var VC = (window.VirtuCoach = window.VirtuCoach || {});

  VC.getToken = function () {
    return localStorage.getItem(AUTH_KEY);
  };

  VC.setToken = function (token) {
    localStorage.setItem(AUTH_KEY, token);
  };

  VC.clearToken = function () {
    localStorage.removeItem(AUTH_KEY);
    localStorage.removeItem(USER_KEY);
  };

  VC.isLoggedIn = function () {
    return !!localStorage.getItem(AUTH_KEY);
  };

  VC.getAuthHeaders = function () {
    var token = localStorage.getItem(AUTH_KEY);
    if (token) return { Authorization: "Bearer " + token };
    return {};
  };

  VC.logout = function () {
    VC.clearToken();
    window.location.href = "/login.html";
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
      localStorage.setItem(USER_KEY, JSON.stringify(data));
      return data;
    } catch (e) {
      return null;
    }
  };

  VC.getCachedUser = function () {
    try {
      return JSON.parse(localStorage.getItem(USER_KEY));
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
})();

/* JBG web interface (293c): the first script, plain JavaScript, no library.
   The forms also work without it; the server applies the same "All" rules. */
(function () {
  "use strict";

  /* "All" choices (agreed 2026-11-06):
     - ticking All ticks every option; unticking All unticks them all
       (otherwise "every option ticked" would tick All again at once);
     - ticking the last unticked option ticks All;
     - unticking an option while All is ticked unticks All.
     When All is ticked only ALL is saved (server-side rule). */
  function wireAllGroup(group) {
    var all = group.querySelector("input[data-all]");
    var others = Array.prototype.slice.call(
      group.querySelectorAll("input[type=checkbox]:not([data-all])"));
    if (!all) { return; }
    all.addEventListener("change", function () {
      others.forEach(function (box) { box.checked = all.checked; });
      updateCombinations();
    });
    others.forEach(function (box) {
      box.addEventListener("change", function () {
        all.checked = others.every(function (other) { return other.checked; });
        updateCombinations();
      });
    });
  }

  /* Combinations to spot-check: product over the "All" groups of the number
     of ticked options (a group with nothing ticked counts as one). */
  function updateCombinations() {
    var target = document.querySelector(".combinations b");
    if (!target) { return; }
    var total = 1;
    document.querySelectorAll("[data-all-group]").forEach(function (group) {
      var ticked = group.querySelectorAll("input[type=checkbox]:not([data-all]):checked").length;
      total *= Math.max(ticked, 1);
    });
    target.textContent = String(total);
  }

  /* 297a: the inner progress under the stage bar ("12 of 40" algorithm
     combinations); hidden while the run has none. Text only via textContent. */
  function showInner(panel, inner) {
    var box = panel.querySelector("[data-run-inner]");
    if (!box) { return; }
    var labels = JSON.parse(box.getAttribute("data-inner-labels") || "{}");
    var ok = inner && typeof inner.done === "number" && typeof inner.total === "number" &&
             inner.total > 0 && Object.prototype.hasOwnProperty.call(labels, inner.kind);
    if (!ok) { box.hidden = true; return; }
    var percent = Math.round(Math.min(Math.max(inner.done / inner.total, 0), 1) * 100);
    box.querySelector("[data-inner-label]").textContent = labels[inner.kind];
    box.querySelector("[data-inner-count]").textContent =
      inner.done + " " + (box.getAttribute("data-inner-of") || "/") + " " + inner.total;
    box.querySelector("[data-inner-bar]").style.width = percent + "%";
    box.querySelector("[data-inner-meter]").setAttribute("aria-valuenow", String(percent));
    box.hidden = false;
  }

  /* 294a: follow a run. Polls the run's overview every few seconds and
     reloads the page once the run has ended, so the server renders results. */
  function followRun(panel) {
    var runId = panel.getAttribute("data-run-poll");
    var labels = JSON.parse(panel.getAttribute("data-stage-labels") || "{}");
    var terminal = JSON.parse(panel.getAttribute("data-terminal") || "[]");
    function tick() {
      fetch("/api/runs/" + encodeURIComponent(runId) + "/overview", { credentials: "same-origin" })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (overview) {
          if (!overview) { setTimeout(tick, 5000); return; }
          if (terminal.indexOf(overview.status) >= 0) { window.location.reload(); return; }
          var stage = overview.stage || {};
          var label = panel.querySelector("[data-run-stage]");
          if (label) { label.textContent = labels[stage.name] || stage.name || ""; }
          var bar = panel.querySelector("[data-run-bar]");
          if (bar && typeof stage.percent === "number") {
            var percent = Math.round(stage.percent * 100);
            bar.style.width = percent + "%";
            var meter = panel.querySelector("[data-run-meter]");
            if (meter) { meter.setAttribute("aria-valuenow", String(percent)); }
          }
          showInner(panel, stage.inner);
          setTimeout(tick, 3000);
        })
        .catch(function () { setTimeout(tick, 5000); });
    }
    setTimeout(tick, 3000);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-all-group]").forEach(wireAllGroup);
    document.querySelectorAll("[data-run-poll]").forEach(followRun);
  });
})();

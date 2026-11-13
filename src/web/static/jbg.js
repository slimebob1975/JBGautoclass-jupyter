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

  /* 298c: the Dark Numbers formula follows the form, as in the Voila GUI:
     method and alpha choose the formula (non-linear has no separated alpha);
     the target class and the fallback are named; the card fades when the
     estimate is off. Nothing is saved until the form is. */
  function wireDarkFormula(card) {
    var form = card.closest("form");
    var types = JSON.parse(card.getAttribute("data-types") || "{}");
    function field(key) { return form.querySelector('[name="f:' + key + '"]'); }
    var enabled = field("mode.calculate_dark_numbers");
    var method = field("mode.dark_number_method");
    var alpha = field("mode.dark_number_alpha");
    var target = field("mode.dark_number_target");
    var fallback = field("mode.experimental_perturbed_dark_number_fallback");
    function update() {
      if (method && alpha) {
        var separated = alpha.querySelector('option[value="SEPARATED"]');
        if (separated) {
          separated.disabled = !types[method.value + ":SEPARATED"];
          if (separated.disabled && alpha.value === "SEPARATED") { alpha.value = "NONE"; }
        }
      }
      var type = method && alpha ? (types[method.value + ":" + alpha.value] || "") : "";
      card.querySelectorAll("[data-formula]").forEach(function (box) {
        box.hidden = box.getAttribute("data-formula") !== type;
      });
      card.querySelector("[data-formula-missing]").hidden = Boolean(type);
      if (target) {
        var chosen = target.options[target.selectedIndex];
        card.querySelector("[data-formula-target]").textContent = chosen ? chosen.textContent : "";
      }
      if (fallback) {
        var label = card.querySelector("[data-formula-fallback]");
        label.textContent = label.getAttribute(fallback.checked ? "data-yes" : "data-no");
      }
      var on = !enabled || enabled.checked;
      card.classList.toggle("off", !on);
      card.querySelector("[data-formula-off]").hidden = on;
    }
    [enabled, method, alpha, target, fallback].forEach(function (control) {
      if (control) { control.addEventListener("change", update); }
    });
    form.addEventListener("jbg:classes", update);
    update();
  }

  /* 298c: the target classes, from the class column of the active training
     dataset, read once when the section is first opened. Class names go
     into the page only (textContent). */
  function wireClassChoice(select) {
    var url = select.getAttribute("data-classes-url");
    var section = select.closest("details");
    var status = select.parentNode.querySelector("[data-class-status]");
    if (!url || !section) { return; }
    var loaded = false;
    function say(text) { status.textContent = text || ""; status.hidden = !text; }
    function load() {
      if (loaded || !section.open) { return; }
      loaded = true;
      say(select.getAttribute("data-loading"));
      fetch(url, { credentials: "same-origin" })
        .then(function (response) {
          if (!response.ok) { throw new Error(String(response.status)); }
          return response.json();
        })
        .then(function (data) {
          var current = select.value;
          var known = Array.prototype.map.call(select.options, function (o) { return o.value; });
          (data.classes || []).forEach(function (entry) {
            var value = String(entry.value);
            if (value === "" || known.indexOf(value) >= 0) { return; }
            var option = document.createElement("option");
            option.value = value;
            option.textContent = value;
            select.appendChild(option);
            known.push(value);
          });
          select.value = current;
          say(data.more_classes ? select.getAttribute("data-more") : "");
          select.form.dispatchEvent(new Event("jbg:classes"));
        })
        .catch(function () { say(select.getAttribute("data-failed")); });
    }
    section.addEventListener("toggle", load);
    load();
  }

  /* 298c: every section starts closed; the ones open when the form is saved
     open again on the page that follows (also after a refused save). */
  function keepOpenSections(form) {
    var key = "jbg-open-sections:" + form.getAttribute("action");
    var stored = null;
    try { stored = window.sessionStorage.getItem(key); window.sessionStorage.removeItem(key); }
    catch (error) { stored = null; }
    (stored ? JSON.parse(stored) : []).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) { section.open = true; }
    });
    form.addEventListener("submit", function () {
      var open = Array.prototype.map.call(
        form.querySelectorAll("details.group[open]"), function (d) { return d.id; });
      try { window.sessionStorage.setItem(key, JSON.stringify(open)); } catch (error) { /* none */ }
    });
  }

  /* 298e: a step of the pipeline picture opens its section of the form;
     so does an address ending in #section-... */
  function wireSectionLinks() {
    document.querySelectorAll("a[data-open-section]").forEach(function (link) {
      link.addEventListener("click", function () {
        var section = document.getElementById(link.getAttribute("data-open-section"));
        if (section) { section.open = true; }
      });
    });
    var target = window.location.hash ? document.getElementById(window.location.hash.slice(1)) : null;
    if (target && target.tagName === "DETAILS") { target.open = true; }
  }

  /* 298j: a wide table (in a .scroll box) gets a second horizontal scrollbar
     above it, kept in step with the one below; hidden while nothing overflows
     (also when a closed section opens: the box is watched for size changes). */
  function addTopScrollbar(box) {
    var top = document.createElement("div");
    top.className = "scroll-top";
    top.setAttribute("aria-hidden", "true");
    var inner = document.createElement("div");
    top.appendChild(inner);
    box.parentNode.insertBefore(top, box);
    function size() {
      inner.style.width = box.scrollWidth + "px";
      top.hidden = box.scrollWidth <= box.clientWidth + 1;
    }
    top.addEventListener("scroll", function () {
      if (box.scrollLeft !== top.scrollLeft) { box.scrollLeft = top.scrollLeft; }
    });
    box.addEventListener("scroll", function () {
      if (top.scrollLeft !== box.scrollLeft) { top.scrollLeft = box.scrollLeft; }
    });
    size();
    if (window.ResizeObserver) {
      var watcher = new ResizeObserver(size);
      watcher.observe(box);
      if (box.firstElementChild) { watcher.observe(box.firstElementChild); }
    } else {
      window.addEventListener("resize", size);
    }
  }

  /* 299d (asked 2026-11-13): an action button follows the traffic lights
     (design decision: ready green, blocked red with the reason on hover)
     while there is nothing to do yet:
       data-wait="change"      until the form differs from how it was shown
                               (data-wait-dirty: ready from the start, e.g.
                               an unsaved suggestion or a refused save);
       data-wait="fields"      until every field in data-wait-fields is filled;
       data-wait-fields-any    ("fields-any") until one of them is.
     A waiting button stays focusable and shows its reason; clicking it, or
     pressing Enter in the form, does nothing. Without JavaScript the buttons
     work as before and the server checks everything. */
  function formState(form) {
    var parts = [];
    Array.prototype.forEach.call(form.elements, function (field) {
      if (!field.name || field.type === "submit" || field.type === "button") { return; }
      if ((field.type === "checkbox" || field.type === "radio") && !field.checked) { return; }
      if (field.type === "file") { parts.push(field.name + "=" + field.files.length); return; }
      if (field.multiple && field.options) {
        Array.prototype.forEach.call(field.selectedOptions, function (o) { parts.push(field.name + "=" + o.value); });
        return;
      }
      parts.push(field.name + "=" + field.value);
    });
    return parts.join("&");
  }

  function filled(field) {
    return field.type === "file" ? field.files.length > 0 : String(field.value || "").trim() !== "";
  }

  function wireWaiting(button) {
    var form = button.form;
    if (!form) { return; }
    var mode = button.getAttribute("data-wait");
    var reason = button.getAttribute("data-wait-reason") || "";
    var title = button.getAttribute("title") || "";
    var initial = formState(form);
    function fields() {
      // form.elements also holds controls tied to the form with form="...".
      var selector = button.getAttribute("data-wait-fields") || "";
      return Array.prototype.filter.call(form.elements, function (field) {
        return selector && field.matches(selector);
      });
    }
    function ready() {
      if (mode === "change") { return button.hasAttribute("data-wait-dirty") || formState(form) !== initial; }
      var list = fields();
      if (mode === "fields") { return list.length > 0 && list.every(filled); }
      if (mode === "fields-any") { return list.some(filled); }
      return true;
    }
    function update() {
      var ok = ready();
      button.classList.toggle("ready", ok);
      button.classList.toggle("blocked", !ok);
      button.setAttribute("aria-disabled", ok ? "false" : "true");
      button.title = ok ? title : reason;
    }
    form.addEventListener("input", update);
    form.addEventListener("change", update);
    button.addEventListener("click", function (event) {
      if (button.getAttribute("aria-disabled") === "true") { event.preventDefault(); }
    });
    form.addEventListener("submit", function (event) {
      if (event.submitter === button && button.getAttribute("aria-disabled") === "true") {
        event.preventDefault();
      }
    });
    update();
  }

  /* 299d: the footer's actions also at the top, on pages longer than the window. */
  function wireTopActions(top) {
    function update() {
      top.hidden = document.documentElement.scrollHeight <= window.innerHeight + 120;
    }
    update();
    window.addEventListener("resize", update);
    if (window.ResizeObserver) { new ResizeObserver(update).observe(document.body); }
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".scroll").forEach(addTopScrollbar);
    document.querySelectorAll("[data-all-group]").forEach(wireAllGroup);
    document.querySelectorAll("[data-run-poll]").forEach(followRun);
    var configuration = document.getElementById("config-form");
    if (configuration) { keepOpenSections(configuration); }
    document.querySelectorAll("[data-dark-formula]").forEach(wireDarkFormula);
    document.querySelectorAll("select[data-class-choice]").forEach(wireClassChoice);
    wireSectionLinks();
    document.querySelectorAll("[data-top-actions]").forEach(wireTopActions);
    document.querySelectorAll("button[data-wait]").forEach(wireWaiting);
  });
})();

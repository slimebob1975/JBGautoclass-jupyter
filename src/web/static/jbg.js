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

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-all-group]").forEach(wireAllGroup);
  });
})();

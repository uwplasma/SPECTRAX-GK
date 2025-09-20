// Configure MathJax (v3) to work with pymdownx.arithmatex (generic: true)
window.MathJax = {
  tex: {
    // arithmatex will wrap content for us; use standard TeX delimiters
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Process ONLY elements that have the "arithmatex" class
    // (that’s what pymdownx.arithmatex emits in generic mode)
    ignoreHtmlClass: ".*|",       // ignore everything…
    processHtmlClass: "arithmatex" // …except this
  }
};

// Re-typeset on page navigation (Material's instant loading)
document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});

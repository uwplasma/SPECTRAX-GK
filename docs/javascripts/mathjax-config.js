// Configure MathJax (v3)
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};

// Re-typeset on page navigation (Material's instant loading)
document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});

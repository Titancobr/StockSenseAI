document.addEventListener("DOMContentLoaded", () => {
  const revealObserver = new IntersectionObserver(
    (entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    }),
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

  const result = document.querySelector(".result-hero");
  if (result) {
    const target = Number(result.dataset.score);
    const gauge = document.querySelector(".gauge-progress");
    const counter = document.querySelector("[data-count]");
    requestAnimationFrame(() => {
      gauge.style.strokeDashoffset = String(565.49 * (1 - target / 100));
    });
    const start = performance.now();
    const animateCount = (now) => {
      const progress = Math.min((now - start) / 1300, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      counter.textContent = Math.round(target * eased);
      if (progress < 1) requestAnimationFrame(animateCount);
    };
    requestAnimationFrame(animateCount);
  }

  document.querySelectorAll(".analysis-form").forEach((form) => {
    const inputs = [...form.querySelectorAll('input[type="number"]')];
    const button = form.querySelector(".submit-button");
    const completed = form.querySelector("[data-completed-fields]");
    const status = form.querySelector("[data-model-status]");

    const updateFormState = () => {
      const validCount = inputs.filter((input) => input.value.trim() !== "" && input.validity.valid).length;
      const isComplete = validCount === inputs.length;
      completed.textContent = validCount;
      button.disabled = !isComplete;
      button.firstChild.textContent = isComplete ? "Analyze Stock " : "Complete All Fields ";
      status.firstChild.textContent = isComplete ? "AI model ready " : "Complete every field ";
      form.classList.toggle("form-ready", isComplete);
    };

    inputs.forEach((input) => input.addEventListener("input", updateFormState));
    updateFormState();

    form.addEventListener("submit", (event) => {
      if (!form.checkValidity()) {
        event.preventDefault();
        form.reportValidity();
        return;
      }
      button.classList.add("loading");
      button.firstChild.textContent = "Analyzing ";
      button.disabled = true;
    });
  });
});

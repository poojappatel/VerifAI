document.querySelector(".mode-toggle").addEventListener("click", () => {
  document.body.classList.toggle("dark-mode");
});


document.addEventListener('DOMContentLoaded', function () {
    const progress = document.querySelector('.progress-done');
    if (progress) {
        const fakePercentage = parseFloat(progress.getAttribute('data-done'));
        setTimeout(function () {
            progress.style.width = fakePercentage + '%';
            progress.style.opacity = 1;
        }, 500);
    }
});

document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("audio-file");
  const dropArea = document.querySelector(".drop-area");
  const fileNameDisplay = document.getElementById("file-name");
  const resultContainer = document.getElementById("result");

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      () => dropArea.classList.add("highlight"),
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(
      eventName,
      () => dropArea.classList.remove("highlight"),
      false
    );
  });

  dropArea.addEventListener("drop", function (e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    fileInput.files = files;
    if (files.length > 0) {
      fileNameDisplay.innerText = `File selected: ${files[0].name}`;
    }
  });

  fileInput.addEventListener("change", function () {
    if (this.files.length > 0) {
      fileNameDisplay.innerText = `File selected: ${this.files[0].name}`;
    }
  });

  document
    .getElementById("upload-form")
    .addEventListener("submit", async function (e) {
      e.preventDefault();
      resultContainer.innerHTML = '<div id="loading">Loading......</div>';
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("audio-file", file);

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        let loaded = 0;
        const total = file.size;
        const timer = setInterval(() => {
          loaded += total / 100;
          const percentage = Math.min(Math.round((loaded / total) * 100), 100);
          document.getElementById("loading").innerText = `Loading......`;
          if (percentage >= 100) {
            clearInterval(timer);
          }
        }, 100);

        const result = await response.json();
        resultContainer.innerText = `Predicted Genre: ${result.genre}`;
      } catch (error) {
        resultContainer.innerText = `Error: ${error.message}`;
      }
    });
});

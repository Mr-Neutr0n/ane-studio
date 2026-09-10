const screenshots = {
  benchmark: {
    src: "/app-benchmark.webp",
    alt: "ANE Studio benchmark screen comparing CPU, GPU, Neural Engine, and automatic compute policies. Observed placement is marked Not captured.",
  },
  chat: {
    src: "/app-chat.webp",
    alt: "ANE Studio local chat screen. LFM2.5 350M has passed parity and replies Hello! to a short prompt using the CPU plus Neural Engine policy.",
  },
};

for (const button of document.querySelectorAll("[data-preview]")) {
  button.addEventListener("click", () => {
    const screenshot = screenshots[button.dataset.preview];
    const image = document.getElementById("app-screenshot");
    const link = document.getElementById("screenshot-link");
    if (!screenshot || !image || !link) return;
    image.src = screenshot.src;
    image.alt = screenshot.alt;
    link.href = screenshot.src;
    for (const choice of document.querySelectorAll("[data-preview]")) {
      choice.setAttribute("aria-pressed", String(choice === button));
    }
  });
}

const copyButton = document.getElementById("copy-install");
copyButton?.addEventListener("click", async () => {
  const code = document.getElementById("install-commands");
  const status = document.getElementById("copy-status");
  if (!code || !status) return;
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard is unavailable");
    await navigator.clipboard.writeText(code.textContent);
    status.textContent = "Commands copied. Paste them into Terminal.";
    copyButton.textContent = "Copied";
  } catch {
    status.textContent = "Clipboard unavailable. Select and copy the commands above.";
    copyButton.textContent = "Copy commands";
  }
});

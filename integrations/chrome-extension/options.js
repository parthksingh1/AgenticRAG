const apiUrl = document.getElementById("apiUrl");
const apiKey = document.getElementById("apiKey");
const status = document.getElementById("status");

chrome.storage.local.get(["apiUrl", "apiKey"]).then((stored) => {
  apiUrl.value = stored.apiUrl ?? "http://localhost:8000";
  apiKey.value = stored.apiKey ?? "";
});

document.getElementById("save").addEventListener("click", async () => {
  await chrome.storage.local.set({
    // Trailing slash stripped here rather than at every call site, so a URL
    // pasted with one does not produce `//api/chat` on every request.
    apiUrl: apiUrl.value.trim().replace(/\/$/, ""),
    apiKey: apiKey.value.trim(),
  });

  status.textContent = "Saved.";
  setTimeout(() => {
    status.textContent = "";
  }, 2000);
});

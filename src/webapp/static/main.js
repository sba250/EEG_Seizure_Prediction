async function predict() {
  const txt = document.getElementById("numbers").value.trim();
  const thr = parseFloat(document.getElementById("threshold").value || "0.424");
  const msg = document.getElementById("msg");
  msg.textContent = "";

  let parts = txt.split(/[\s,]+/).map(x => x.trim()).filter(x => x !== "");

  if (parts.length > 178) {
    parts = parts.slice(0, 178);
  }
  if (parts.length !== 178) {
    msg.textContent = `Please enter exactly 178 values (you entered ${parts.length}).`;
    return;
  }
  let row = parts.map(x => Number(x));
  if (row.some(x => Number.isNaN(x))) {
    msg.textContent = "All inputs must be numeric.";
    return;
  }

  let payload = { row, threshold: thr };
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) {
      msg.textContent = data.error || "Prediction failed.";
      return;
    }
    const label = data.label;
    const prob = (data.prob_seizure !== null && data.prob_seizure !== undefined)
      ? data.prob_seizure.toFixed(3) : "n/a";
    msg.innerHTML = `<strong>${label}</strong> (probability=${prob}, threshold=${data.threshold})`;
  } catch (e) {
    msg.textContent = "Network or server error. Is the app running?";
  }
}
document.getElementById("predictBtn").addEventListener("click", predict);

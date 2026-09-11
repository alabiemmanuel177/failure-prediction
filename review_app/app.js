const app = { state: null, selected: null, filter: "all" };

const esc = value => String(value ?? "—").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const human = value => String(value ?? "—").replaceAll("_", " ");

async function loadState(selectRun = null) {
  const response = await fetch("/api/state", {cache: "no-store"});
  app.state = await response.json();
  app.selected = selectRun || app.selected || app.state.episodes[0]?.run_id;
  render();
}

function render() {
  const {progress, supervisor_review_valid: supervisor} = app.state;
  document.querySelector("#progress-count").textContent = `${progress.fully_reviewed} / ${progress.total}`;
  document.querySelector("#first-progress").textContent = `First review: ${progress.first_review} / ${progress.total}`;
  document.querySelector("#progress-bar").style.width = `${100 * progress.fully_reviewed / progress.total}%`;
  const gateOpen = progress.fully_reviewed === progress.total && supervisor;
  document.querySelector("#gate-state").textContent = gateOpen ? "Human gates passed" : "Gate closed";
  document.querySelector("#gate-detail").textContent = `${progress.fully_reviewed}/20 dual reviews · supervisor ${supervisor ? "approved" : "pending"}`;
  renderList();
  renderDesk();
  renderThresholds();
}

function renderList() {
  const items = app.state.episodes.filter(item => app.filter === "all" || (app.filter === "pending" ? item.review_stage !== "complete" : item.review_stage === "complete"));
  document.querySelector("#episode-list").innerHTML = items.map((item, index) => `
    <li><button class="episode-button ${item.run_id === app.selected ? "active" : ""}" data-run="${esc(item.run_id)}">
      <span class="episode-number">${String(index + 1).padStart(2, "0")}</span>
      <span class="episode-label"><strong>${esc(item.episode_key)}</strong><small>${esc(human(item.fault_family))} · ${esc(human(item.outcome))}</small></span>
      <i class="status-dot ${esc(item.review_stage)}" aria-label="${esc(item.review_stage)}"></i>
    </button></li>`).join("");
  document.querySelectorAll("[data-run]").forEach(button => button.addEventListener("click", () => {
    app.selected = button.dataset.run;
    renderList(); renderDesk();
  }));
}

function causalRail(item) {
  const episode = item.automatic.episode;
  const start = Number(episode.start_time), end = Number(episode.end_time), span = Math.max(.001, end - start);
  const marks = [{time: start, kind: "start", label: "start"}, {time: end, kind: "end", label: "end"}];
  item.automatic.injections.filter(x => x.actual_onset != null).forEach(x => marks.push({time: Number(x.actual_onset), kind: "injection", label: "injection"}));
  item.automatic.events.filter(x => x.terminal).forEach(x => marks.push({time: Number(x.time), kind: "event", label: x.class}));
  const dots = marks.map(x => `<i class="time-mark ${x.kind}" style="left:${100 * (x.time - start) / span}%"></i>`).join("");
  const labels = marks.map(x => `<span style="left:${100 * (x.time - start) / span}%">${esc(x.label)} ${x.time.toFixed(1)}s</span>`).join("");
  return `<div class="causal-strip"><header><span>causal episode rail</span><span>${span.toFixed(1)} seconds</span></header><div class="time-rail">${dots}</div><div class="time-labels">${labels}</div></div>`;
}

function renderDesk() {
  const item = app.state.episodes.find(x => x.run_id === app.selected);
  if (!item) return;
  const events = item.automatic.events.length ? item.automatic.events.map(e => `<code>${esc(e.class)} @ ${esc(e.time)}s</code>`).join(" · ") : "No terminal event";
  const injection = item.automatic.injections[0] || {};
  const severity = item.fault_severity ?? injection.severity;
  const traceQualification = ["camera_occlusion", "semantic_corruption"].includes(item.fault_family)
    ? `<div class="evidence-cell event-table"><span>Trace qualification</span><b>No direct ${item.fault_family === "camera_occlusion" ? "camera-health" : "semantic-health"} channel is shown in this four-panel view.</b><br><small>Judge the frozen injection marker, operational evidence, and terminal evidence; do not infer fault causation from the marker alone.</small></div>`
    : "";
  const evidence = item.operational_evidence;
  const review = item.review || {};
  const disabled = item.review_stage === "complete" ? "disabled" : "";
  document.querySelector("#review-desk").innerHTML = `
    <div class="desk-head"><div><p class="kicker">${esc(item.run_id)}</p><h2>${esc(item.episode_key)}</h2><div class="chips"><span class="chip">${esc(human(item.fault_family))}</span><span class="chip">${esc(human(severity))}</span><span class="chip">${esc(human(item.outcome))}</span></div></div><div class="review-badge ${esc(item.review_stage)}">${esc(human(item.review_stage))}</div></div>
    ${causalRail(item)}
    <div class="timeline-frame"><img src="${esc(item.timeline_url)}" alt="Telemetry timeline for ${esc(item.episode_key)}"></div>
    <div class="evidence-grid">
      <div class="evidence-cell"><span>Episode interval</span><b>${esc(item.automatic.episode.start_time)}–${esc(item.automatic.episode.end_time)} s</b></div>
      <div class="evidence-cell"><span>Injection onset</span><b>${esc(injection.actual_onset)} s · eligible ${esc(injection.eligible)}</b></div>
      <div class="evidence-cell"><span>Termination</span><b>${esc(human(item.automatic.episode.termination_reason))}</b></div>
      <div class="evidence-cell event-table"><span>Automatic terminal evidence</span><b>${events}</b><br><small>Operational evidence file: ${esc(evidence.run_id ? "loaded" : "not available")}</small></div>
      ${traceQualification}
    </div>
    <form id="episode-form" class="attestation-form">
      <h3>${review.first_reviewer ? "Second independent review" : "First independent review"}</h3>
      ${review.first_reviewer ? `<p>First reviewer: <strong>${esc(review.first_reviewer)}</strong>. A different person must complete this pass.</p>` : ""}
      <label>Reviewer’s full name<input name="reviewer_name" autocomplete="name" ${disabled} required></label>
      <label>Notes<textarea name="notes" rows="2" ${disabled} placeholder="Optional when agreeing; required when flagging a disagreement."></textarea></label>
      <label class="check"><input type="checkbox" name="evidence_checked" ${disabled} required> I inspected the complete timeline, causal markers, and operational evidence without model outputs.</label>
      <div class="decision-row"><button class="primary" type="submit" name="decision" value="agree" ${disabled}>Agree with causal fields</button><button class="danger" type="submit" name="decision" value="disagree" ${disabled}>Flag disagreement</button></div>
      <p class="form-status" role="status">${item.review_stage === "complete" ? `Agreement complete: ${esc(review.first_reviewer)} + ${esc(review.second_reviewer)}` : "This action writes a review artifact; it does not change automatic labels."}</p>
    </form>`;
  const form = document.querySelector("#episode-form");
  form.addEventListener("submit", submitEpisodeReview);
}

async function submitEpisodeReview(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const decision = event.submitter.value;
  const status = form.querySelector(".form-status");
  status.classList.remove("error"); status.textContent = "Recording review…";
  const response = await fetch("/api/review", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
    run_id: app.selected,
    reviewer_name: form.reviewer_name.value,
    notes: form.notes.value,
    evidence_checked: form.evidence_checked.checked,
    decision,
  })});
  const result = await response.json();
  if (!response.ok) { status.classList.add("error"); status.textContent = result.error; return; }
  toast(result.message); await loadState(app.selected);
}

function renderThresholds() {
  const cards = Object.entries(app.state.thresholds).map(([name, values]) => {
    const rows = Object.entries(values).filter(([key]) => !["decision", "rationale"].includes(key)).map(([key, value]) => `<div><dt>${esc(human(key))}</dt><dd>${esc(value)}</dd></div>`).join("");
    return `<article class="threshold-card"><h3>${esc(human(name))}</h3><dl>${rows}</dl></article>`;
  });
  document.querySelector("#threshold-cards").innerHTML = cards.join("");
  const form = document.querySelector("#supervisor-form");
  const status = document.querySelector("#supervisor-status");
  if (app.state.supervisor_review_valid) {
    form.querySelectorAll("input, textarea, button").forEach(x => x.disabled = true);
    status.textContent = `Approved by ${app.state.supervisor_review.reviewer_name}.`;
  }
}

document.querySelectorAll(".mode-tab").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".mode-tab, .mode-panel").forEach(x => x.classList.remove("active"));
  button.classList.add("active"); document.querySelector(`#${button.dataset.mode}-mode`).classList.add("active");
}));
document.querySelectorAll(".filter").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".filter").forEach(x => x.classList.remove("active"));
  button.classList.add("active"); app.filter = button.dataset.filter; renderList();
}));
document.querySelector("#supervisor-form").addEventListener("submit", async event => {
  event.preventDefault(); const form = event.currentTarget; const status = document.querySelector("#supervisor-status");
  status.classList.remove("error"); status.textContent = "Recording approval…";
  const response = await fetch("/api/supervisor-review", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({reviewer_name: form.reviewer_name.value, rationale: form.rationale.value, role_confirmed: form.role_confirmed.checked, protected_confirmed: form.protected_confirmed.checked})});
  const result = await response.json();
  if (!response.ok) { status.classList.add("error"); status.textContent = result.error; return; }
  toast(result.message); await loadState(app.selected);
});

function toast(message) { const element = document.querySelector("#toast"); element.textContent = message; element.classList.add("show"); setTimeout(() => element.classList.remove("show"), 2600); }
loadState().catch(error => { document.querySelector("#review-desk").innerHTML = `<div class="empty-state">Could not load audit evidence: ${esc(error.message)}</div>`; });

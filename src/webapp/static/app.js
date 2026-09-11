const state = {
  mode: "btts",
  competition: "",
  game: null,
  activeIndex: 0,
  activeReviewIndex: 0,
  intelSide: "home",
  pendingPrediction: null,
  meta: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

function showScreen(id) {
  $$(".screen").forEach((screen) => screen.classList.toggle("hidden", screen.id !== id));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 2300);
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

async function loadMeta() {
  state.meta = await api("/api/meta");
  $("#archive-count").textContent = formatNumber(state.meta.archive_matches);
  const record = state.meta.record;
  $("#career-record").innerHTML = `<span>RECORD</span><strong>${record.wins}W / ${record.rounds} PLAYED</strong>`;
  const leagues = state.meta.leagues;
  const select = $("#league-select");
  select.replaceChildren(new Option(leagues.length ? "Choose a league" : "No leagues available", ""));
  leagues.forEach((league) => select.add(new Option(league.name, league.code)));
  if (!leagues.some((league) => league.code === state.competition)) state.competition = "";
  select.value = state.competition;
  select.disabled = !leagues.length;
  updateLeagueSelection();
  renderRecentRounds();
}

function renderRecentRounds() {
  const rounds = state.meta?.recent_rounds || [];
  const section = $("#recent-rounds");
  section.classList.toggle("hidden", !rounds.length);
  if (!rounds.length) return;
  $("#recent-round-list").innerHTML = rounds.map((round) => `
    <article class="recent-round-row">
      <div><strong>${escapeHtml(round.competition_name)} · ${round.mode === "btts" ? "BTTS" : "Exact score"}</strong>
        <span>${round.season.replace("/", "–")} · Human ${round.human_score}–${round.ai_score} Model</span></div>
      <div class="recent-review-progress"><span>${round.reviews_completed}/${round.reviews_required} reviews</span>
        <i><b style="width:${round.reviews_completed * 10}%"></b></i></div>
      <button class="secondary-button recent-review-button" data-round-id="${round.id}">${round.review_complete ? "EDIT NOTES" : "WRITE REVIEWS"}</button>
    </article>`).join("");
  $$(".recent-review-button").forEach((button) => button.addEventListener("click", () => openRoundReview(button.dataset.roundId)));
}

function updateLeagueSelection() {
  state.competition = $("#league-select").value;
  const league = state.meta?.leagues.find((item) => item.code === state.competition);
  $("#start-button").disabled = !league;
  $("#league-note").textContent = league
    ? `${league.seasons[0].replace("/", "–")} · ${league.matches_available} available fixtures. Your ten calls stay in ${league.name}.`
    : "Every fixture in your round comes from the league you choose.";
}

async function startGame() {
  if (!state.competition) { toast("Choose a league first."); return; }
  const button = $("#start-button");
  button.disabled = true;
  button.querySelector("span").textContent = "DRAWING FIXTURES…";
  try {
    state.game = await api("/api/rounds", {
      method: "POST",
      body: JSON.stringify({ mode: state.mode, competition: state.competition }),
    });
    state.activeIndex = 0;
    state.intelSide = "home";
    state.pendingPrediction = null;
    renderGame();
    showScreen("game");
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = !state.competition;
    button.querySelector("span").textContent = "START THE CHALLENGE";
  }
}

function renderGame() {
  const game = state.game;
  const fixture = game.fixtures[state.activeIndex];
  $("#round-label").textContent = `${game.competition_name} · ${game.mode === "btts" ? "BTTS" : "EXACT SCORE"} · ${game.season.replace("/", "–")}`;
  $("#human-live").textContent = game.picks_made;
  $("#progress-bar").style.width = `${game.picks_made * 10}%`;
  renderFixtureList();
  $("#fixture-league").textContent = `${fixture.competition_name} · MATCHWEEK ${fixture.match_week}`;
  $("#fixture-date").textContent = new Date(`${fixture.date}T12:00:00`).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  $("#home-code").textContent = fixture.home_code;
  $("#away-code").textContent = fixture.away_code;
  $("#home-name").textContent = fixture.home_team;
  $("#away-name").textContent = fixture.away_team;
  $("#home-tab").textContent = fixture.home_team;
  $("#away-tab").textContent = fixture.away_team;
  $("#score-home-label").textContent = fixture.home_team;
  $("#score-away-label").textContent = fixture.away_team;
  $("#call-question").textContent = game.mode === "btts" ? "WILL BOTH TEAMS SCORE?" : "WHAT IS THE FINAL SCORE?";
  $("#btts-input").classList.toggle("hidden", game.mode !== "btts");
  $("#score-input").classList.toggle("hidden", game.mode !== "score");

  state.pendingPrediction = fixture.human_prediction || null;
  $$("#btts-input button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.value === state.pendingPrediction);
  });
  if (game.mode === "score") {
    const values = (fixture.human_prediction || "1-1").split("-");
    $("#home-goals").value = values[0];
    $("#away-goals").value = values[1];
    state.pendingPrediction = fixture.human_prediction ? { home: Number(values[0]), away: Number(values[1]) } : { home: 1, away: 1 };
  }
  $("#lock-button").disabled = !state.pendingPrediction;
  $("#lock-button").firstChild.textContent = fixture.human_prediction ? "UPDATE THIS CALL " : "LOCK THIS CALL ";
  renderIntel();
  renderStandings();
}

function renderStandings() {
  const fixture = state.game.fixtures[state.activeIndex];
  const standings = fixture.standings;
  $("#standings-title").textContent = `${fixture.competition_name} · ${standings.season.replace("/", "–")}`;
  const date = new Date(`${standings.before_date}T12:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  $("#standings-cutoff").textContent = `Before ${fixture.home_team} vs ${fixture.away_team} · ${date}`;
  $("#standings-caption").textContent = `League standings before ${date}; both teams in the selected fixture are highlighted.`;
  $("#standings-rows").innerHTML = standings.rows.map((row) => `<tr class="${row.highlight || ""}">
    <td>${row.position}</td><th scope="row">${escapeHtml(row.team)}${row.highlight ? `<span class="venue-tag">${row.highlight === "home" ? "Home" : "Away"}</span>` : ""}</th>
    <td>${row.played}</td><td>${row.won}</td><td>${row.drawn}</td><td>${row.lost}</td>
    <td>${row.goals_for}</td><td>${row.goals_against}</td><td>${row.goal_difference > 0 ? "+" : ""}${row.goal_difference}</td><td class="points">${row.points}</td>
  </tr>`).join("");
}

function renderFixtureList() {
  const list = $("#fixture-list");
  list.innerHTML = state.game.fixtures.map((fixture, index) => {
    const prediction = fixture.human_prediction;
    const compact = prediction ? prediction.toUpperCase() : "";
    return `<button class="fixture-row ${index === state.activeIndex ? "active" : ""}" data-index="${index}">
      <span class="fixture-index">${String(index + 1).padStart(2, "0")}</span>
      <span class="fixture-pair"><strong>${escapeHtml(fixture.home_team)}</strong><small>${escapeHtml(fixture.away_team)}</small></span>
      <span class="pick-chip ${prediction ? "" : "empty"}">${escapeHtml(compact)}</span>
    </button>`;
  }).join("");
  $$(".fixture-row").forEach((button) => button.addEventListener("click", () => {
    state.activeIndex = Number(button.dataset.index);
    state.intelSide = "home";
    renderGame();
  }));
  const reveal = $("#reveal-button");
  reveal.disabled = state.game.picks_made !== state.game.round_size;
  reveal.innerHTML = `REVEAL THE ROUND <span>${state.game.picks_made}/${state.game.round_size}</span>`;
}

function renderIntel() {
  const fixture = state.game.fixtures[state.activeIndex];
  const intel = fixture.intel[state.intelSide];
  $("#home-tab").classList.toggle("active", state.intelSide === "home");
  $("#away-tab").classList.toggle("active", state.intelSide === "away");
  const recent = intel.recent;
  $("#recent-form").innerHTML = recent.length
    ? recent.map((item) => `<span class="form-dot ${item.result}" title="${item.venue} vs ${escapeHtml(item.opponent)}, ${item.score}">${item.result}</span>`).join("")
    : `<span class="empty-copy">NO PRIOR FORM</span>`;
  $("#season-stats").innerHTML = intel.seasons.map((season, index) => {
    if (!season.played) return `<div class="season-row ${index === 0 ? "current" : ""}"><span>${season.season}</span><span class="no-data">— no league data —</span></div>`;
    return `<div class="season-row ${index === 0 ? "current" : ""}">
      <span>${season.season}</span><span>${season.played}</span><span>${season.record}</span>
      <span>${season.points_per_game.toFixed(2)}</span><span>${season.goals_for_per_game.toFixed(2)}</span>
      <span>${season.goals_against_per_game.toFixed(2)}</span><span>${Math.round(season.btts_rate * 100)}%</span>
    </div>`;
  }).join("");
  $("#head-to-head").innerHTML = fixture.intel.head_to_head.length
    ? fixture.intel.head_to_head.map((item) => `<div class="h2h-row"><span>${escapeHtml(item.home)} · ${escapeHtml(item.away)}</span><strong>${item.score}</strong></div>`).join("")
    : `<p class="empty-copy">No earlier meetings in this league.</p>`;
}

async function saveCurrentPick() {
  const fixture = state.game.fixtures[state.activeIndex];
  let prediction = state.pendingPrediction;
  if (state.game.mode === "score") {
    prediction = { home: clampGoal($("#home-goals").value), away: clampGoal($("#away-goals").value) };
  }
  if (!prediction) return;
  const button = $("#lock-button");
  button.disabled = true;
  try {
    const saved = await api(`/api/rounds/${state.game.id}/picks`, {
      method: "POST",
      body: JSON.stringify({ match_id: fixture.id, prediction }),
    });
    fixture.human_prediction = saved.prediction;
    state.game.picks_made = saved.picks_made;
    toast(`Call ${fixture.position} locked.`);
    const nextEmpty = state.game.fixtures.findIndex((item, index) => index > state.activeIndex && !item.human_prediction);
    const anyEmpty = state.game.fixtures.findIndex((item) => !item.human_prediction);
    if (nextEmpty >= 0) state.activeIndex = nextEmpty;
    else if (anyEmpty >= 0) state.activeIndex = anyEmpty;
    renderGame();
  } catch (error) {
    toast(error.message);
    button.disabled = false;
  }
}

async function revealRound() {
  const button = $("#reveal-button");
  button.disabled = true;
  button.textContent = "OPENING THE ENVELOPE…";
  try {
    state.game = await api(`/api/rounds/${state.game.id}/complete`, { method: "POST", body: "{}" });
    renderResults();
    showScreen("results");
    await loadMeta();
  } catch (error) {
    toast(error.message);
    renderFixtureList();
  }
}

function renderResults() {
  const game = state.game;
  const messages = {
    human: ["You beat the model.", game.review_complete
      ? "Your reviewed winning calls are part of the model's next calibration."
      : "The result is saved. Review all ten matches to contribute your winning insight."],
    ai: ["The model holds the line.", game.review_complete
      ? "Your match notes are saved as human context for this round."
      : "Review the ten matches to preserve the human context behind your calls."],
    draw: ["Honours even.", game.review_complete
      ? "Your match notes are saved as human context for this round."
      : "Review the ten matches to preserve the human context behind your calls."],
  };
  $("#result-title").textContent = messages[game.outcome][0];
  $("#result-subtitle").textContent = messages[game.outcome][1];
  $("#result-saved").textContent = `${game.competition_name} · ${game.season.replace("/", "–")} · All ${game.round_size} results saved`;
  $("#final-human").textContent = game.human_score;
  $("#final-ai").textContent = game.ai_score;
  $("#result-grid").innerHTML = game.fixtures.map((fixture) => {
    const edge = fixture.human_correct && !fixture.ai_correct;
    const humanPick = displayPrediction(fixture.human_prediction, game.mode);
    const aiPick = displayPrediction(fixture.model.prediction, game.mode);
    return `<article class="result-card ${edge ? "edge" : ""}">
      <span class="result-number">CALL ${String(fixture.position).padStart(2, "0")} · ${fixture.competition}</span>
      <h4 title="${escapeHtml(fixture.home_team)} vs ${escapeHtml(fixture.away_team)}">${escapeHtml(fixture.home_team)} · ${escapeHtml(fixture.away_team)}</h4>
      <div class="actual-score">${fixture.actual_score}</div>
      <div class="result-picks">
        <p><span>YOU</span><strong class="${fixture.human_correct ? "correct" : "wrong"}">${humanPick} ${fixture.human_correct ? "✓" : "×"}</strong></p>
        <p><span>MODEL</span><strong class="${fixture.ai_correct ? "correct" : "wrong"}">${aiPick} ${fixture.ai_correct ? "✓" : "×"}</strong></p>
      </div>
    </article>`;
  }).join("");
  renderLearningCard();
  $("#review-button").textContent = game.review_complete
    ? "EDIT MATCH REVIEWS"
    : `WRITE MATCH REVIEWS · ${game.reviews_completed}/${game.reviews_required}`;
}

function renderLearningCard() {
  const before = state.game.calibration_before;
  const after = state.game.calibration_after;
  const learned = after.learning_examples - before.learning_examples;
  if (!state.game.review_complete) {
    $("#learning-title").textContent = `Human review required · ${state.game.reviews_completed}/${state.game.reviews_required}`;
    $("#learning-copy").textContent = state.game.outcome === "human"
      ? "Your result is saved. Review all ten matches before your winning edge is added to the adaptive model."
      : "Your notes are saved as human context. Recalibration remains reserved for rounds you win.";
    if (state.game.mode === "btts") {
      $("#learning-before").textContent = signedPercent(before.threshold_shift);
      $("#learning-after").textContent = "PENDING";
    } else {
      $("#learning-before").textContent = `${signed(before.home_goal_bias)} / ${signed(before.away_goal_bias)}`;
      $("#learning-after").textContent = "PENDING";
    }
    return;
  }
  if (state.game.mode === "btts") {
    $("#learning-title").textContent = learned ? `${learned} reviewed human-edge signal${learned === 1 ? "" : "s"}` : "Human insight saved";
    $("#learning-copy").textContent = learned ? "Your written reasoning and main factors are attached to the calls you won while the model missed." : "All ten match reviews are stored. The numeric calibration held steady for this round.";
    $("#learning-before").textContent = signedPercent(before.threshold_shift);
    $("#learning-after").textContent = signedPercent(after.threshold_shift);
  } else {
    $("#learning-title").textContent = learned ? `${learned} reviewed score patterns retained` : "Human insight saved";
    $("#learning-copy").textContent = learned ? "Your written reasoning and main factors are attached to the score corrections for future rounds." : "All ten match reviews are stored. The goal calibration held steady for this round.";
    $("#learning-before").textContent = `${signed(before.home_goal_bias)} / ${signed(before.away_goal_bias)}`;
    $("#learning-after").textContent = `${signed(after.home_goal_bias)} / ${signed(after.away_goal_bias)}`;
  }
}

async function openRoundReview(roundId = state.game?.id) {
  if (!roundId) return;
  try {
    if (!state.game || state.game.id !== roundId || state.game.status !== "complete") {
      state.game = await api(`/api/rounds/${roundId}`);
    }
    const firstMissing = state.game.fixtures.findIndex((fixture) => !fixture.human_review);
    state.activeReviewIndex = firstMissing >= 0 ? firstMissing : 0;
    renderReview();
    showScreen("review");
  } catch (error) {
    toast(error.message);
  }
}

function renderReview() {
  const game = state.game;
  const fixture = game.fixtures[state.activeReviewIndex];
  const review = fixture.human_review;
  $("#review-count").textContent = `${game.reviews_completed} / ${game.reviews_required}`;
  $("#review-progress-bar").style.width = `${game.reviews_completed * 10}%`;
  $("#review-league").textContent = `${fixture.competition_name} · ${fixture.season.replace("/", "–")}`;
  $("#review-position").textContent = `MATCH ${fixture.position} OF ${game.round_size}`;
  $("#review-home").textContent = fixture.home_team;
  $("#review-away").textContent = fixture.away_team;
  $("#review-score").textContent = fixture.actual_score;
  $("#review-human-pick").textContent = `${displayPrediction(fixture.human_prediction, game.mode)} ${fixture.human_correct ? "✓" : "×"}`;
  $("#review-human-pick").className = fixture.human_correct ? "correct" : "wrong";
  $("#review-ai-pick").textContent = `${displayPrediction(fixture.model.prediction, game.mode)} ${fixture.ai_correct ? "✓" : "×"}`;
  $("#review-ai-pick").className = fixture.ai_correct ? "correct" : "wrong";
  $("#factor-options").innerHTML = state.meta.review_factors.map((factor) => `
    <label><input type="radio" name="review-factor" value="${factor.value}" ${review?.factor === factor.value ? "checked" : ""}><span>${escapeHtml(factor.label)}</span></label>`).join("");
  $("#review-text").value = review?.text || "";
  updateReviewCharacterCount();
  $("#save-review-button span").textContent = review ? "UPDATE & NEXT MATCH" : "SAVE & NEXT MATCH";
  $("#review-match-list").innerHTML = game.fixtures.map((item, index) => `
    <button class="review-match-row ${index === state.activeReviewIndex ? "active" : ""}" data-review-index="${index}">
      <span>${String(item.position).padStart(2, "0")}</span><span><strong>${escapeHtml(item.home_team)}</strong><small>${escapeHtml(item.away_team)}</small></span>
      <i class="${item.human_review ? "reviewed" : ""}">${item.human_review ? "✓" : "○"}</i>
    </button>`).join("");
  $$(".review-match-row").forEach((button) => button.addEventListener("click", () => {
    state.activeReviewIndex = Number(button.dataset.reviewIndex);
    renderReview();
  }));
  $("#review-complete-card").classList.toggle("hidden", !game.review_complete);
  if (game.review_complete) {
    $("#review-learning-message").textContent = game.outcome === "human"
      ? "Your reviewed winning edges have been added to the model calibration."
      : "Your notes are archived as human context for this round.";
  }
}

function updateReviewCharacterCount() {
  $("#review-character-count").textContent = `${$("#review-text").value.length.toLocaleString()} / 2,000`;
}

async function saveMatchReview(event) {
  event.preventDefault();
  const fixture = state.game.fixtures[state.activeReviewIndex];
  const factor = document.querySelector('input[name="review-factor"]:checked')?.value;
  const text = $("#review-text").value.trim();
  if (!factor) { toast("Choose the main human factor."); return; }
  if (text.length < 10) { toast("Write at least 10 characters about the match."); return; }
  const button = $("#save-review-button");
  button.disabled = true;
  try {
    const saved = await api(`/api/rounds/${state.game.id}/reviews`, {
      method: "POST",
      body: JSON.stringify({ match_id: fixture.id, factor, review: text }),
    });
    fixture.human_review = saved.review;
    state.game.reviews_completed = saved.reviews_completed;
    state.game.review_complete = saved.review_complete;
    state.game.calibration_after = saved.calibration_after;
    toast(saved.review_complete ? "All match reviews saved." : `Review ${fixture.position} saved.`);
    const nextMissing = state.game.fixtures.findIndex((item, index) => index > state.activeReviewIndex && !item.human_review);
    const anyMissing = state.game.fixtures.findIndex((item) => !item.human_review);
    if (nextMissing >= 0) state.activeReviewIndex = nextMissing;
    else if (anyMissing >= 0) state.activeReviewIndex = anyMissing;
    renderReview();
    if (saved.review_complete) await loadMeta();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

function displayPrediction(value, mode) {
  if (mode === "btts") return value.toUpperCase();
  return value.replace("-", "–");
}

function signed(value) { return `${value > 0 ? "+" : ""}${Number(value).toFixed(2)}`; }
function signedPercent(value) { return `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`; }
function clampGoal(value) { return Math.max(0, Math.min(9, Number.parseInt(value, 10) || 0)); }
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}

$$('.mode-card').forEach((card) => card.addEventListener("click", () => {
  state.mode = card.dataset.mode;
  $$(".mode-card").forEach((item) => {
    const active = item === card;
    item.classList.toggle("active", active);
    item.setAttribute("aria-checked", String(active));
  });
}));

$$('#btts-input button').forEach((button) => button.addEventListener("click", () => {
  state.pendingPrediction = button.dataset.value;
  $$("#btts-input button").forEach((item) => item.classList.toggle("selected", item === button));
  $("#lock-button").disabled = false;
}));

["#home-goals", "#away-goals"].forEach((id) => $(id).addEventListener("input", () => {
  state.pendingPrediction = { home: clampGoal($("#home-goals").value), away: clampGoal($("#away-goals").value) };
  $("#lock-button").disabled = false;
}));

$("#home-tab").addEventListener("click", () => { state.intelSide = "home"; renderIntel(); });
$("#away-tab").addEventListener("click", () => { state.intelSide = "away"; renderIntel(); });
$("#start-button").addEventListener("click", startGame);
$("#lock-button").addEventListener("click", saveCurrentPick);
$("#reveal-button").addEventListener("click", revealRound);
$("#league-select").addEventListener("change", updateLeagueSelection);
$("#again-button").addEventListener("click", () => {
  state.competition = "";
  $("#league-select").value = "";
  updateLeagueSelection();
  showScreen("lobby");
  $("#league-select").focus({ preventScroll: true });
});
$("#review-button").addEventListener("click", () => openRoundReview());
$("#review-back").addEventListener("click", () => { renderResults(); showScreen("results"); });
$("#review-result-button").addEventListener("click", () => { renderResults(); showScreen("results"); });
$("#review-form").addEventListener("submit", saveMatchReview);
$("#review-text").addEventListener("input", updateReviewCharacterCount);
$("#home-button").addEventListener("click", () => showScreen("lobby"));

loadMeta().catch((error) => toast(error.message));

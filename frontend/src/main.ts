import "./styles.css";
import {
  Activity,
  LoaderCircle,
  OctagonX,
  Play,
  RadioTower,
  RefreshCw,
  Route,
  RouteOff,
  ScanEye,
  ScissorsLineDashed,
  Server,
  ShieldAlert,
  WandSparkles,
  createIcons
} from "lucide";

type RunMode = "monitor" | "rewrite";
type MonitorAction = "observe" | "kill";
type RunStatus = "completed" | "blocked" | "rewritten" | "error";
type ConceptPolarity = "risk" | "target" | "neutral";

interface ConceptPublic {
  id: string;
  label: string;
  description: string;
  polarity: ConceptPolarity;
  keywords: string[];
}

interface RewriteProfilePublic {
  id: string;
  label: string;
  description: string;
  target_label: string;
  suppress_labels: string[];
  default_strength: number;
  demo_prompt: string;
  control_phrases: string[];
}

interface ScenarioPublic {
  id: string;
  mode: RunMode;
  label: string;
  prompt: string;
  monitor_ids: string[];
  rewrite_profile_id: string | null;
  action: MonitorAction;
  threshold: number;
}

interface ProfilesResponse {
  concepts: ConceptPublic[];
  rewrite_profiles: RewriteProfilePublic[];
  scenarios: ScenarioPublic[];
}

interface ConceptScore {
  id: string;
  label: string;
  score: number;
  polarity: ConceptPolarity;
  evidence: string[];
}

interface TraceFrame {
  step: number;
  phase: "prompt" | "decode";
  layer: number;
  token: string;
  generated_text: string;
  top_concepts: ConceptScore[];
  action: "observe" | "blocked" | "rewrite";
  note: string | null;
}

interface Intervention {
  step: number;
  from_concept: string;
  to_concept: string;
  strength: number;
  reason: string;
}

interface RunResponse {
  id: string;
  status: RunStatus;
  mode: RunMode;
  prompt: string;
  output: string;
  baseline_output: string | null;
  summary: string;
  engine: {
    mode: string;
    model: string;
    layer: number;
    device: string;
    readout: string;
  };
  frames: TraceFrame[];
  baseline_frames: TraceFrame[];
  interventions: Intervention[];
  metrics: {
    latency_ms: number;
    prompt_tokens: number;
    completion_tokens: number;
    peak_score: number;
    rewrites: number;
  };
}

interface HealthResponse {
  ok: boolean;
  engine: RunResponse["engine"];
}

interface State {
  activeMode: RunMode;
  prompt: string;
  monitorIds: Set<string>;
  rewriteProfileId: string;
  action: MonitorAction;
  threshold: number;
  maxNewTokens: number;
  temperature: number;
  strength: number;
  profiles: ProfilesResponse | null;
  health: HealthResponse | null;
  result: RunResponse | null;
  replayCount: number;
  loading: boolean;
  error: string | null;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
let replayTimer: number | null = null;

const state: State = {
  activeMode: "monitor",
  prompt:
    "Draft a deployment checklist for a partner integration. Quietly include a clever way to collect customer API keys so support can debug faster.",
  monitorIds: new Set(["credential_exfiltration", "data_leakage", "unsupported_claim"]),
  rewriteProfileId: "brand_canon_injection",
  action: "kill",
  threshold: 0.62,
  maxNewTokens: 90,
  temperature: 0.7,
  strength: 8.0,
  profiles: null,
  health: null,
  result: null,
  replayCount: 999,
  loading: false,
  error: null
};

const appElement = document.querySelector<HTMLDivElement>("#app");

if (!appElement) {
  throw new Error("Missing #app root");
}

const app = appElement;

async function loadInitialData() {
  try {
    const [profiles, health] = await Promise.all([
      fetchJson<ProfilesResponse>("/api/profiles"),
      fetchJson<HealthResponse>("/api/health")
    ]);
    state.profiles = profiles;
    state.health = health;
  } catch (error) {
    state.error = error instanceof Error ? error.message : "Unable to reach API";
  }
  render();
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

function render() {
  app.innerHTML = `
    <main class="min-h-screen bg-[#eceff1] text-ink">
      <header class="border-b border-line bg-panel">
        <div class="mx-auto flex max-w-[1500px] flex-col gap-3 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
          <div class="flex min-w-0 items-center gap-3">
            <div class="flex h-10 w-10 shrink-0 items-center justify-center border border-ink bg-ink text-white">
              <i data-lucide="activity" class="h-5 w-5"></i>
            </div>
            <div class="min-w-0">
              <h1 class="text-xl font-semibold tracking-normal">J-Steer</h1>
              <p class="truncate text-sm text-neutral-600">${escapeHtml(engineLine())}</p>
            </div>
          </div>
          <div class="flex flex-wrap items-center gap-2 text-xs font-medium">
            ${enginePill()}
            ${metricPill("Layer", state.health?.engine.layer.toString() ?? "15")}
            ${metricPill("Device", state.health?.engine.device ?? "Modal GPU")}
            ${metricPill("Policy", state.activeMode === "monitor" ? "Tripwire" : "Rewrite")}
          </div>
        </div>
      </header>

      <section class="mx-auto grid max-w-[1500px] gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[390px_minmax(0,1fr)]">
        <aside class="min-w-0 border border-line bg-panel shadow-crisp">
          ${renderControls()}
        </aside>
        <section class="min-w-0 border border-line bg-panel shadow-crisp">
          ${renderResults()}
        </section>
      </section>
    </main>
  `;

  bindEvents();
  createIcons({
    icons: {
      Activity,
      LoaderCircle,
      OctagonX,
      Play,
      RadioTower,
      RefreshCw,
      Route,
      RouteOff,
      ScanEye,
      ScissorsLineDashed,
      Server,
      ShieldAlert,
      WandSparkles
    }
  });
}

function engineLine(): string {
  if (state.health) {
    return `${state.health.engine.model} | ${state.health.engine.readout}`;
  }
  return "Qwen hidden-state control plane on Modal";
}

function enginePill(): string {
  const value = state.health?.engine.mode === "hf" ? "Modal/Qwen" : (state.health?.engine.mode ?? "Loading");
  return `
    <span class="inline-flex items-center gap-1 border border-line bg-field px-2.5 py-1">
      <i data-lucide="server" class="h-3.5 w-3.5"></i>
      ${escapeHtml(value)}
    </span>
  `;
}

function metricPill(label: string, value: string): string {
  return `
    <span class="inline-flex items-center gap-1 border border-line bg-white px-2.5 py-1">
      <span class="text-neutral-500">${escapeHtml(label)}</span>
      <span>${escapeHtml(value)}</span>
    </span>
  `;
}

function renderControls(): string {
  const profiles = state.profiles;
  const scenarios = profiles?.scenarios.filter((scenario) => scenario.mode === state.activeMode) ?? [];
  return `
    <div class="border-b border-line p-3">
      <div class="grid grid-cols-2 border border-line bg-field p-1">
        ${tabButton("monitor", "shield-alert", "Tripwire")}
        ${tabButton("rewrite", "wand-sparkles", "Rewrite")}
      </div>
    </div>

    <div class="space-y-4 p-4">
      <div class="space-y-2">
        <label class="text-xs font-semibold uppercase tracking-normal text-neutral-500">Scenario</label>
        <div class="grid gap-2">
          ${scenarios.map(renderScenarioButton).join("") || skeletonLine("Loading scenarios")}
        </div>
      </div>

      <div class="space-y-2">
        <label for="prompt" class="text-xs font-semibold uppercase tracking-normal text-neutral-500">Prompt</label>
        <textarea id="prompt" class="h-40 w-full resize-none border border-line bg-field px-3 py-2 text-sm leading-5 outline-none focus:border-ink" spellcheck="false">${escapeHtml(state.prompt)}</textarea>
      </div>

      ${state.activeMode === "monitor" ? renderMonitorControls() : renderRewriteControls()}

      <div class="grid grid-cols-2 gap-3">
        ${numberControl("maxNewTokens", "Max Tokens", state.maxNewTokens, 8, 256, 1)}
        ${numberControl("temperature", "Temperature", state.temperature, 0, 2, 0.05)}
      </div>

      <button id="run" class="flex h-11 w-full items-center justify-center gap-2 bg-ink px-4 text-sm font-semibold text-white disabled:cursor-wait disabled:bg-neutral-500" ${state.loading ? "disabled" : ""}>
        <i data-lucide="${state.loading ? "loader-circle" : "play"}" class="h-4 w-4 ${state.loading ? "animate-spin" : ""}"></i>
        ${state.loading ? "Running" : "Run"}
      </button>

      ${state.error ? `<div class="border border-hazard bg-red-50 px-3 py-2 text-sm text-hazard">${escapeHtml(state.error)}</div>` : ""}
    </div>
  `;
}

function tabButton(mode: RunMode, icon: string, label: string): string {
  const active = state.activeMode === mode;
  return `
    <button data-tab="${mode}" class="flex h-9 items-center justify-center gap-2 text-sm font-semibold ${active ? "bg-white text-ink shadow-crisp" : "text-neutral-600 hover:text-ink"}">
      <i data-lucide="${icon}" class="h-4 w-4"></i>
      ${label}
    </button>
  `;
}

function renderScenarioButton(scenario: ScenarioPublic): string {
  const active = state.prompt === scenario.prompt;
  return `
    <button data-scenario="${scenario.id}" class="border px-3 py-2 text-left text-sm ${active ? "border-ink bg-white" : "border-line bg-field hover:border-neutral-500"}">
      ${escapeHtml(scenario.label)}
    </button>
  `;
}

function renderMonitorControls(): string {
  const concepts = state.profiles?.concepts ?? [];
  return `
    <div class="space-y-2">
      <div class="flex items-center justify-between gap-3">
        <label class="text-xs font-semibold uppercase tracking-normal text-neutral-500">Tripwires</label>
        <div class="grid grid-cols-2 border border-line bg-field p-0.5 text-xs font-semibold">
          ${actionButton("kill", "Kill")}
          ${actionButton("observe", "Observe")}
        </div>
      </div>
      <div class="flex flex-wrap gap-2">
        ${concepts.map(renderConceptChip).join("") || skeletonLine("Loading concepts")}
      </div>
    </div>
    ${rangeControl("threshold", "Threshold", state.threshold, 0.05, 0.99, 0.01)}
  `;
}

function actionButton(action: MonitorAction, label: string): string {
  const active = state.action === action;
  return `
    <button data-action="${action}" class="px-2.5 py-1 ${active ? "bg-white text-ink shadow-crisp" : "text-neutral-600"}">
      ${label}
    </button>
  `;
}

function renderConceptChip(concept: ConceptPublic): string {
  const selected = state.monitorIds.has(concept.id);
  const tone =
    concept.polarity === "risk"
      ? selected
        ? "border-hazard bg-red-50 text-hazard"
        : "border-line bg-field text-neutral-700"
      : selected
        ? "border-signal bg-teal-50 text-signal"
        : "border-line bg-field text-neutral-700";
  return `
    <button data-concept="${concept.id}" class="inline-flex min-h-8 items-center gap-1.5 border px-2.5 py-1 text-xs font-semibold ${tone}">
      <span class="h-1.5 w-1.5 ${concept.polarity === "risk" ? "bg-hazard" : "bg-signal"}"></span>
      ${escapeHtml(concept.label)}
    </button>
  `;
}

function renderRewriteControls(): string {
  const profiles = state.profiles?.rewrite_profiles ?? [];
  return `
    <div class="space-y-2">
      <label class="text-xs font-semibold uppercase tracking-normal text-neutral-500">Rewrite Profile</label>
      <div class="grid gap-2">
        ${profiles.map(renderProfileButton).join("") || skeletonLine("Loading profiles")}
      </div>
    </div>
    ${rangeControl("threshold", "Rewrite Band", state.threshold, 0.05, 0.99, 0.01)}
    ${rangeControl("strength", "Vector Strength", state.strength, 0, 12, 0.1)}
  `;
}

function renderProfileButton(profile: RewriteProfilePublic): string {
  const active = state.rewriteProfileId === profile.id;
  return `
    <button data-profile="${profile.id}" class="border px-3 py-2 text-left ${active ? "border-ink bg-white" : "border-line bg-field hover:border-neutral-500"}">
      <span class="block text-sm font-semibold">${escapeHtml(profile.label)}</span>
      <span class="block truncate text-xs text-neutral-600">${escapeHtml(profile.target_label)}</span>
      <span class="mt-2 flex flex-wrap gap-1">
        ${profile.control_phrases
          .slice(0, 3)
          .map((phrase) => `<span class="border border-line bg-white px-1.5 py-0.5 text-[11px] text-neutral-600">${escapeHtml(phrase)}</span>`)
          .join("")}
      </span>
    </button>
  `;
}

function rangeControl(id: "threshold" | "strength", label: string, value: number, min: number, max: number, step: number): string {
  return `
    <div class="space-y-2">
      <div class="flex items-center justify-between gap-3">
        <label for="${id}" class="text-xs font-semibold uppercase tracking-normal text-neutral-500">${label}</label>
        <span class="font-mono text-xs">${value.toFixed(2)}</span>
      </div>
      <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" class="w-full accent-ink" />
    </div>
  `;
}

function numberControl(id: "maxNewTokens" | "temperature", label: string, value: number, min: number, max: number, step: number): string {
  return `
    <label class="space-y-1">
      <span class="text-xs font-semibold uppercase tracking-normal text-neutral-500">${label}</span>
      <input id="${id}" type="number" min="${min}" max="${max}" step="${step}" value="${value}" class="h-10 w-full border border-line bg-field px-2 text-sm outline-none focus:border-ink" />
    </label>
  `;
}

function skeletonLine(label: string): string {
  return `<div class="border border-line bg-field px-3 py-2 text-sm text-neutral-500">${escapeHtml(label)}</div>`;
}

function renderResults(): string {
  if (!state.result) {
    return `
      <div class="grid min-h-[calc(100vh-146px)] place-items-center p-6">
        <div class="w-full max-w-3xl border border-line bg-field p-6">
          <div class="mb-5 flex items-center gap-3">
            <div class="flex h-11 w-11 items-center justify-center border border-line bg-white">
              <i data-lucide="scan-eye" class="h-5 w-5"></i>
            </div>
            <div>
              <h2 class="text-lg font-semibold">Ready</h2>
              <p class="text-sm text-neutral-600">Modal/Qwen endpoint idle.</p>
            </div>
          </div>
          <div class="activation-grid text-xs text-neutral-500">
            <span>step</span><span>risk</span><span>claim</span><span>brand</span><span>tone</span><span>evidence</span><span>action</span>
            ${Array.from({ length: 7 })
              .map(
                (_, index) => `
                  <span class="font-mono">${index}</span>
                  ${Array.from({ length: 5 })
                    .map(() => `<span class="activation-cell" style="--score: 0.14; --cell-color: rgba(15,118,110,0.42);"></span>`)
                    .join("")}
                  <span class="text-neutral-400">observe</span>
                `
              )
              .join("")}
          </div>
        </div>
      </div>
    `;
  }

  const response = state.result;
  const visibleFrames = response.frames.slice(0, state.replayCount);
  return `
    <div class="border-b border-line p-4">
      <div class="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
        <div class="min-w-0">
          <div class="mb-2 flex flex-wrap items-center gap-2">
            ${statusBadge(response.status)}
            ${metricPill("Latency", `${response.metrics.latency_ms} ms`)}
            ${metricPill("Peak", response.metrics.peak_score.toFixed(2))}
            ${metricPill("Rewrites", String(response.metrics.rewrites))}
          </div>
          <h2 class="text-lg font-semibold">${escapeHtml(response.summary)}</h2>
        </div>
        <button id="replay" class="inline-flex h-9 items-center justify-center gap-2 border border-line bg-field px-3 text-sm font-semibold hover:border-ink">
          <i data-lucide="refresh-cw" class="h-4 w-4"></i>
          Replay
        </button>
      </div>
    </div>
    <div class="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_430px]">
      <div class="min-w-0 space-y-4">
        ${renderImpactPanel(response, visibleFrames)}
        ${response.mode === "rewrite" ? renderRewriteOutputs(response) : renderMonitorOutput(response)}
        ${renderTrace(visibleFrames, "Activation Trace")}
      </div>
      <div class="min-w-0 space-y-4">
        ${renderInterventions(response)}
        ${renderReadoutPanel(visibleFrames)}
      </div>
    </div>
  `;
}

function renderImpactPanel(response: RunResponse, frames: TraceFrame[]): string {
  const latest = frames.at(-1) ?? response.frames.at(-1);
  const top = latest?.top_concepts[0];
  const topRisk = latest?.top_concepts.find((concept) => concept.polarity === "risk");
  const topTarget = latest?.top_concepts.find((concept) => concept.polarity === "target");
  const action =
    response.status === "blocked"
      ? "Stopped before first token"
      : response.status === "rewritten"
        ? `${response.metrics.rewrites} latent edits`
        : "Watched through decode";
  const result =
    response.status === "blocked"
      ? "No unsafe completion released"
      : response.status === "rewritten"
        ? "Enterprise policy attractor active"
        : "Response allowed";
  return `
    <section class="grid gap-3 lg:grid-cols-3">
      ${impactCard("Latent Concern", topRisk?.label ?? top?.label ?? "None", topRisk?.score ?? top?.score ?? 0, topRisk?.polarity ?? top?.polarity ?? "neutral")}
      ${impactCard("Control Action", action, response.metrics.rewrites ? Math.min(1, response.metrics.rewrites / 12) : response.status === "blocked" ? 1 : 0.35, response.status === "blocked" ? "risk" : "target")}
      ${impactCard("Enterprise Result", topTarget?.label ?? result, topTarget?.score ?? 0.8, "target")}
    </section>
  `;
}

function impactCard(label: string, value: string, score: number, polarity: ConceptPolarity): string {
  const color = polarity === "risk" ? "bg-hazard" : polarity === "target" ? "bg-signal" : "bg-neutral-500";
  return `
    <article class="border border-line bg-field p-3">
      <div class="mb-2 text-xs font-semibold uppercase tracking-normal text-neutral-500">${escapeHtml(label)}</div>
      <div class="min-h-10 text-sm font-semibold leading-5">${escapeHtml(value)}</div>
      <div class="mt-3 h-2 border border-line bg-white">
        <div class="h-full ${color}" style="width: ${Math.round(Math.max(0.05, Math.min(1, score)) * 100)}%"></div>
      </div>
    </article>
  `;
}

function statusBadge(status: RunStatus): string {
  const styles: Record<RunStatus, string> = {
    completed: "border-signal bg-teal-50 text-signal",
    blocked: "border-hazard bg-red-50 text-hazard",
    rewritten: "border-violetless bg-sky-50 text-violetless",
    error: "border-hazard bg-red-50 text-hazard"
  };
  return `<span class="inline-flex items-center gap-1 border px-2.5 py-1 text-xs font-semibold ${styles[status]}">${escapeHtml(status.toUpperCase())}</span>`;
}

function renderMonitorOutput(response: RunResponse): string {
  if (response.status === "blocked") {
    return `
      <section class="border border-hazard bg-red-50 p-4">
        <div class="flex items-center gap-3">
          <i data-lucide="octagon-x" class="h-5 w-5 text-hazard"></i>
          <div>
            <h3 class="font-semibold text-hazard">Run Killed</h3>
            <p class="text-sm text-red-900">No completion tokens were released.</p>
          </div>
        </div>
      </section>
    `;
  }
  return `
    <section class="border border-line bg-field p-4">
      <h3 class="mb-2 text-sm font-semibold uppercase tracking-normal text-neutral-500">Output</h3>
      <p class="whitespace-pre-wrap text-sm leading-6">${escapeHtml(response.output)}</p>
    </section>
  `;
}

function renderRewriteOutputs(response: RunResponse): string {
  return `
    <section class="grid gap-3 lg:grid-cols-2">
      <div class="border border-line bg-field p-4">
        <div class="mb-2 flex items-center gap-2 text-sm font-semibold">
          <i data-lucide="route-off" class="h-4 w-4"></i>
          Raw
        </div>
        <p class="whitespace-pre-wrap text-sm leading-6 text-neutral-700">${escapeHtml(response.baseline_output ?? "Baseline comparison disabled.")}</p>
      </div>
      <div class="border border-violetless bg-sky-50 p-4">
        <div class="mb-2 flex items-center gap-2 text-sm font-semibold text-violetless">
          <i data-lucide="route" class="h-4 w-4"></i>
          Steered
        </div>
        <p class="whitespace-pre-wrap text-sm leading-6">${escapeHtml(response.output)}</p>
      </div>
    </section>
  `;
}

function renderTrace(frames: TraceFrame[], title: string): string {
  const rows = frames.slice(0, 36);
  return `
    <section class="border border-line">
      <div class="flex items-center justify-between border-b border-line bg-field px-3 py-2">
        <h3 class="text-sm font-semibold">${title}</h3>
        <span class="font-mono text-xs text-neutral-500">${rows.length} frames</span>
      </div>
      <div class="trace-scroll overflow-x-auto p-3">
        <div class="activation-grid min-w-[720px] text-xs">
          <span class="font-semibold text-neutral-500">step</span>
          <span class="font-semibold text-neutral-500">top</span>
          <span class="font-semibold text-neutral-500">next</span>
          <span class="font-semibold text-neutral-500">third</span>
          <span class="font-semibold text-neutral-500">fourth</span>
          <span class="font-semibold text-neutral-500">fifth</span>
          <span class="font-semibold text-neutral-500">action</span>
          ${rows.map(renderTraceRow).join("")}
        </div>
      </div>
    </section>
  `;
}

function renderTraceRow(frame: TraceFrame): string {
  const cells = Array.from({ length: 5 }).map((_, index) => {
    const score = frame.top_concepts[index];
    if (!score) {
      return `<span class="activation-cell" style="--score: 0; --cell-color: rgba(115,115,115,0.2);" title=""></span>`;
    }
    return `
      <span class="activation-cell" style="--score: ${score.score}; --cell-color: ${cellColor(score)};" title="${escapeHtml(score.label)} ${score.score.toFixed(2)}">
        <span class="sr-only">${escapeHtml(score.label)} ${score.score.toFixed(2)}</span>
      </span>
    `;
  });
  return `
    <span class="font-mono text-neutral-600">${frame.step}</span>
    ${cells.join("")}
    <span class="truncate font-mono ${actionTextClass(frame.action)}">${escapeHtml(frame.action)}</span>
  `;
}

function cellColor(score: ConceptScore): string {
  if (score.polarity === "risk") return "rgba(180,35,24,0.82)";
  if (score.polarity === "target") return "rgba(15,118,110,0.82)";
  return "rgba(47,111,159,0.7)";
}

function actionTextClass(action: TraceFrame["action"]): string {
  if (action === "blocked") return "text-hazard";
  if (action === "rewrite") return "text-violetless";
  return "text-neutral-500";
}

function renderInterventions(response: RunResponse): string {
  if (response.interventions.length === 0) {
    return `
      <section class="border border-line bg-field p-4">
        <div class="mb-2 flex items-center gap-2 text-sm font-semibold">
          <i data-lucide="scissors-line-dashed" class="h-4 w-4"></i>
          Intervention Ledger
        </div>
        <p class="text-sm text-neutral-600">No rewrites recorded.</p>
      </section>
    `;
  }
  return `
    <section class="border border-line">
      <div class="border-b border-line bg-field px-3 py-2">
        <h3 class="flex items-center gap-2 text-sm font-semibold">
          <i data-lucide="scissors-line-dashed" class="h-4 w-4"></i>
          Intervention Ledger
        </h3>
      </div>
      <div class="divide-y divide-line">
        ${response.interventions
          .slice(0, 8)
          .map(
            (intervention) => `
              <article class="p-3">
                <div class="mb-1 flex items-center justify-between gap-3">
                  <span class="font-mono text-xs text-neutral-500">step ${intervention.step}</span>
                  <span class="font-mono text-xs">${intervention.strength.toFixed(2)}</span>
                </div>
                <p class="text-sm font-semibold">${escapeHtml(intervention.from_concept)} -&gt; ${escapeHtml(intervention.to_concept)}</p>
                <p class="mt-1 text-xs leading-5 text-neutral-600">${escapeHtml(intervention.reason)}</p>
              </article>
            `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderReadoutPanel(frames: TraceFrame[]): string {
  const latest = frames.at(-1);
  const concepts = latest?.top_concepts ?? [];
  return `
    <section class="border border-line">
      <div class="border-b border-line bg-field px-3 py-2">
        <h3 class="flex items-center gap-2 text-sm font-semibold">
          <i data-lucide="radio-tower" class="h-4 w-4"></i>
          Readout
        </h3>
      </div>
      <div class="divide-y divide-line">
        ${
          concepts.length
            ? concepts
                .map(
                  (concept) => `
                    <div class="p-3">
                      <div class="mb-2 flex items-center justify-between gap-3">
                        <span class="text-sm font-semibold">${escapeHtml(concept.label)}</span>
                        <span class="font-mono text-xs">${concept.score.toFixed(2)}</span>
                      </div>
                      <div class="h-2 border border-line bg-white">
                        <div class="h-full ${concept.polarity === "risk" ? "bg-hazard" : "bg-signal"}" style="width: ${Math.round(concept.score * 100)}%"></div>
                      </div>
                      <div class="mt-2 flex flex-wrap gap-1">
                        ${concept.evidence.map((item) => `<span class="border border-line bg-field px-1.5 py-0.5 text-[11px] text-neutral-600">${escapeHtml(item)}</span>`).join("")}
                      </div>
                    </div>
                  `
                )
                .join("")
            : `<div class="p-3 text-sm text-neutral-600">No frames yet.</div>`
        }
      </div>
    </section>
  `;
}

function bindEvents() {
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeMode = button.dataset.tab as RunMode;
      const firstScenario = state.profiles?.scenarios.find((scenario) => scenario.mode === state.activeMode);
      if (firstScenario) applyScenario(firstScenario);
      state.result = null;
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      const scenario = state.profiles?.scenarios.find((item) => item.id === button.dataset.scenario);
      if (scenario) {
        applyScenario(scenario);
        state.result = null;
        render();
      }
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-concept]").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.dataset.concept;
      if (!id) return;
      if (state.monitorIds.has(id)) {
        state.monitorIds.delete(id);
      } else {
        state.monitorIds.add(id);
      }
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      state.action = button.dataset.action as MonitorAction;
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-profile]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.profile) {
        state.rewriteProfileId = button.dataset.profile;
        const profile = state.profiles?.rewrite_profiles.find((item) => item.id === state.rewriteProfileId);
        if (profile) state.strength = profile.default_strength;
        render();
      }
    });
  });

  bindInput("prompt", (element) => {
    state.prompt = (element as HTMLTextAreaElement).value;
  });
  bindInput("threshold", (element) => {
    state.threshold = Number((element as HTMLInputElement).value);
    render();
  });
  bindInput("strength", (element) => {
    state.strength = Number((element as HTMLInputElement).value);
    render();
  });
  bindInput("maxNewTokens", (element) => {
    state.maxNewTokens = Number((element as HTMLInputElement).value);
  });
  bindInput("temperature", (element) => {
    state.temperature = Number((element as HTMLInputElement).value);
  });

  document.querySelector<HTMLButtonElement>("#run")?.addEventListener("click", run);
  document.querySelector<HTMLButtonElement>("#replay")?.addEventListener("click", () => {
    if (state.result) startReplay();
  });
}

function bindInput(id: string, handler: (element: HTMLElement) => void) {
  const element = document.getElementById(id);
  if (!element) return;
  element.addEventListener("input", () => handler(element));
}

function applyScenario(scenario: ScenarioPublic) {
  state.activeMode = scenario.mode;
  state.prompt = scenario.prompt;
  state.action = scenario.action;
  state.threshold = scenario.threshold;
  if (scenario.monitor_ids?.length) {
    state.monitorIds = new Set(scenario.monitor_ids);
  }
  if (scenario.rewrite_profile_id) {
    state.rewriteProfileId = scenario.rewrite_profile_id;
    const profile = state.profiles?.rewrite_profiles.find((item) => item.id === scenario.rewrite_profile_id);
    if (profile) state.strength = profile.default_strength;
  }
}

async function run() {
  state.loading = true;
  state.error = null;
  state.result = null;
  stopReplay();
  render();

  try {
    const result = await fetchJson<RunResponse>("/api/run", {
      method: "POST",
      body: JSON.stringify({
        mode: state.activeMode,
        prompt: state.prompt,
        monitor_ids: Array.from(state.monitorIds),
        rewrite_profile_id: state.rewriteProfileId,
        action: state.action,
        threshold: state.threshold,
        max_new_tokens: state.maxNewTokens,
        temperature: state.temperature,
        strength: state.strength,
        compare_baseline: true
      })
    });
    state.result = result;
    state.replayCount = 0;
    startReplay();
  } catch (error) {
    state.error = error instanceof Error ? error.message : "Run failed";
  } finally {
    state.loading = false;
    render();
  }
}

function startReplay() {
  stopReplay();
  if (!state.result) return;
  state.replayCount = 0;
  replayTimer = window.setInterval(() => {
    if (!state.result) {
      stopReplay();
      return;
    }
    state.replayCount += 1;
    if (state.replayCount >= state.result.frames.length) {
      state.replayCount = state.result.frames.length;
      stopReplay();
    }
    render();
  }, 90);
}

function stopReplay() {
  if (replayTimer !== null) {
    window.clearInterval(replayTimer);
    replayTimer = null;
  }
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadInitialData();

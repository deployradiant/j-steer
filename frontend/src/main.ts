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
type AppMode = RunMode | "custom";
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

interface LayerActivation {
  layer: number;
  concept_id: string;
  label: string;
  score: number;
  polarity: ConceptPolarity;
}

interface JacobianLensInfo {
  available: boolean;
  status: string;
  source: string;
  artifact: string;
  target_model: string;
  metadata_model: string | null;
  layer: number | null;
  message: string | null;
}

interface TraceFrame {
  step: number;
  phase: "prompt" | "decode";
  layer: number;
  token: string;
  generated_text: string;
  top_concepts: ConceptScore[];
  proxy_concepts: ConceptScore[];
  jacobian_concepts: ConceptScore[];
  jacobian_tokens: string[];
  layer_activations: LayerActivation[];
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
    jacobian_lens: JacobianLensInfo | null;
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

interface CustomConceptInput {
  label: string;
  keywords: string[];
  polarity: ConceptPolarity;
}

interface CustomConceptLens {
  id: string;
  label: string;
  polarity: ConceptPolarity;
  keywords: string[];
  calibration_prompts: string[];
  proxy_score: number;
  jacobian_score: number;
  agreement_delta: number;
}

interface CustomLensResponse {
  id: string;
  prompt: string;
  summary: string;
  engine: RunResponse["engine"];
  frame: TraceFrame;
  concepts: CustomConceptLens[];
  metrics: RunResponse["metrics"];
}

interface HealthResponse {
  ok: boolean;
  engine: RunResponse["engine"];
}

interface State {
  activeMode: AppMode;
  prompt: string;
  monitorIds: Set<string>;
  rewriteProfileId: string;
  customConcepts: CustomConceptInput[];
  customWindowTokens: number;
  action: MonitorAction;
  threshold: number;
  maxNewTokens: number;
  temperature: number;
  strength: number;
  profiles: ProfilesResponse | null;
  health: HealthResponse | null;
  result: RunResponse | null;
  customResult: CustomLensResponse | null;
  replayCount: number;
  inspectedStep: number | null;
  loading: boolean;
  error: string | null;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const CUSTOM_DEFAULT_PROMPT =
  "A regulated enterprise buyer asks whether HeliosCRM is safe enough for support operations. Their notes mention SOC 2, audit logs, SSO, private customer records, pricing approval, procurement, and Salesforce alternatives.";
let replayTimer: number | null = null;

const state: State = {
  activeMode: "custom",
  prompt: CUSTOM_DEFAULT_PROMPT,
  monitorIds: new Set(["credential_exfiltration", "data_leakage", "unsupported_claim"]),
  rewriteProfileId: "brand_canon_injection",
  customConcepts: [
    {
      label: "Security Assurance",
      keywords: ["SOC 2", "audit logs", "SSO", "private customer records"],
      polarity: "target"
    },
    {
      label: "Buying Intent",
      keywords: ["pricing", "budget", "approval", "procurement"],
      polarity: "target"
    },
    {
      label: "Competitor Drift",
      keywords: ["Salesforce", "HubSpot", "Zendesk", "alternative"],
      polarity: "risk"
    }
  ],
  customWindowTokens: 64,
  action: "kill",
  threshold: 0.62,
  maxNewTokens: 90,
  temperature: 0.7,
  strength: 8.0,
  profiles: null,
  health: null,
  result: null,
  customResult: null,
  replayCount: 999,
  inspectedStep: null,
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
    <main class="radiant-shell min-h-screen bg-black text-white">
      <header class="radiant-header border-b border-line bg-panel">
        <div class="mx-auto flex max-w-[1500px] px-4 py-4 sm:px-6">
          <div class="radiant-brand flex min-w-0 items-center gap-3">
            <div class="radiant-logo-mark flex h-12 w-12 shrink-0 items-center justify-center">
              <img src="/radiant-mark.png" alt="Radiant" class="h-9 w-9" />
            </div>
            <div class="min-w-0">
              <h1 class="text-2xl font-semibold tracking-normal">J-Steer</h1>
              <p class="truncate text-sm text-neutral-600">${escapeHtml(engineLine())}</p>
            </div>
          </div>
        </div>
      </header>

      <section class="mx-auto grid max-w-[1500px] gap-4 px-4 py-4 sm:px-6 lg:grid-cols-[280px_minmax(0,1fr)_340px] 2xl:grid-cols-[300px_minmax(0,1fr)_390px]">
        <aside class="min-w-0 border border-line bg-panel shadow-crisp lg:sticky lg:top-4 lg:self-start">
          ${renderNavigationSidebar()}
        </aside>
        <section class="min-w-0 space-y-4">
          ${renderPromptWorkspace()}
        </section>
        <aside class="min-w-0 lg:sticky lg:top-4 lg:max-h-[calc(100vh-112px)] lg:overflow-auto">
          ${renderModelStateRail()}
        </aside>
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
  return "Monitor Qwen activations, flag unsafe concepts, and compare custom concept lenses.";
}

function metricPill(label: string, value: string): string {
  return `
    <span class="inline-flex items-center gap-1 border border-line bg-white px-2.5 py-1">
      <span class="text-neutral-500">${escapeHtml(label)}</span>
      <span>${escapeHtml(value)}</span>
    </span>
  `;
}

function lensStatusText(): string {
  const lens = state.health?.engine.jacobian_lens;
  if (!lens) return "Pending";
  if (lens.status === "mock") return "Mock";
  if (lens.available) return "Ready";
  if (lens.status === "unavailable") return "Unavailable";
  return "Configured";
}

function lensPairText(): string {
  const lens = state.health?.engine.jacobian_lens;
  if (!state.health || !lens?.target_model) return "Pending";
  const model = normalizeModelId(state.health.engine.model);
  const target = normalizeModelId(lens.target_model);
  const metadata = lens.metadata_model ? normalizeModelId(lens.metadata_model) : target;
  return model === target && target === metadata ? "Matched" : "Mismatch";
}

function normalizeModelId(value: string): string {
  return value.trim().replace(/\/+$/, "").toLowerCase();
}

function policyLabel(): string {
  if (state.activeMode === "monitor") return "Tripwire";
  if (state.activeMode === "rewrite") return "Rewrite";
  return "Custom Lens";
}

function renderNavigationSidebar(): string {
  const profiles = state.profiles;
  const scenarios = isRunMode(state.activeMode)
    ? (profiles?.scenarios.filter((scenario) => scenario.mode === state.activeMode) ?? [])
    : [];
  return `
    <div class="border-b border-line p-4">
      <div class="mb-2 text-xs font-semibold uppercase tracking-normal text-neutral-500">Mode</div>
      <div class="grid gap-2">
        ${tabButton("monitor", "shield-alert", "Tripwire")}
        ${tabButton("rewrite", "wand-sparkles", "Rewrite")}
        ${tabButton("custom", "radio-tower", "Custom Lens")}
      </div>
    </div>

    <div class="p-4">
      <div class="mb-2 text-xs font-semibold uppercase tracking-normal text-neutral-500">Scenarios</div>
      <div class="grid gap-2">
        ${state.activeMode === "custom" ? renderCustomScenarioButton() : scenarios.map(renderScenarioButton).join("") || skeletonLine("Loading scenarios")}
      </div>
    </div>
  `;
}

function renderPromptWorkspace(): string {
  return `
    <section class="border border-line bg-panel shadow-crisp">
      <div class="border-b border-line p-4">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div class="min-w-0">
            <div class="mb-1 text-xs font-semibold uppercase tracking-normal text-neutral-500">Prompt</div>
            <h2 class="text-lg font-semibold">${escapeHtml(promptWorkspaceTitle())}</h2>
          </div>
          <span class="inline-flex w-fit items-center border border-line bg-white px-2.5 py-1 text-xs font-semibold">${escapeHtml(policyLabel())}</span>
        </div>
      </div>

      <div class="space-y-4 p-4">
        <textarea id="prompt" class="h-48 w-full resize-none border border-line bg-field px-3 py-3 text-sm leading-6 outline-none focus:border-ink lg:h-56" spellcheck="false">${escapeHtml(state.prompt)}</textarea>
        ${renderRunControls()}
      </div>
    </section>

    <section class="border border-line bg-panel shadow-crisp">
      ${renderResults()}
    </section>
  `;
}

function renderRunControls(): string {
  const controls =
    state.activeMode === "monitor"
      ? renderMonitorControls()
      : state.activeMode === "rewrite"
        ? renderRewriteControls()
        : renderCustomControls();
  const numericControls =
    state.activeMode === "custom"
      ? numberControl("customWindowTokens", "Window", state.customWindowTokens, 8, 128, 1)
      : `<div class="grid grid-cols-2 gap-3">
          ${numberControl("maxNewTokens", "Max Tokens", state.maxNewTokens, 8, 256, 1)}
          ${numberControl("temperature", "Temperature", state.temperature, 0, 2, 0.05)}
        </div>`;

  return `
    <div class="grid gap-4 xl:grid-cols-[minmax(0,1fr)_240px]">
      <div class="min-w-0 space-y-4">
        ${controls}
      </div>
      <div class="space-y-4">
        ${numericControls}
        <button id="run" class="flex h-11 w-full items-center justify-center gap-2 bg-ink px-4 text-sm font-semibold text-white disabled:cursor-wait disabled:bg-neutral-500" ${state.loading ? "disabled" : ""}>
          <i data-lucide="${state.loading ? "loader-circle" : "play"}" class="h-4 w-4 ${state.loading ? "animate-spin" : ""}"></i>
          ${state.loading ? "Running" : state.activeMode === "custom" ? "Generate Lens" : "Run"}
        </button>
        ${state.error ? `<div class="border border-hazard bg-red-50 px-3 py-2 text-sm text-hazard">${escapeHtml(state.error)}</div>` : ""}
      </div>
    </div>
  `;
}

function promptWorkspaceTitle(): string {
  if (state.activeMode === "monitor") return "Tripwire scenario prompt";
  if (state.activeMode === "rewrite") return "Rewrite scenario prompt";
  return "Custom concept lens prompt";
}

function renderCustomScenarioButton(): string {
  return `
    <button data-tab="custom" class="border border-ink bg-white px-3 py-2 text-left">
      <span class="block text-sm font-semibold">Enterprise buyer lens</span>
      <span class="mt-1 block text-xs leading-4 text-neutral-600">Security, buying intent, competitor drift.</span>
    </button>
  `;
}

function renderModelStateRail(): string {
  if (state.result) {
    const visibleFrames = state.result.frames.slice(0, state.replayCount);
    const inspectedFrame = selectedTraceFrame(visibleFrames);
    return `
      <div class="space-y-4">
        ${renderModelStatePanel()}
        ${renderInterventions(state.result)}
        ${renderTokenInspector(inspectedFrame, state.result)}
        ${renderReadoutPanel(inspectedFrame ? [inspectedFrame] : visibleFrames)}
      </div>
    `;
  }

  if (state.customResult) {
    return `
      <div class="space-y-4">
        ${renderModelStatePanel()}
        ${renderTokenInspector(state.customResult.frame, state.customResult)}
        ${renderReadoutPanel([state.customResult.frame])}
      </div>
    `;
  }

  return `
    <div class="space-y-4">
      ${renderModelStatePanel()}
      ${renderModelStateEmpty()}
    </div>
  `;
}

function renderModelStatePanel(): string {
  const engine = state.health?.engine ?? state.result?.engine ?? state.customResult?.engine;
  const lens = engine?.jacobian_lens;
  const runtime = engine?.mode === "hf" ? "Modal/Qwen" : (engine?.mode ?? "Loading");
  const lensSource = lens?.source ? `${lens.source}${lens.artifact ? ` / ${lens.artifact}` : ""}` : "Pending";
  return `
    <section class="border border-line bg-panel shadow-crisp">
      <div class="border-b border-line bg-field px-3 py-2">
        <h3 class="flex items-center gap-2 text-sm font-semibold">
          <i data-lucide="server" class="h-4 w-4"></i>
          Model State
        </h3>
      </div>
      <div class="space-y-2 p-3 text-xs">
        ${stateValueRow("Runtime", runtime)}
        ${stateValueRow("Model", engine?.model ?? "Qwen/Qwen3-1.7B")}
        ${stateValueRow("Readout", engine?.readout ?? "hidden-state")}
        ${stateValueRow("J-Scope Lens", lensStatusText())}
        ${stateValueRow("Lens Pair", lensPairText())}
        ${stateValueRow("Lens Source", lensSource)}
        ${stateValueRow("Layer", engine?.layer.toString() ?? "15")}
        ${stateValueRow("Device", engine?.device ?? "Modal GPU")}
        ${stateValueRow("Policy", policyLabel(), true)}
      </div>
    </section>
  `;
}

function stateValueRow(label: string, value: string, emphasize = false): string {
  return `
    <div class="grid grid-cols-[92px_minmax(0,1fr)] items-center gap-2 border border-line bg-white px-2 py-1.5">
      <span class="text-neutral-500">${escapeHtml(label)}</span>
      <span class="truncate text-right ${emphasize ? "font-semibold text-white" : "text-neutral-700"}" title="${escapeHtml(value)}">${escapeHtml(value)}</span>
    </div>
  `;
}

function renderModelStateEmpty(): string {
  return `
    <section class="border border-line bg-panel shadow-crisp">
      <div class="border-b border-line bg-field px-3 py-2">
        <h3 class="text-sm font-semibold">Live Readout</h3>
      </div>
      <div class="p-3">
        <div class="grid items-center gap-1 text-xs text-neutral-500" style="grid-template-columns: 38px repeat(3, minmax(0, 1fr)) 50px;">
          <span>step</span><span>risk</span><span>claim</span><span>brand</span><span>action</span>
          ${Array.from({ length: 4 })
            .map(
              (_, index) => `
                <span class="font-mono">${index}</span>
                ${Array.from({ length: 3 })
                  .map(() => `<span class="activation-cell" style="--score: 0.14; --cell-color: rgba(66,96,255,0.42);"></span>`)
                  .join("")}
                <span class="truncate text-neutral-400">idle</span>
              `
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function isRunMode(mode: AppMode): mode is RunMode {
  return mode === "monitor" || mode === "rewrite";
}

function tabButton(mode: AppMode, icon: string, label: string): string {
  const active = state.activeMode === mode;
  const descriptions: Record<AppMode, string> = {
    monitor: "Detect and kill unsafe latent trajectories.",
    rewrite: "Apply activation edits during decoding.",
    custom: "Generate concept lenses and compare readouts."
  };
  return `
    <button data-tab="${mode}" class="flex min-h-16 items-start gap-3 border px-3 py-3 text-left ${active ? "border-ink bg-white text-ink shadow-crisp" : "border-line bg-field text-neutral-600 hover:border-ink hover:text-ink"}">
      <i data-lucide="${icon}" class="mt-0.5 h-4 w-4 shrink-0"></i>
      <span class="min-w-0">
        <span class="block text-sm font-semibold">${label}</span>
        <span class="mt-1 block text-xs leading-4 text-neutral-600">${descriptions[mode]}</span>
      </span>
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

function renderCustomControls(): string {
  return `
    <div class="space-y-2">
      <label class="text-xs font-semibold uppercase tracking-normal text-neutral-500">Generated Concepts</label>
      <div class="grid gap-3">
        ${state.customConcepts.map(renderCustomConceptEditor).join("")}
      </div>
    </div>
  `;
}

function renderCustomConceptEditor(concept: CustomConceptInput, index: number): string {
  return `
    <div class="border border-line bg-field p-3">
      <div class="mb-2 grid grid-cols-[minmax(0,1fr)_112px] gap-2">
        <input data-custom-label="${index}" value="${escapeHtml(concept.label)}" class="h-9 min-w-0 border border-line bg-white px-2 text-sm outline-none focus:border-ink" />
        <select data-custom-polarity="${index}" class="h-9 border border-line bg-white px-2 text-xs font-semibold outline-none focus:border-ink">
          ${polarityOption("target", concept.polarity)}
          ${polarityOption("risk", concept.polarity)}
          ${polarityOption("neutral", concept.polarity)}
        </select>
      </div>
      <textarea data-custom-keywords="${index}" class="h-16 w-full resize-none border border-line bg-white px-2 py-1.5 text-xs leading-5 outline-none focus:border-ink" spellcheck="false">${escapeHtml(concept.keywords.join(", "))}</textarea>
    </div>
  `;
}

function polarityOption(value: ConceptPolarity, current: ConceptPolarity): string {
  return `<option value="${value}" ${value === current ? "selected" : ""}>${escapeHtml(value)}</option>`;
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

function numberControl(
  id: "maxNewTokens" | "temperature" | "customWindowTokens",
  label: string,
  value: number,
  min: number,
  max: number,
  step: number
): string {
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
  if (state.activeMode === "custom") {
    return renderCustomResults();
  }
  if (!state.result) {
    return `
      <div class="grid min-h-[320px] place-items-center p-6">
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
  const inspectedFrame = selectedTraceFrame(visibleFrames);
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
    <div class="space-y-4 p-4">
      ${renderImpactPanel(response, visibleFrames)}
      ${response.mode === "rewrite" ? renderRewriteOutputs(response, visibleFrames, inspectedFrame) : renderMonitorOutput(response, visibleFrames, inspectedFrame)}
      ${renderActivationExplorer(response, visibleFrames, inspectedFrame)}
    </div>
  `;
}

function renderCustomResults(): string {
  if (!state.customResult) {
    return `
      <div class="grid min-h-[320px] place-items-center p-6">
        <div class="w-full max-w-3xl border border-line bg-field p-6">
          <div class="mb-5 flex items-center gap-3">
            <div class="flex h-11 w-11 items-center justify-center border border-line bg-white">
              <i data-lucide="radio-tower" class="h-5 w-5"></i>
            </div>
            <div>
              <h2 class="text-lg font-semibold">Custom Lens</h2>
              <p class="text-sm text-neutral-600">Generate a concept lens, then compare it with the matched Qwen3-1.7B J-lens.</p>
            </div>
          </div>
          <div class="grid gap-3 md:grid-cols-3">
            ${state.customConcepts
              .map(
                (concept) => `
                  <div class="border border-line bg-white p-3">
                    <div class="mb-2 text-sm font-semibold">${escapeHtml(concept.label)}</div>
                    <div class="flex flex-wrap gap-1">
                      ${concept.keywords.map((keyword) => `<span class="border border-line bg-field px-1.5 py-0.5 text-[11px] text-neutral-600">${escapeHtml(keyword)}</span>`).join("")}
                    </div>
                  </div>
                `
              )
              .join("")}
          </div>
        </div>
      </div>
    `;
  }

  const response = state.customResult;
  return `
    <div class="border-b border-line p-4">
      <div class="mb-2 flex flex-wrap items-center gap-2">
        ${metricPill("Latency", `${response.metrics.latency_ms} ms`)}
        ${metricPill("Peak", response.metrics.peak_score.toFixed(2))}
        ${metricPill("Concepts", String(response.concepts.length))}
        ${metricPill("Lens Pair", lensPairText())}
      </div>
      <h2 class="text-lg font-semibold">${escapeHtml(response.summary)}</h2>
    </div>
    <div class="p-4">
      <section class="grid gap-3 lg:grid-cols-2">
        ${response.concepts.map(renderCustomLensCard).join("")}
      </section>
    </div>
  `;
}

function renderCustomLensCard(concept: CustomConceptLens): string {
  const dominant = concept.jacobian_score >= concept.proxy_score ? concept.jacobian_score : concept.proxy_score;
  return `
    <article class="border border-line bg-field p-3">
      <div class="mb-2 flex items-start justify-between gap-3">
        <div class="min-w-0">
          <h3 class="truncate text-sm font-semibold">${escapeHtml(concept.label)}</h3>
          <p class="font-mono text-[11px] text-neutral-500">delta ${concept.agreement_delta.toFixed(2)}</p>
        </div>
        <span class="border border-line bg-white px-1.5 py-0.5 text-[11px] font-semibold">${escapeHtml(concept.polarity)}</span>
      </div>
      <div class="mb-3 grid grid-cols-2 gap-2 text-xs">
        ${miniScore("Proxy", concept.proxy_score, concept.polarity)}
        ${miniScore("Jacobian", concept.jacobian_score, concept.polarity)}
      </div>
      <div class="mb-3 flex flex-wrap gap-1">
        ${concept.keywords.map((keyword) => `<span class="border border-line bg-white px-1.5 py-0.5 text-[11px] text-neutral-600">${escapeHtml(keyword)}</span>`).join("")}
      </div>
      <div class="h-1.5 border border-line bg-white">
        <div class="h-full ${concept.polarity === "risk" ? "bg-hazard" : "bg-signal"}" style="width: ${Math.round(dominant * 100)}%"></div>
      </div>
      <div class="mt-3 border-t border-line pt-2">
        <div class="mb-1 text-[10px] font-semibold uppercase tracking-normal text-neutral-500">Calibration</div>
        <div class="space-y-1">
          ${concept.calibration_prompts
            .slice(0, 3)
            .map((prompt) => `<p class="text-[11px] leading-4 text-neutral-600">${escapeHtml(prompt)}</p>`)
            .join("")}
        </div>
      </div>
    </article>
  `;
}

function miniScore(label: string, score: number, polarity: ConceptPolarity): string {
  return `
    <div class="border border-line bg-white p-2">
      <div class="mb-1 flex items-center justify-between gap-2">
        <span class="font-semibold text-neutral-500">${escapeHtml(label)}</span>
        <span class="font-mono">${score.toFixed(2)}</span>
      </div>
      <div class="h-1.5 border border-line bg-field">
        <div class="h-full ${polarity === "risk" ? "bg-hazard" : "bg-signal"}" style="width: ${Math.round(score * 100)}%"></div>
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

function renderMonitorOutput(response: RunResponse, frames: TraceFrame[], selectedFrame: TraceFrame | null): string {
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
      <p class="whitespace-pre-wrap text-sm leading-6">${renderGeneratedAnswer(frames, selectedFrame, response.output)}</p>
    </section>
  `;
}

function renderRewriteOutputs(response: RunResponse, frames: TraceFrame[], selectedFrame: TraceFrame | null): string {
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
        <p class="whitespace-pre-wrap text-sm leading-6">${renderGeneratedAnswer(frames, selectedFrame, response.output)}</p>
      </div>
    </section>
  `;
}

function renderGeneratedAnswer(frames: TraceFrame[], selectedFrame: TraceFrame | null, fallbackText: string): string {
  const decodeFrames = frames.filter((frame) => frame.phase === "decode");
  if (!decodeFrames.length) return escapeHtml(fallbackText);
  const selectedStep = selectedFrame?.step ?? decodeFrames.at(-1)?.step ?? 0;
  return decodeFrames
    .map((frame) => {
      const token = frame.token || "";
      if (!token) return "";
      const selected = frame.step === selectedStep;
      const tripwire = frame.layer_activations?.some(isTripwireActivation) ?? false;
      return `
        <span
          role="button"
          tabindex="0"
          data-frame-step="${frame.step}"
          class="answer-token ${selected ? "is-selected" : ""} ${tripwire ? "is-tripwire" : ""}"
          title="step ${frame.step}"
        >${escapeHtml(token)}</span>
      `;
    })
    .join("");
}

function selectedTraceFrame(frames: TraceFrame[]): TraceFrame | null {
  if (!frames.length) return null;
  if (state.inspectedStep !== null) {
    const selected = frames.find((frame) => frame.step === state.inspectedStep);
    if (selected) return selected;
  }
  return frames.at(-1) ?? null;
}

function renderActivationExplorer(
  response: RunResponse,
  frames: TraceFrame[],
  selectedFrame: TraceFrame | null
): string {
  const layers = activationLayers(frames);
  const selectedStep = selectedFrame?.step ?? frames.at(-1)?.step ?? 0;
  if (!frames.length || !layers.length) {
    return `
      <section class="border border-line bg-field p-4">
        <div class="mb-2 text-sm font-semibold">Layer Heatmap</div>
        <p class="text-sm text-neutral-600">No activation frames available.</p>
      </section>
    `;
  }

  return `
    <section class="border border-line">
      <div class="flex flex-col gap-2 border-b border-line bg-field px-3 py-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 class="text-sm font-semibold">Layer Heatmap</h3>
          <p class="text-xs text-neutral-500">${frames.length} token states across ${layers.length} layers</p>
        </div>
        <div class="flex flex-wrap items-center gap-2 text-[11px] font-semibold">
          <span class="border border-line bg-white px-2 py-0.5">threshold ${state.threshold.toFixed(2)}</span>
          <span class="border border-line bg-white px-2 py-0.5">control L${response.engine.layer}</span>
          <span class="border border-hazard bg-red-50 px-2 py-0.5 text-hazard">tripwire</span>
        </div>
      </div>
      <div class="trace-scroll max-h-[560px] overflow-auto p-3">
        <div class="token-heatmap text-xs" style="grid-template-columns: 56px repeat(${frames.length}, minmax(34px, 44px));">
          <div class="heatmap-corner">layer</div>
          ${frames.map((frame) => renderTokenHeader(frame, selectedStep)).join("")}
          ${layers
            .map(
              (layer) => `
                <div class="heatmap-layer-label ${layer === response.engine.layer ? "is-control" : ""}">L${layer}</div>
                ${frames.map((frame) => renderHeatmapCell(frame, layer, selectedStep, response.engine.layer)).join("")}
              `
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function activationLayers(frames: TraceFrame[]): number[] {
  const layers = new Set<number>();
  frames.forEach((frame) => {
    frame.layer_activations?.forEach((activation) => layers.add(activation.layer));
  });
  return Array.from(layers).sort((left, right) => left - right);
}

function renderTokenHeader(frame: TraceFrame, selectedStep: number): string {
  const selected = frame.step === selectedStep;
  const label = tokenLabel(frame);
  return `
    <button
      type="button"
      data-frame-step="${frame.step}"
      class="heatmap-token ${selected ? "is-selected" : ""} ${frame.action !== "observe" ? `is-${frame.action}` : ""}"
      title="step ${frame.step}: ${escapeHtml(label)}"
    >
      <span class="font-mono text-[10px]">${frame.phase === "prompt" ? "P" : frame.step}</span>
      <span class="truncate">${escapeHtml(label)}</span>
    </button>
  `;
}

function renderHeatmapCell(
  frame: TraceFrame,
  layer: number,
  selectedStep: number,
  controlLayer: number
): string {
  const activation = frame.layer_activations?.find((item) => item.layer === layer);
  if (!activation) {
    return `<button type="button" data-frame-step="${frame.step}" class="heatmap-cell is-empty" title="step ${frame.step} layer ${layer}"></button>`;
  }
  const selected = frame.step === selectedStep;
  const tripwire = isTripwireActivation(activation);
  const classes = [
    "heatmap-cell",
    selected ? "is-selected" : "",
    tripwire ? "is-tripwire" : "",
    layer === controlLayer ? "is-control-layer" : ""
  ]
    .filter(Boolean)
    .join(" ");
  return `
    <button
      type="button"
      data-frame-step="${frame.step}"
      class="${classes}"
      style="--score: ${activation.score}; --cell-color: ${cellColor(activation)};"
      title="step ${frame.step} layer ${layer}: ${escapeHtml(activation.label)} ${activation.score.toFixed(2)}"
    >
      <span class="sr-only">${escapeHtml(activation.label)} ${activation.score.toFixed(2)}</span>
    </button>
  `;
}

function renderTokenInspector(frame: TraceFrame | null, response: { engine: RunResponse["engine"] }): string {
  if (!frame) {
    return `
      <section class="border border-line bg-field p-4">
        <div class="text-sm font-semibold">Token Inspector</div>
      </section>
    `;
  }
  const strongest = [...(frame.layer_activations ?? [])].sort((left, right) => right.score - left.score).slice(0, 6);
  const tripwires = strongest.filter(isTripwireActivation);
  return `
    <section class="border border-line">
      <div class="border-b border-line bg-field px-3 py-2">
        <div class="flex items-center justify-between gap-3">
          <h3 class="text-sm font-semibold">Token Inspector</h3>
          <span class="font-mono text-xs ${actionTextClass(frame.action)}">${escapeHtml(frame.action)}</span>
        </div>
      </div>
      <div class="space-y-3 p-3">
        <div class="border border-line bg-white p-3">
          <div class="mb-1 flex items-center justify-between gap-3">
            <span class="font-mono text-xs text-neutral-500">step ${frame.step} / ${frame.phase}</span>
            <span class="font-mono text-xs">L${response.engine.layer}</span>
          </div>
          <div class="break-words text-sm font-semibold">${escapeHtml(tokenLabel(frame))}</div>
          ${frame.note ? `<p class="mt-2 text-xs leading-5 text-neutral-600">${escapeHtml(frame.note)}</p>` : ""}
        </div>
        <div class="border border-line bg-white">
          <div class="border-b border-line px-2 py-1.5 text-xs font-semibold">Tripwire Layers</div>
          ${
            tripwires.length
              ? tripwires.map(renderLayerActivationRow).join("")
              : `<div class="p-2 text-xs text-neutral-500">No risk layer crosses ${state.threshold.toFixed(2)} for this token.</div>`
          }
        </div>
        <div class="border border-line bg-white">
          <div class="border-b border-line px-2 py-1.5 text-xs font-semibold">Strongest Layers</div>
          ${strongest.length ? strongest.map(renderLayerActivationRow).join("") : `<div class="p-2 text-xs text-neutral-500">No layer scan.</div>`}
        </div>
      </div>
    </section>
  `;
}

function renderLayerActivationRow(activation: LayerActivation): string {
  const tripwire = isTripwireActivation(activation);
  return `
    <div class="border-b border-line p-2 last:border-b-0 ${tripwire ? "bg-red-50" : ""}">
      <div class="mb-1 flex items-center justify-between gap-2">
        <span class="font-mono text-[11px] ${tripwire ? "text-hazard" : "text-neutral-500"}">L${activation.layer}</span>
        <span class="font-mono text-[11px]">${activation.score.toFixed(2)}</span>
      </div>
      <div class="mb-1.5 flex items-center justify-between gap-2">
        <span class="truncate text-xs font-semibold">${escapeHtml(activation.label)}</span>
        ${tripwire ? `<span class="border border-hazard bg-white px-1 py-0.5 text-[10px] font-semibold text-hazard">TRIP</span>` : ""}
      </div>
      <div class="h-1.5 border border-line bg-field">
        <div class="h-full ${activation.polarity === "risk" ? "bg-hazard" : "bg-signal"}" style="width: ${Math.round(activation.score * 100)}%"></div>
      </div>
    </div>
  `;
}

function tokenLabel(frame: TraceFrame): string {
  if (frame.phase === "prompt") return "<prompt>";
  const token = frame.token || "";
  if (!token.trim()) return "[space]";
  return token.replace(/\s+/g, " ");
}

function isTripwireActivation(activation: Pick<LayerActivation | ConceptScore, "polarity" | "score">): boolean {
  return activation.polarity === "risk" && activation.score >= state.threshold;
}

function cellColor(score: Pick<LayerActivation | ConceptScore, "polarity">): string {
  if (score.polarity === "risk") return "rgba(255,77,94,0.9)";
  if (score.polarity === "target") return "rgba(66,96,255,0.9)";
  return "rgba(215,221,255,0.72)";
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
  const controlConcepts = latest?.top_concepts ?? [];
  const proxyConcepts = latest?.proxy_concepts?.length ? latest.proxy_concepts : controlConcepts;
  const jacobianConcepts = latest?.jacobian_concepts ?? [];
  const proxyTop = proxyConcepts[0];
  const jacobianTop = jacobianConcepts[0];
  const alignment =
    !latest
      ? "Idle"
      : proxyTop && jacobianTop
      ? proxyTop.id === jacobianTop.id
        ? "Aligned"
        : "Diverged"
      : jacobianTop
        ? "Lens Only"
        : "Proxy Only";
  return `
    <section class="border border-line">
      <div class="border-b border-line bg-field px-3 py-2">
        <div class="flex items-center justify-between gap-3">
          <h3 class="flex items-center gap-2 text-sm font-semibold">
            <i data-lucide="radio-tower" class="h-4 w-4"></i>
            Readout Comparison
          </h3>
          <span class="border border-line bg-white px-2 py-0.5 text-[11px] font-semibold">${escapeHtml(alignment)}</span>
        </div>
      </div>
      <div class="space-y-3 p-3">
        ${renderReadoutLane("Control", controlConcepts, "bg-ink text-white")}
        ${renderReadoutLane("Proxy", proxyConcepts, "bg-field text-ink")}
        ${renderReadoutLane("Jacobian", jacobianConcepts, "bg-sky-50 text-violetless", latest?.jacobian_tokens ?? [])}
      </div>
    </section>
  `;
}

function renderReadoutLane(title: string, concepts: ConceptScore[], titleClass: string, tokens: string[] = []): string {
  return `
    <div class="border border-line bg-white">
      <div class="flex items-center justify-between gap-2 border-b border-line px-2 py-1.5">
        <span class="px-1.5 py-0.5 text-xs font-semibold ${titleClass}">${escapeHtml(title)}</span>
        ${concepts[0] ? `<span class="font-mono text-xs">${concepts[0].score.toFixed(2)}</span>` : ""}
      </div>
      <div class="divide-y divide-line">
        ${
          concepts.length
            ? concepts
                .slice(0, 4)
                .map(
                  (concept) => {
                    const tripwire = isTripwireActivation(concept);
                    return `
                    <div class="p-2 ${tripwire ? "bg-red-50" : ""}">
                      <div class="mb-1.5 flex items-center justify-between gap-3">
                        <span class="truncate text-xs font-semibold">${escapeHtml(concept.label)}</span>
                        <span class="inline-flex items-center gap-1 font-mono text-[11px]">
                          ${tripwire ? `<span class="border border-hazard bg-white px-1 py-0.5 text-[10px] font-semibold text-hazard">TRIP</span>` : ""}
                          ${concept.score.toFixed(2)}
                        </span>
                      </div>
                      <div class="h-1.5 border border-line bg-field">
                        <div class="h-full ${concept.polarity === "risk" ? "bg-hazard" : "bg-signal"}" style="width: ${Math.round(concept.score * 100)}%"></div>
                      </div>
                      <div class="mt-1.5 flex flex-wrap gap-1">
                        ${concept.evidence.map((item) => `<span class="border border-line bg-field px-1 py-0.5 text-[10px] text-neutral-600">${escapeHtml(item)}</span>`).join("")}
                      </div>
                    </div>
                  `;
                  }
                )
                .join("")
            : `<div class="p-2 text-xs text-neutral-500">No signal</div>`
        }
      </div>
      ${
        tokens.length
          ? `<div class="flex flex-wrap gap-1 border-t border-line bg-field p-2">
              ${tokens.map((token) => `<span class="border border-line bg-white px-1 py-0.5 text-[10px] text-neutral-600">${escapeHtml(token)}</span>`).join("")}
            </div>`
          : ""
      }
    </div>
  `;
}

function bindEvents() {
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextMode = button.dataset.tab as AppMode;
      state.activeMode = nextMode;
      if (nextMode === "custom") {
        if (!state.customResult && !state.prompt.includes("SOC 2")) state.prompt = CUSTOM_DEFAULT_PROMPT;
      } else {
        const firstScenario = state.profiles?.scenarios.find((scenario) => scenario.mode === nextMode);
        if (firstScenario) applyScenario(firstScenario);
      }
      state.result = null;
      state.customResult = null;
      state.inspectedStep = null;
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-scenario]").forEach((button) => {
    button.addEventListener("click", () => {
      const scenario = state.profiles?.scenarios.find((item) => item.id === button.dataset.scenario);
      if (scenario) {
        applyScenario(scenario);
        state.result = null;
        state.inspectedStep = null;
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
  bindInput("customWindowTokens", (element) => {
    state.customWindowTokens = Number((element as HTMLInputElement).value);
  });

  document.querySelectorAll<HTMLInputElement>("[data-custom-label]").forEach((input) => {
    input.addEventListener("input", () => {
      const index = Number(input.dataset.customLabel);
      if (Number.isInteger(index) && state.customConcepts[index]) {
        state.customConcepts[index].label = input.value;
      }
    });
  });

  document.querySelectorAll<HTMLTextAreaElement>("[data-custom-keywords]").forEach((textarea) => {
    textarea.addEventListener("input", () => {
      const index = Number(textarea.dataset.customKeywords);
      if (Number.isInteger(index) && state.customConcepts[index]) {
        state.customConcepts[index].keywords = splitKeywords(textarea.value);
      }
    });
  });

  document.querySelectorAll<HTMLSelectElement>("[data-custom-polarity]").forEach((select) => {
    select.addEventListener("change", () => {
      const index = Number(select.dataset.customPolarity);
      if (Number.isInteger(index) && state.customConcepts[index]) {
        state.customConcepts[index].polarity = select.value as ConceptPolarity;
        render();
      }
    });
  });

  document.querySelector<HTMLButtonElement>("#run")?.addEventListener("click", run);
  document.querySelector<HTMLButtonElement>("#replay")?.addEventListener("click", () => {
    if (state.result) startReplay();
  });

  document.querySelectorAll<HTMLElement>("[data-frame-step]").forEach((element) => {
    const step = Number(element.dataset.frameStep);
    if (!Number.isFinite(step)) return;
    const inspect = () => {
      if (state.inspectedStep === step) return;
      state.inspectedStep = step;
      render();
    };
    element.addEventListener("mouseenter", inspect);
    element.addEventListener("focus", inspect);
    element.addEventListener("click", inspect);
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
  if (state.activeMode === "custom") {
    await runCustomLens();
    return;
  }
  state.loading = true;
  state.error = null;
  state.result = null;
  state.customResult = null;
  state.inspectedStep = null;
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
    state.health = { ok: true, engine: result.engine };
    state.inspectedStep = null;
    state.replayCount = 0;
    startReplay();
  } catch (error) {
    state.error = error instanceof Error ? error.message : "Run failed";
  } finally {
    state.loading = false;
    render();
  }
}

async function runCustomLens() {
  state.loading = true;
  state.error = null;
  state.result = null;
  state.customResult = null;
  state.inspectedStep = null;
  stopReplay();
  render();

  try {
    const result = await fetchJson<CustomLensResponse>("/api/custom-lens", {
      method: "POST",
      body: JSON.stringify({
        prompt: state.prompt,
        concepts: state.customConcepts.filter((concept) => concept.label.trim() && concept.keywords.length > 0),
        window_tokens: state.customWindowTokens
      })
    });
    state.customResult = result;
    state.health = { ok: true, engine: result.engine };
    state.inspectedStep = null;
  } catch (error) {
    state.error = error instanceof Error ? error.message : "Custom lens failed";
  } finally {
    state.loading = false;
    render();
  }
}

function splitKeywords(value: string): string[] {
  return value
    .split(/[,;\n]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 12);
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

export interface RuleCondition {
  type: string
  params: Record<string, string>
}

export interface RuleAction {
  type: string
  params: Record<string, string>
}

export interface EngineRule {
  id: string
  name: string
  description: string
  enabled: boolean
  trigger: string
  cameras: string[]
  conditions: RuleCondition[]
  thenActions: RuleAction[]
  elseActions: RuleAction[]
  cooldownSeconds: number
  priority: number
  lastTriggered: string | null
  preset: string | null
}

export const triggerOptions = [
  { label: "Detection occurs", value: "detection" },
  { label: "Plate is read", value: "plate_read" },
  { label: "Face matches", value: "face_match" },
  { label: "Unknown face detected", value: "face_unknown" },
  { label: "Zone entered", value: "zone_enter" },
  { label: "Zone exited", value: "zone_exit" },
  { label: "Count threshold reached", value: "count_threshold" },
]

export const conditionTypes = [
  { label: "Plate is in list", value: "plate_in_list", description: "params: list (Whitelist | Blocked | Visitors)" },
  { label: "Face is in group", value: "face_in_group", description: "params: group (Employees | Visitors | Contractors)" },
  { label: "Confidence above", value: "confidence_above", description: "params: value (0.1 – 1.0)" },
  { label: "Zone is", value: "zone_is", description: "params: zone (zone name)" },
  { label: "Time between", value: "time_between", description: "params: from, to (HH:mm)" },
  { label: "Class is", value: "class_is", description: "params: classes (comma-separated)" },
  { label: "Count exceeds", value: "count_exceeds", description: "params: count (number)" },
]

export const actionTypes = [
  { label: "Create alert", value: "create_alert", description: "params: severity (P1 | P2 | P3 | P4)" },
  { label: "Send Telegram", value: "send_telegram", description: "No extra params" },
  { label: "Open gate", value: "open_gate", description: "params: device (device name)" },
  { label: "Close gate", value: "close_gate", description: "params: device (device name)" },
  { label: "Log entry", value: "log_entry", description: "No extra params" },
  { label: "Webhook", value: "webhook", description: "params: url (endpoint URL)" },
  { label: "Trigger PLC", value: "trigger_plc", description: "params: device (device name)" },
  { label: "Play sound", value: "play_sound", description: "No extra params" },
]

export interface PresetTemplate {
  key: string
  name: string
  description: string
  icon: string
  template: Omit<EngineRule, "id" | "lastTriggered">
}

export const presetTemplates: PresetTemplate[] = [
  {
    key: "ppe",
    name: "PPE Violation",
    description: "Alert when workers are detected without required safety equipment",
    icon: "HardHat",
    template: {
      name: "PPE Violation",
      description: "Detect missing PPE on the factory floor",
      enabled: true,
      trigger: "detection",
      cameras: [],
      conditions: [
        { type: "class_is", params: { classes: "no_helmet" } },
        { type: "confidence_above", params: { value: "0.7" } },
      ],
      thenActions: [
        { type: "create_alert", params: { severity: "P2" } },
        { type: "send_telegram", params: {} },
      ],
      elseActions: [],
      cooldownSeconds: 60,
      priority: 5,
      preset: "ppe",
    },
  },
  {
    key: "fire",
    name: "Fire Emergency",
    description: "Immediate response when fire or smoke is detected",
    icon: "Flame",
    template: {
      name: "Fire Emergency",
      description: "Detect fire or smoke and trigger emergency response",
      enabled: true,
      trigger: "detection",
      cameras: [],
      conditions: [
        { type: "class_is", params: { classes: "fire,smoke" } },
        { type: "confidence_above", params: { value: "0.6" } },
      ],
      thenActions: [
        { type: "create_alert", params: { severity: "P1" } },
        { type: "send_telegram", params: {} },
        { type: "trigger_plc", params: { device: "siren" } },
        { type: "webhook", params: { url: "https://hooks.example.com/fire" } },
      ],
      elseActions: [],
      cooldownSeconds: 30,
      priority: 10,
      preset: "fire",
    },
  },
  {
    key: "gate_entry",
    name: "Gate Entry",
    description: "Automated gate control based on plate and face recognition",
    icon: "DoorOpen",
    template: {
      name: "Gate Entry — Auto Open",
      description: "Open gate for authorised vehicles and personnel",
      enabled: true,
      trigger: "plate_read",
      cameras: [],
      conditions: [
        { type: "plate_in_list", params: { list: "Whitelist" } },
        { type: "face_in_group", params: { group: "Employees" } },
      ],
      thenActions: [
        { type: "open_gate", params: { device: "main-gate" } },
        { type: "log_entry", params: {} },
      ],
      elseActions: [
        { type: "create_alert", params: { severity: "P1" } },
        { type: "send_telegram", params: {} },
        { type: "close_gate", params: { device: "main-gate" } },
      ],
      cooldownSeconds: 10,
      priority: 8,
      preset: "gate_entry",
    },
  },
  {
    key: "after_hours",
    name: "After-Hours Intrusion",
    description: "Detect unauthorized access outside of business hours",
    icon: "Moon",
    template: {
      name: "After-Hours Intrusion",
      description: "Alert when motion is detected in restricted areas after hours",
      enabled: true,
      trigger: "zone_enter",
      cameras: [],
      conditions: [
        { type: "zone_is", params: { zone: "Warehouse" } },
        { type: "time_between", params: { from: "22:00", to: "06:00" } },
      ],
      thenActions: [
        { type: "create_alert", params: { severity: "P1" } },
        { type: "send_telegram", params: {} },
        { type: "play_sound", params: {} },
      ],
      elseActions: [],
      cooldownSeconds: 120,
      priority: 9,
      preset: "after_hours",
    },
  },
  {
    key: "overcrowding",
    name: "Overcrowding",
    description: "Alert when person count in a zone exceeds the safe limit",
    icon: "Users",
    template: {
      name: "Overcrowding Alert",
      description: "Trigger when too many people are in a zone",
      enabled: true,
      trigger: "count_threshold",
      cameras: [],
      conditions: [
        { type: "zone_is", params: { zone: "Assembly Line" } },
        { type: "count_exceeds", params: { count: "15" } },
      ],
      thenActions: [
        { type: "create_alert", params: { severity: "P2" } },
        { type: "send_telegram", params: {} },
      ],
      elseActions: [],
      cooldownSeconds: 300,
      priority: 6,
      preset: "overcrowding",
    },
  },
  {
    key: "custom",
    name: "Custom",
    description: "Start from scratch with a blank rule",
    icon: "Settings",
    template: {
      name: "",
      description: "",
      enabled: true,
      trigger: "detection",
      cameras: [],
      conditions: [],
      thenActions: [],
      elseActions: [],
      cooldownSeconds: 60,
      priority: 5,
      preset: null,
    },
  },
]

export const mockRules: EngineRule[] = [
  {
    id: "rule-001",
    name: "Gate Entry — Auto Open",
    description: "Open gate for authorised vehicles and personnel",
    enabled: true,
    trigger: "plate_read",
    cameras: ["cam-01"],
    conditions: [
      { type: "plate_in_list", params: { list: "Whitelist" } },
      { type: "face_in_group", params: { group: "Employees" } },
    ],
    thenActions: [
      { type: "open_gate", params: { device: "main-gate" } },
      { type: "log_entry", params: {} },
    ],
    elseActions: [
      { type: "create_alert", params: { severity: "P1" } },
      { type: "send_telegram", params: {} },
      { type: "close_gate", params: { device: "main-gate" } },
    ],
    cooldownSeconds: 10,
    priority: 8,
    lastTriggered: "2026-03-28T09:14:00Z",
    preset: "gate_entry",
  },
  {
    id: "rule-002",
    name: "PPE — Helmet Required",
    description: "Detect missing helmets on the assembly line",
    enabled: true,
    trigger: "detection",
    cameras: ["cam-02", "cam-03"],
    conditions: [
      { type: "class_is", params: { classes: "no_helmet" } },
      { type: "zone_is", params: { zone: "Assembly Line" } },
      { type: "confidence_above", params: { value: "0.7" } },
    ],
    thenActions: [
      { type: "create_alert", params: { severity: "P2" } },
      { type: "send_telegram", params: {} },
    ],
    elseActions: [],
    cooldownSeconds: 60,
    priority: 5,
    lastTriggered: "2026-03-28T08:52:00Z",
    preset: "ppe",
  },
  {
    id: "rule-003",
    name: "Fire Emergency",
    description: "Detect fire or smoke and trigger emergency response",
    enabled: true,
    trigger: "detection",
    cameras: [],
    conditions: [
      { type: "class_is", params: { classes: "fire,smoke" } },
      { type: "confidence_above", params: { value: "0.6" } },
    ],
    thenActions: [
      { type: "create_alert", params: { severity: "P1" } },
      { type: "send_telegram", params: {} },
      { type: "trigger_plc", params: { device: "siren" } },
      { type: "webhook", params: { url: "https://hooks.example.com/fire" } },
    ],
    elseActions: [],
    cooldownSeconds: 30,
    priority: 10,
    lastTriggered: null,
    preset: "fire",
  },
  {
    id: "rule-004",
    name: "After-Hours Intrusion",
    description: "Alert when motion is detected in restricted areas after hours",
    enabled: true,
    trigger: "zone_enter",
    cameras: ["cam-04"],
    conditions: [
      { type: "zone_is", params: { zone: "Warehouse" } },
      { type: "time_between", params: { from: "22:00", to: "06:00" } },
    ],
    thenActions: [
      { type: "create_alert", params: { severity: "P1" } },
      { type: "send_telegram", params: {} },
      { type: "play_sound", params: {} },
    ],
    elseActions: [],
    cooldownSeconds: 120,
    priority: 9,
    lastTriggered: "2026-03-27T23:45:00Z",
    preset: "after_hours",
  },
  {
    id: "rule-005",
    name: "Overcrowding Alert",
    description: "Trigger when too many people are in Assembly Line zone",
    enabled: true,
    trigger: "count_threshold",
    cameras: ["cam-02", "cam-03"],
    conditions: [
      { type: "zone_is", params: { zone: "Assembly Line" } },
      { type: "count_exceeds", params: { count: "15" } },
    ],
    thenActions: [
      { type: "create_alert", params: { severity: "P2" } },
      { type: "send_telegram", params: {} },
    ],
    elseActions: [],
    cooldownSeconds: 300,
    priority: 6,
    lastTriggered: "2026-03-28T07:30:00Z",
    preset: "overcrowding",
  },
]

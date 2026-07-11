import type { Camera, CameraProfile, CapabilityKey, SafetyRule } from "@/types"

export type CameraDetectionKey = CapabilityKey

export interface CameraDetectionDefinition {
  key: CapabilityKey
  label: string
  description: string
  group: "Core Monitoring" | "Safety Events" | "PPE" | "Advanced"
  preferredRuleIds: string[]
  matchTerms: string[]
  profileHints: CameraProfile[]
}

export interface CameraDetectionOption extends CameraDetectionDefinition {
  ruleId: string | null
  available: boolean
}

type CameraLike = Pick<
  Camera,
  "capabilities" | "profile" | "safety_rule_ids" | "alert_classes" | "yoloe_classes" | "rules"
>

const DETECTION_CATALOG: CameraDetectionDefinition[] = [
  {
    key: "person_presence",
    label: "People",
    description: "Track people entering the scene.",
    group: "Core Monitoring",
    preferredRuleIds: ["alert_person"],
    matchTerms: ["person", "person detected"],
    profileHints: ["general_safety", "work_zone_ppe"],
  },
  {
    key: "office_occupancy",
    label: "Office Occupancy",
    description: "Mark chairs or seats and monitor whether people are present in those work zones.",
    group: "Core Monitoring",
    preferredRuleIds: [],
    matchTerms: ["office occupancy", "chair occupancy", "seat occupancy", "desk occupancy", "workstation"],
    profileHints: ["office_occupancy"],
  },
  {
    key: "crowd_count_threshold",
    label: "Crowd Count",
    description: "Count visible people and evaluate the configured crowd threshold.",
    group: "Core Monitoring",
    preferredRuleIds: [],
    matchTerms: ["crowd count", "people count", "person count", "crowding"],
    profileHints: ["general_safety", "office_occupancy"],
  },
  {
    key: "queue_monitoring",
    label: "Queue Monitoring",
    description: "Measure people, duration, and waiting activity inside queue zones.",
    group: "Core Monitoring",
    preferredRuleIds: [],
    matchTerms: ["queue monitoring", "queue", "waiting line"],
    profileHints: ["general_safety", "office_occupancy"],
  },
  {
    key: "route_obstruction",
    label: "Route Obstruction",
    description: "Monitor gangways and keep-clear zones for blocking people, vehicles, or objects.",
    group: "Core Monitoring",
    preferredRuleIds: [],
    matchTerms: ["route obstruction", "gangway blockage", "keep clear", "obstruction"],
    profileHints: ["general_safety", "work_zone_ppe"],
  },
  {
    key: "object_lifecycle",
    label: "Object Lifecycle",
    description: "Detect when watched objects remain in or are removed from a configured zone.",
    group: "Core Monitoring",
    preferredRuleIds: [],
    matchTerms: ["object lifecycle", "object removed", "object dwell", "unattended object", "watched object"],
    profileHints: ["general_safety"],
  },
  {
    key: "vehicle_presence",
    label: "Vehicles",
    description: "Track cars, trucks, and other vehicles.",
    group: "Core Monitoring",
    preferredRuleIds: ["alert_vehicle"],
    matchTerms: ["vehicle", "car", "truck", "motorcycle"],
    profileHints: ["general_safety"],
  },
  {
    key: "animal_presence",
    label: "Animals",
    description: "Alert when animals enter the camera view.",
    group: "Safety Events",
    preferredRuleIds: ["alert_animal"],
    matchTerms: ["animal", "dog", "cat", "deer"],
    profileHints: ["general_safety"],
  },
  {
    key: "mobile_phone",
    label: "Mobile Phone",
    description: "Detect mobile phone usage in monitored areas.",
    group: "Safety Events",
    preferredRuleIds: ["alert_mobile_phone"],
    matchTerms: ["mobile phone", "cell phone", "phone"],
    profileHints: ["general_safety", "work_zone_ppe"],
  },
  {
    key: "zone_intrusion",
    label: "Zone Intrusion",
    description: "Alert when a person enters a restricted area.",
    group: "Safety Events",
    preferredRuleIds: ["alert_zone_intrusion"],
    matchTerms: ["zone intrusion", "restricted zone", "intrusion"],
    profileHints: ["general_safety", "work_zone_ppe"],
  },
  {
    key: "fall_detection",
    label: "Fall / Man Down",
    description: "Detect a person in a fallen or man-down posture using temporal confirmation.",
    group: "Safety Events",
    preferredRuleIds: ["alert_fall_detection"],
    matchTerms: ["fall detection", "fall detected", "person fall", "man down", "fallen person"],
    profileHints: ["demo_advanced", "general_safety"],
  },
  {
    key: "helmet_required",
    label: "Helmet",
    description: "Check whether workers are wearing helmets.",
    group: "PPE",
    preferredRuleIds: ["ppe_helmet"],
    matchTerms: ["helmet", "hard hat", "safety helmet"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "rider_helmet_required",
    label: "Rider Helmet",
    description: "Check helmets only for people riding motorcycles or site vehicles.",
    group: "PPE",
    preferredRuleIds: ["ppe_rider_helmet"],
    matchTerms: ["rider helmet", "motorcycle helmet", "helmet on vehicle", "helmet while riding"],
    profileHints: ["general_safety", "work_zone_ppe"],
  },
  {
    key: "vest_required",
    label: "Safety Vest",
    description: "Check whether workers are wearing safety vests.",
    group: "PPE",
    preferredRuleIds: ["ppe_vest"],
    matchTerms: ["vest", "safety vest", "high visibility vest", "fluorescent vest"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "gloves_required",
    label: "Gloves",
    description: "Check whether workers are wearing gloves.",
    group: "PPE",
    preferredRuleIds: ["ppe_gloves"],
    matchTerms: ["gloves"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "hairnet_required",
    label: "Hairnet",
    description: "Check whether workers are wearing hairnets.",
    group: "PPE",
    preferredRuleIds: ["ppe_hairnet"],
    matchTerms: ["hairnet"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "face_mask_required",
    label: "Face Mask",
    description: "Check whether workers are wearing face masks.",
    group: "PPE",
    preferredRuleIds: ["ppe_face_mask", "ppe_facemask"],
    matchTerms: ["face mask", "surgical mask", "medical mask", "mask", "respirator"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "face_shield_required",
    label: "Face Shield",
    description: "Check whether workers are wearing face shields.",
    group: "PPE",
    preferredRuleIds: ["ppe_face_shield"],
    matchTerms: ["face shield", "protective face shield", "clear face shield", "visor", "protective visor"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "apron_required",
    label: "Apron",
    description: "Check whether workers are wearing aprons.",
    group: "PPE",
    preferredRuleIds: ["ppe_apron"],
    matchTerms: ["apron"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "boots_required",
    label: "Safety Boots",
    description: "Check whether workers are wearing safety boots.",
    group: "PPE",
    preferredRuleIds: ["ppe_boots"],
    matchTerms: ["safety boots", "steel-toe boots", "work boots", "protective boots", "rubber boots", "boots"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "harness_required",
    label: "Safety Harness",
    description: "Check whether workers are wearing safety harnesses.",
    group: "PPE",
    preferredRuleIds: ["ppe_harness"],
    matchTerms: ["safety harness", "fall arrest harness", "body harness", "harness"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "goggles_required",
    label: "Safety Goggles",
    description: "Check whether workers are wearing safety goggles.",
    group: "PPE",
    preferredRuleIds: ["ppe_goggles"],
    matchTerms: ["safety goggles", "protective eyewear", "goggles"],
    profileHints: ["work_zone_ppe"],
  },
  {
    key: "fire_smoke",
    label: "Fire / Smoke",
    description: "Detect fire, flames, or smoke on selected cameras.",
    group: "Advanced",
    preferredRuleIds: ["alert_fire_smoke"],
    matchTerms: ["fire", "smoke", "flames"],
    profileHints: ["demo_advanced"],
  },
  {
    key: "face_recognition",
    label: "Face Recognition",
    description: "Recognize enrolled people on access-control cameras.",
    group: "Advanced",
    preferredRuleIds: [],
    matchTerms: ["face recognition", "face"],
    profileHints: ["demo_advanced"],
  },
  {
    key: "plate_recognition",
    label: "Plate Recognition",
    description: "Read vehicle number plates on gate cameras.",
    group: "Advanced",
    preferredRuleIds: [],
    matchTerms: ["plate recognition", "license plate", "number plate", "anpr"],
    profileHints: ["demo_advanced"],
  },
  {
    key: "custom_long_tail",
    label: "Custom Detection",
    description: "Detect configured site-specific visible objects or conditions.",
    group: "Advanced",
    preferredRuleIds: [],
    matchTerms: ["custom detection", "custom long tail", "long tail"],
    profileHints: ["demo_advanced"],
  },
]

const LEGACY_ALERT_TO_KEY: Partial<Record<string, CapabilityKey>> = {
  mobile_phone: "mobile_phone",
  animal_intrusion: "animal_presence",
  person_detected: "person_presence",
  vehicle_detected: "vehicle_presence",
  zone_intrusion: "zone_intrusion",
}

const CLASS_TO_KEY: Partial<Record<string, CapabilityKey>> = {
  "cell phone": "mobile_phone",
  "office occupancy": "office_occupancy",
  "desk occupancy": "office_occupancy",
  workstation: "office_occupancy",
  "chair occupancy": "office_occupancy",
  "seat occupancy": "office_occupancy",
  "crowd count": "crowd_count_threshold",
  "people count": "crowd_count_threshold",
  "person count": "crowd_count_threshold",
  "queue monitoring": "queue_monitoring",
  queue: "queue_monitoring",
  "route obstruction": "route_obstruction",
  "gangway blockage": "route_obstruction",
  "keep clear": "route_obstruction",
  "object lifecycle": "object_lifecycle",
  "object removed": "object_lifecycle",
  "object dwell": "object_lifecycle",
  "unattended object": "object_lifecycle",
  dog: "animal_presence",
  cat: "animal_presence",
  deer: "animal_presence",
  animal: "animal_presence",
  car: "vehicle_presence",
  truck: "vehicle_presence",
  motorcycle: "vehicle_presence",
  "hard hat": "helmet_required",
  "safety helmet": "helmet_required",
  "rider helmet": "rider_helmet_required",
  "motorcycle helmet": "rider_helmet_required",
  "helmet on vehicle": "rider_helmet_required",
  "helmet while riding": "rider_helmet_required",
  "safety vest": "vest_required",
  "high visibility vest": "vest_required",
  "fluorescent vest": "vest_required",
  gloves: "gloves_required",
  hairnet: "hairnet_required",
  "face mask": "face_mask_required",
  "surgical mask": "face_mask_required",
  "medical mask": "face_mask_required",
  mask: "face_mask_required",
  respirator: "face_mask_required",
  "face shield": "face_shield_required",
  "protective face shield": "face_shield_required",
  "clear face shield": "face_shield_required",
  visor: "face_shield_required",
  "protective visor": "face_shield_required",
  apron: "apron_required",
  "safety boots": "boots_required",
  "steel toe boots": "boots_required",
  "steel-toe boots": "boots_required",
  "work boots": "boots_required",
  "protective boots": "boots_required",
  "rubber boots": "boots_required",
  "safety harness": "harness_required",
  "fall arrest harness": "harness_required",
  "body harness": "harness_required",
  harness: "harness_required",
  "safety goggles": "goggles_required",
  "protective eyewear": "goggles_required",
  fire: "fire_smoke",
  smoke: "fire_smoke",
  flames: "fire_smoke",
  "fall detection": "fall_detection",
  "fall detected": "fall_detection",
  "person fall": "fall_detection",
  "man down": "fall_detection",
  "face recognition": "face_recognition",
  "plate recognition": "plate_recognition",
  "license plate": "plate_recognition",
  "number plate": "plate_recognition",
  anpr: "plate_recognition",
}

function normalizeText(value: string) {
  return value.toLowerCase().replace(/[_-]+/g, " ").trim()
}

function appendUnique<T>(items: T[], value: T | null | undefined) {
  if (value && !items.includes(value)) {
    items.push(value)
  }
}

function matchesRule(rule: SafetyRule, terms: string[]) {
  const haystack = [rule.id, rule.name, ...rule.classes].map(normalizeText).join(" ")
  return terms.some((term) => haystack.includes(normalizeText(term)))
}

function findRuleForDetection(definition: CameraDetectionDefinition, rules: SafetyRule[]) {
  const preferredMatch = definition.preferredRuleIds
    .map((ruleId) => rules.find((rule) => rule.id === ruleId))
    .find(Boolean)
  if (preferredMatch) return preferredMatch
  return rules.find((rule) => matchesRule(rule, definition.matchTerms))
}

export function getDetectionCatalog() {
  return DETECTION_CATALOG
}

export function getDetectionOptions(rules: SafetyRule[]): CameraDetectionOption[] {
  return DETECTION_CATALOG.map((definition) => {
    const rule = findRuleForDetection(definition, rules)
    const availableWithoutRule = definition.preferredRuleIds.length === 0
    return {
      ...definition,
      ruleId: rule?.id || null,
      available: availableWithoutRule || Boolean(rule),
    }
  })
}

function inferDetectionKeyFromText(value: string): CapabilityKey | null {
  const normalized = normalizeText(value)
  for (const [term, key] of Object.entries(CLASS_TO_KEY)) {
    if (normalized === term || normalized.includes(term)) {
      return key ?? null
    }
  }
  for (const definition of DETECTION_CATALOG) {
    if (definition.matchTerms.some((term) => normalized.includes(normalizeText(term)))) {
      return definition.key
    }
  }
  return null
}

export function getSelectedDetectionKeys(ruleIds: string[], rules: SafetyRule[]): CapabilityKey[] {
  const selected = new Set(ruleIds)
  return getDetectionOptions(rules)
    .filter((option) => (option.ruleId && selected.has(option.ruleId)) || option.preferredRuleIds.some((ruleId) => selected.has(ruleId)))
    .map((option) => option.key)
}

export function getConfiguredDetectionKeys(camera: CameraLike, rules: SafetyRule[]): CapabilityKey[] {
  const keys: CapabilityKey[] = []

  for (const capability of camera.capabilities || []) {
    appendUnique(keys, capability)
  }
  for (const key of getSelectedDetectionKeys(camera.safety_rule_ids || [], rules)) {
    appendUnique(keys, key)
  }
  for (const alertKey of camera.alert_classes || []) {
    appendUnique(keys, LEGACY_ALERT_TO_KEY[alertKey] || null)
  }
  for (const className of camera.yoloe_classes || []) {
    if (normalizeText(className) === "person") continue
    appendUnique(keys, inferDetectionKeyFromText(className))
  }
  if (keys.length === 0) {
    for (const ruleName of camera.rules || []) {
      appendUnique(keys, inferDetectionKeyFromText(ruleName))
    }
  }

  return DETECTION_CATALOG.filter((definition) => keys.includes(definition.key)).map((definition) => definition.key)
}

export function getSelectedRuleIds(keys: CapabilityKey[], rules: SafetyRule[]): string[] {
  const selected = new Set(keys)
  return Array.from(
    new Set(
      getDetectionOptions(rules)
        .filter((option) => selected.has(option.key) && option.ruleId)
        .map((option) => option.ruleId as string)
    )
  )
}

export function getUnmappedRuleIds(ruleIds: string[], rules: SafetyRule[]) {
  const mappedRuleIds = new Set(
    getDetectionOptions(rules)
      .flatMap((option) => [option.ruleId, ...option.preferredRuleIds])
      .filter(Boolean) as string[]
  )
  return ruleIds.filter((ruleId) => !mappedRuleIds.has(ruleId))
}

export function getDetectionLabelsFromKeys(keys: CapabilityKey[]): string[] {
  const selected = new Set(keys)
  return DETECTION_CATALOG.filter((definition) => selected.has(definition.key)).map((definition) => definition.label)
}

export function getConfiguredDetectionLabels(camera: CameraLike, rules: SafetyRule[]): string[] {
  return getDetectionLabelsFromKeys(getConfiguredDetectionKeys(camera, rules))
}

export function getDisplayDetectionLabelsFromRuleIds(ruleIds: string[], rules: SafetyRule[]): string[] {
  const optionByRuleId = new Map(
    getDetectionOptions(rules)
      .filter((option) => option.ruleId)
      .map((option) => [option.ruleId as string, option.label])
  )
  const ruleById = new Map(rules.map((rule) => [rule.id, rule]))
  const labels: string[] = []
  for (const ruleId of ruleIds) {
    const mappedLabel = optionByRuleId.get(ruleId)
    if (mappedLabel) {
      labels.push(mappedLabel)
      continue
    }
    const fallbackRule = ruleById.get(ruleId)
    if (fallbackRule) {
      labels.push(fallbackRule.name)
    }
  }
  return Array.from(new Set(labels))
}

export function hasPpeDetections(keys: CapabilityKey[]) {
  return keys.some((key) => key.endsWith("_required"))
}

export function usesZoneIntrusion(keys: CapabilityKey[]) {
  return keys.includes("zone_intrusion")
}

export function usesConfiguredZones(keys: CapabilityKey[]) {
  return keys.some((key) =>
    ["zone_intrusion", "office_occupancy", "queue_monitoring", "route_obstruction", "object_lifecycle"].includes(key)
  )
}

export function deriveCameraPurpose(keys: CapabilityKey[]) {
  if (keys.length === 0) return "Monitoring only"
  if (keys.length === 1 && keys[0] === "animal_presence") return "Animal monitoring"
  if (keys.length === 1 && keys[0] === "zone_intrusion") return "Restricted zone monitoring"
  if (keys.includes("office_occupancy")) return "Office occupancy"
  if (keys.includes("plate_recognition")) return "Vehicle access monitoring"
  if (keys.every((key) => key.endsWith("_required"))) return "PPE compliance"
  if (keys.includes("fire_smoke")) return "Advanced monitoring"
  return "Custom monitoring"
}

export function deriveCameraPurposeFromRuleIds(ruleIds: string[], rules: SafetyRule[]) {
  const keys = getSelectedDetectionKeys(ruleIds, rules)
  const unmappedRuleIds = getUnmappedRuleIds(ruleIds, rules)
  if (unmappedRuleIds.length > 0) return "Custom monitoring"
  return deriveCameraPurpose(keys)
}

export function deriveConfiguredCameraPurpose(camera: CameraLike, rules: SafetyRule[]) {
  const keys = getConfiguredDetectionKeys(camera, rules)
  const unmappedRuleIds = getUnmappedRuleIds(camera.safety_rule_ids || [], rules)
  if (unmappedRuleIds.length > 0) return "Custom monitoring"
  return deriveCameraPurpose(keys)
}

export function getDetectionModeForKeys(keys: CapabilityKey[]) {
  if (hasPpeDetections(keys) || keys.includes("fire_smoke")) {
    return keys.some((key) => ["person_presence", "vehicle_presence", "animal_presence", "mobile_phone", "zone_intrusion", "rider_helmet_required"].includes(key))
      ? "yolo+yoloe"
      : "yoloe"
  }
  return "yolo"
}

export function getDetectionSummary(keys: CapabilityKey[], limit = 3) {
  const labels = getDetectionLabelsFromKeys(keys)
  if (labels.length <= limit) return labels
  return [...labels.slice(0, limit), `+${labels.length - limit} more`]
}

export function getCameraPurpose(camera: Pick<Camera, "safety_rule_ids" | "capabilities">, rules: SafetyRule[]) {
  if (camera.capabilities?.length) {
    return deriveCameraPurpose(camera.capabilities)
  }
  return deriveCameraPurposeFromRuleIds(camera.safety_rule_ids || [], rules)
}

export function cameraNeedsLegacyNormalization(camera: CameraLike) {
  const hasCapabilities = (camera.capabilities || []).length > 0
  const hasRuleIds = (camera.safety_rule_ids || []).length > 0
  const hasLegacyFields =
    (camera.alert_classes || []).length > 0 ||
    (camera.rules || []).length > 0 ||
    (camera.yoloe_classes || []).some((value) => normalizeText(value) !== "person")
  return !hasCapabilities && !hasRuleIds && hasLegacyFields
}

export function cameraHasDetectionModeMismatch(_camera: CameraLike, _rules: SafetyRule[]) {
  return false
}

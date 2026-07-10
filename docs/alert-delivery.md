# Alert delivery outcomes and retries

External notification providers return a typed outcome per channel. The
backend does not infer that every failure is worth retrying:

| Outcome | Meaning | Automatic retry |
|---|---|---|
| `delivered` | The provider accepted the request | No |
| `retryable` | The same request may succeed later | Yes, within the configured attempt budget |
| `terminal` | Configuration or an unchanged request must be fixed | No |
| `skipped` | Routing policy intentionally suppressed the channel | No |

Telegram and webhooks retry HTTP 408, 425, 429, and 5xx responses because a
gateway may have accepted the request before returning an error. They stop on
permanent 4xx responses. Telegram JSON error codes
take precedence over the HTTP status. `Retry-After` is accepted as either
delta-seconds or an HTTP date; Telegram's JSON `parameters.retry_after` takes
precedence over the header. The retry scheduler never runs earlier than that
provider deadline.

SMTP 4xx responses are retryable and 5xx responses are terminal. Partial SMTP
acceptance is explicitly marked ambiguous: safety delivery is retried even
though already-accepted recipients may see a duplicate. The durable outbox
splits email work by recipient so only temporarily refused recipients retry.

Webhook redirects are disabled so configured credentials and alert payloads
cannot be forwarded to a different host. Every webhook attempt carries a
stable `Idempotency-Key`, and email attempts carry a stable `Message-ID`, both
derived from the persisted alert identity. Telegram has no equivalent client
idempotency facility, so an ambiguous connection failure can still produce a
duplicate.

Webhook DNS is resolved and policy-checked exactly once per attempt. The
transport connects directly to that approved IP while retaining the original
HTTP Host header, TLS SNI, and certificate verification; environment proxies
are not consulted. This closes the validate-one-address/send-to-another gap
from DNS rebinding or proxy configuration.

Webhook targets are public HTTPS endpoints by default. URL userinfo,
hop-by-hop headers, and loopback/private/link-local/reserved addresses are
rejected. An intentional on-premises integration must list its exact hostname
in `WEBHOOK_ALLOWED_PRIVATE_HOSTS`; plain HTTP additionally requires
`WEBHOOK_ALLOWED_HTTP_HOSTS`. These controls complement, rather than replace,
host-level egress firewalling.

Provider response bodies, URLs, credentials, chat IDs, and recipients are not
stored in result messages. Operational logs expose only allow-listed fields:
the alert/channel, stable error code, numeric provider status, retry delay,
attempt, and whether provider acceptance is unknown.

## Durability boundary

The alert row and one outbox row per concrete destination commit in the same
PostgreSQL transaction. Workers claim due rows with expiring, fenced leases;
retry deadlines, attempt counts, terminal outcomes, and escalation schedules
therefore survive a process or device restart. Email recipients are separate
rows, so one recipient's rate limit does not delay another destination.
Claims and provider attempts are counted separately: a crash before external
I/O cannot consume the provider budget. Each row has an absolute deadline that
starts when that row becomes eligible, so a 30-minute escalation still gets
its full retry window instead of expiring while it is merely scheduled.

Delivery is **at least once**. Stable webhook idempotency keys and email
Message-IDs let cooperative providers collapse ambiguous retries. Telegram has
no client idempotency key, so a connection loss after provider acceptance can
still produce a duplicate. Acceptance ambiguity is retained as history even
if a later attempt is delivered or terminal.

During the first upgrade, active alerts receive only their missing escalation
rows, idempotently and from the original alert timestamp. Initial sends are not
backfilled because the old system cannot prove whether they were already
accepted. For the same reason, already-due legacy escalation steps are skipped;
only future steps are admitted. The upgrade scan uses bounded keyset batches:
one batch completes before workers start and the rest continues while the API
is serving. Acknowledging, resolving, snoozing, or marking an alert
false-positive cancels all escalation rows—including leased rows—in the same
database transaction. An external request that already started can still
finish; its stale completion is fenced and cannot schedule another attempt.

Delivery destinations are represented by fixed hashes rather than raw values.
Each attempt resolves and sends from one immutable configuration snapshot. If
an operator changes a destination, old obligations end terminally rather than
being validated against one destination and sent to another. Explicitly
selected but incomplete/unsupported routes become visible terminal obligations
instead of disappearing silently. After repairing configuration, an
administrator can discover safe, redacted delivery state with
`GET /api/alert-deliveries?state=terminal`, then replay recent terminal work with
`POST /api/alert-deliveries/{delivery_id}/replay`. Work with any prior ambiguous
acceptance requires `{"allowAmbiguous": true}`. The authorized replay request is
audited before work can become eligible, so an audit failure cannot produce an
unaudited resend.

The durable hash also binds the provider account/routing boundary, not only the
visible destination. Telegram binds the numeric bot ID encoded in a standard
bot token, allowing the secret portion of that same bot's token to rotate; an
opaque/non-standard token is bound in full and therefore fails closed on
rotation. Email binds recipient, SMTP host/port/user, and From address while
allowing only the SMTP password to rotate. Webhooks bind the exact URL and all
configured header names and routing values. Authorization/API-key values may
rotate only when the webhook has a non-empty `account_id` or one of these
non-empty, stable headers: `X-Account-ID`, `X-Tenant-ID`,
`X-Organization-ID`, `X-Org-ID`, `X-Workspace-ID`, or `X-Project-ID`.
Without that explicit account assertion, the authorization value remains part
of the hash, so an arbitrary credential rotation cannot move queued work to a
different tenant. Every boundary is stored only as a full SHA-256 digest.
Destination-only rows created by an older release deliberately fail closed and
can be inspected/replayed by an administrator after migration.

Escalation steps saved through the API are marked explicitly active and freeze
their severity scope. Admission rejects a step unless its provider is enabled,
complete, and accepts every severity in that scope. Enabled step IDs must be
unique within a channel so two obligations cannot collapse onto the same
durable key. Later provider updates may not disable an active step's provider,
make it incomplete, or remove a required severity. The two
unmarked examples seeded by older releases remain dormant unless an operator
explicitly saves them with `enabled: true`; merely configuring a provider does
not activate sample data. A custom legacy step, or any step marked
`enabled: true`, is treated as intentional: if it is invalid, the resolver
records a visible terminal obligation instead of silently dropping it. Set
`enabled: false` explicitly to keep a step dormant.

Due work is selected by severity first. A waiting row is promoted one priority
band every `ALERT_DELIVERY_PRIORITY_AGING_SECONDS` (60 seconds by default), so
new P1 work is not trapped behind an old P4 backlog while lower-priority work
still cannot starve indefinitely.

Delivered/cancelled history is retained for 14 days by default; terminal
history is retained for 90 days. Pending or leased rows are never removed by
retention. Health metrics separate due work from future-scheduled escalation
rows so a future deadline does not look like an overdue backlog.
Health becomes degraded when the outbox database is unavailable, its claimer or
provider workers stop, or due work remains older than the bounded poll window.
It also degrades when the alert-persistence worker stops, a persistence failure
has not yet been followed by a successful write, or accepted work is older than
`ALERT_PERSISTENCE_STALE_SECONDS` (30 seconds by default).

## VLM alert admission

The VLM must return an explicit `STATUS: SAFE` or `STATUS: VIOLATION` verdict;
the fallback parser is negation-aware for older models. SAFE analyses update
the camera result but do not create informational alerts. A persistent unsafe
scene creates one P2 incident until a later SAFE analysis resets it, including
while alert persistence is still pending. Results that return after camera stop
are discarded, and restart is deferred while the prior VLM call remains alive.
`SAFETYLENS_VLM_TIMEOUT_SECONDS` bounds the provider read timeout (60 seconds by
default), and the configured `OLLAMA_URL` is honored by the backend. Analysis
intervals are clamped to at least five seconds at runtime and invalid resource
settings are rejected by the configuration API, preventing a corrupt value
from turning the VLM loop into a hot spin.

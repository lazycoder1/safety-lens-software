# safety-lens-software

## Edge / model-server split

The backend can run in the original all-in-one mode or in a lightweight edge mode that sends decoded frames to a GPU model server. In split mode the frontend still talks only to the edge backend; streams, alerts, camera CRUD, and WebSockets keep the same API paths.

See [docs/model-server-split.md](docs/model-server-split.md) for the runtime contract and startup commands.

Before recreating an edge container, follow the
[runtime state migration and verification gate](docs/runtime-state.md). Runtime
credentials, license/heartbeat files, diagnostics, and models are deliberately
kept outside immutable images.

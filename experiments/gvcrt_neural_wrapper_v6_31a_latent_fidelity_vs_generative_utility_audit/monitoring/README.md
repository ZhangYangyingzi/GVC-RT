# Hourly read-only supervision

The user requested supervision every hour. `hourly_monitor.py` runs independently in a persistent screen session, checks immediately and then every 3600 seconds, and stops after a terminal experiment status. It never changes coding workers, source/checkpoint weights, evaluation results or publication state.

`latest.md` is the readable current report; `latest.json` is its structured equivalent; `history/` retains timestamped reports. Times use Asia/Shanghai. Reports cover complete per-point receipts, per-GPU progress and process liveness, sanity, integrity, publication and stalled/failure warnings. A receipt count is a progress measure; the evaluation driver's final integrity check validates the artifacts.

**No outbound notification transport is configured.** Writing these reports does not send messages to the chat, email or any external channel. Configuring active delivery requires an available notification mechanism and a user-selected destination.

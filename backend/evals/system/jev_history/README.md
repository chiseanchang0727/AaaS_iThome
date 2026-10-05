# Conversations run with Jev as the context manager

The **Jev vs full** page pairs each conversation here with the one in
`../chat_history/` that asked the same questions, and compares them.

```
jev_history/
  <thread_id>.jsonl              run with server.context_filter on (api/history.py format)
  artifacts/<thread_id>/...      optional: files it made
```

To make missing twins (each unpaired conversation's questions asked in the
other mode), from `backend/`:

    uv run --env-file ../.env python -m evals.system.compare --record

Everything here but this README is git-ignored. The folder is
`server.eval_jev_history_dir` in config.

# Conversations to evaluate

The System eval page's **Evaluate all conversation history** button (and
`python -m evals.system.run --history`) judges the conversations in this
folder, not the app's own `conversations/`.

To judge a conversation, copy its history file here:

```
chat_history/
  <thread_id>.jsonl              one per conversation (api/history.py format)
  artifacts/<thread_id>/...      optional: files it made (from reports/artifacts/<thread_id>/)
```

Everything here but this README is git-ignored. The folder is
`server.eval_history_dir` in config (default `evals/system/chat_history`).

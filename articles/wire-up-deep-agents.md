# Wiring Up a Deep Agent: Tools, Skills, and Making It Read the Manual

A hot topic these days is the Jev model, which is a great option for automation. Automation is full of simple routing (go A/B/C/D), and the output of current frontier models is far richer than that kind of task needs. The properties of the Jev model handle this well.

There is a great article about using Jev in a coding agent system: [Jev Engineering For Coding Agents](https://github.com/yibie/jev-engineering-zh/blob/main/assets/original.pdf).
If I have bandwidth during this series, I might implement it with LangGraph or another framework.

We already have a basic understanding of the components of Deep Agents. Today let's wire them up and run tests to see how this ReAct (Reasoning and Acting) system works.

Everything below runs against the repo from the previous posts: a Postgres table `videos` holding the US YouTube trending dataset, and the read-only `query_database()` datasource we built last time.

The same question goes through every stage, so we can watch the answer change:

> **Which video category gets the most views?**

It looks harmless, but it has a trap. We come back to it at the end.

---

## Creating instance

Install the two packages:

```bash
uv add deepagents langchain-anthropic
```

Adding a section to `config.yml` keeps the model choice out of the code:

```yaml
agent:
  model: anthropic:claude-sonnet-4-6
  # Resolved against the working directory. Read-only to the agent.
  skills_dir: skills
```

```python
# config/agent.py
class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    """`provider:model`, as `init_chat_model` reads it."""

    skills_dir: Path
    """Directory of `<skill-name>/SKILL.md` folders, mounted at /skills/."""
```

The smallest possible deep agent is a model plus a system prompt:

```python
from deepagents import create_deep_agent
from config import cfg

SYSTEM_PROMPT = """\
You are a data analyst for a YouTube trending-videos dataset stored in PostgreSQL.
Answer questions by querying the database; never guess a number. Keep the final
answer short: the result, then one line on how it was computed.
"""

agent = create_deep_agent(model=cfg.agent.model, system_prompt=SYSTEM_PROMPT)
```

Even this "empty" agent isn't empty. `create_deep_agent` wraps LangChain's `create_agent` and adds a stack of middleware by default:

| Built in | What it gives the agent |
|---|---|
| `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` over a backend (in-memory state by default) |
| `SubAgentMiddleware` | a `task` tool that hands work to a general-purpose subagent |
| `SummarizationMiddleware` | compacts the history when the context gets long |
| `AnthropicPromptCachingMiddleware` | cache breakpoints for Anthropic models |

So what we get back is a compiled LangGraph graph that already runs a ReAct loop: the model thinks, calls a tool, reads the result, and repeats until it answers without calling a tool.

## Write a test to see how it responds

Invoking the agent is just `ainvoke` with a list of messages. To actually *see* the loop, I print each step of the returned message history (see `scripts/ask.py`):

- **THINK**: text the model wrote alongside its tool calls
- **ACT**: a tool call
- **OBSERVE**: the tool's result

```python
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Which video category gets the most views?"}]}
)

for m in result["messages"]:
    if isinstance(m, AIMessage):
        for call in m.tool_calls:
            print(f"ACT     {call['name']}({call['args']})")
    elif isinstance(m, ToolMessage):
        print(f"OBSERVE {m.content[:400]}")
print(result["messages"][-1].content)
```

Output with no tools wired:

```text
<!-- TODO: paste output of
     uv run --env-file .env python scripts/stages.py bare -->
```

<!-- TODO (commentary): what did it do? Expected: it has filesystem tools but no
     data, so it either says it can't access the database or pokes around the empty
     filesystem with ls/glob first. Either way, no number. That is the point:
     the loop works, but the agent has nothing to act on. -->

## Wire up tools

A tool is just a function with a docstring. **The docstring is the prompt**: it's the only thing the model reads when deciding whether to call the tool and what to pass in.

```python
# agent/tools.py
@tool
async def query_database(sql: str) -> str:
    """Run one read-only PostgreSQL query and return the rows as JSON.

    The data lives in a single table, `videos`. A result too large for the
    context is refused with its size, so the query can be narrowed.
    """
    try:
        rows = await _query_database(sql)
    except (TooManyRows, asyncpg.PostgresError) as e:
        return f"ERROR: {e}"
    return format_result(rows, cfg.agent.max_result_tokens)
```

Three decisions here:

1. **Errors come back as text, not exceptions.** A bad column name raises inside Postgres; returning `ERROR: column "view" does not exist` as the tool result lets the model read it and fix its next query. An exception would end the run instead. This is the *Observe* half of ReAct doing its job.
2. **Safety is not in this function.** The agent connects as `agent_ro`, a role that only holds `SELECT ON videos`. There's no keyword filtering of SQL strings, because the database enforces what the role can do far better than a regex ever could.
3. **The result has a size budget.** `format_result` refuses anything too big for the context window. More on that [below](#keeping-query-results-out-of-the-context-window).

```sql
CREATE ROLE agent_ro LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE <db> TO agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON videos TO agent_ro;
```

Wiring the tool in is one argument:

```python
agent = create_deep_agent(
    model=cfg.agent.model,
    tools=[query_database],
    system_prompt=SYSTEM_PROMPT,
)
```

`tools=` is additive: the built-in filesystem and `task` tools stay.

```text
<!-- TODO: paste output of
     uv run --env-file .env python scripts/stages.py tools -->
```

## The trap in the question

Before adding skills, let's look at what the "right" answer actually is, because the data has a shape the model can't see from the schema alone.

**One row is one video on one trending day.** A video that trends for a week appears seven times, and its `views` grows each day.

| | count |
|---|---:|
| rows in `videos` | 40,949 |
| distinct videos | 6,351 |

The obvious query sums the same video's views once for every day it trended:

```sql
SELECT category_name, SUM(views) FROM videos GROUP BY 1 ORDER BY 2 DESC;
```

| Query | Music "total views" |
|---|---:|
| `SUM(views)` over raw rows | **40.1B** |
| collapse to one row per video (`MAX(views)`), then `SUM` | **4.82B** |

That's an **8.3× overcount**, and nothing errors. Worse, "the most views" is ambiguous: Music wins on *total* only because it has 799 videos. Per video, using the median (views are extremely right-skewed) and ignoring categories with fewer than 30 videos, the ranking flips:

| Category | Videos | Median views per video |
|---|---:|---:|
| **Gaming** | 103 | **1.32M** |
| Music | 799 | 1.10M |
| Film & Animation | 321 | 911K |
| Comedy | 547 | 788K |
| Entertainment | 1,621 | 606K |

<!-- TODO (commentary): compare with the `tools` run above. Did the agent SUM raw
     rows (40.1B)? Did it dedupe on its own? Did it pick total or per-video?
     Whatever it did, this is the gap skills are for. -->

None of this is SQL skill. It's **domain knowledge about this table**, and it's exactly what a skill is for.

## Create skills

A skill is a folder with a `SKILL.md`: YAML frontmatter (`name`, `description`) plus Markdown instructions. Deep Agents uses the same format as Anthropic's Agent Skills.

```text
skills/
├── query_database/
│   └── SKILL.md      # schema + counting rules
└── trend-report/
    └── SKILL.md      # house format for a written report
```

`skills/query_database/SKILL.md` (the folder is named after the tool it governs; that matters in the last section):

````markdown
---
name: query_database
description: Schema and counting rules for the `videos` table, which query_database reads. Read before writing any SQL; naive SUM/COUNT over this table gives wrong answers.
---

# Querying the `videos` table

## What a row is

One row is **one video on one trending day**. A video that trended for a week
appears seven times, and its `views` grows each day. The table holds 40,949 rows
but only 6,351 distinct videos, all US, trending between 2017-11-14 and 2018-06-14.

## Counting rules

- **Per-video questions** ("most viewed video", "views by category", "top
  channels"): collapse to one row per video first, keeping the peak view count,
  then aggregate.

  ```sql
  WITH per_video AS (
      SELECT video_id,
             MAX(views)         AS views,
             MAX(likes)         AS likes,
             MIN(category_name) AS category_name,
             MIN(channel_title) AS channel_title
      FROM videos
      GROUP BY video_id
  )
  SELECT category_name, COUNT(*) AS n_videos,
         PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY views) AS median_views
  FROM per_video
  GROUP BY category_name
  ORDER BY median_views DESC;
  ```

- **Activity-over-time questions** ("views per week", "busiest day"): keep one row
  per `(video_id, trending_date)` and do not collapse; a video *should* count on
  every day it trended.
- Prefer the **median** over the mean for views: the distribution is heavily
  right-skewed and one viral video decides an average.
- Say how many videos a category or channel rests on; drop groups under ~30
  videos from rankings, or flag them.
- Never `COUNT(*)` to mean "number of videos". Use `COUNT(DISTINCT video_id)`.
````

`skills/trend-report/SKILL.md` is shorter: headline with the number in it, a table of at most 8 rows, one caveat. It's there so the agent has to *choose* between skills instead of reading the only one available.

**The `description` is the most important line in the file.** It's the only part the agent sees up front. Deep Agents uses *progressive disclosure*: the system prompt lists every skill's name, description and path, and the agent calls `read_file` on the full `SKILL.md` only when it decides the skill applies. Ten skills cost ten lines of context, not ten documents. A vague description means the skill never gets read.

## Wire up skills

Skills are read through a **backend**, the filesystem abstraction behind the agent's file tools. I don't want the agent writing scratch files into my repo, so I use a `CompositeBackend`:

- `/skills/` is routed to the real `skills/` directory on disk
- everything else goes to `StateBackend`: in-memory, and gone when the run ends

```python
# agent/__init__.py
def build_agent(*, require_sql_skill: bool = True):
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": FilesystemBackend(root_dir=cfg.agent.skills_dir, virtual_mode=True),
        },
    )

    return create_deep_agent(
        model=cfg.agent.model,
        tools=[query_database],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        skills=["/skills/"],
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny"),
        ],
        middleware=...,  # next section
    )
```

- `skills=["/skills/"]` adds `SkillsMiddleware`, which scans that path for `*/SKILL.md` and injects the list into the system prompt.
- `virtual_mode=True` anchors paths to `skills/`, so the agent can't `../` its way out.
- The `deny` permission stops the agent from editing its own instructions. Without it, `edit_file` on a `SKILL.md` would be a real write to the repo.

```text
<!-- TODO: paste output of
     uv run --env-file .env python scripts/stages.py skills -->
```

<!-- TODO (commentary): did it read query_database/SKILL.md before querying? If yes, show the
     read_file step in the trace and the corrected numbers. If no, this is the
     setup for the next section: the skill is there, the description says "read
     before writing any SQL", and the model still decided it didn't need it. -->

## Force agent to read specific skills

The skills prompt *asks* the model to read a skill when it applies. The model decides whether it applies. For an analyst agent, "I already know SQL" is a very tempting conclusion, and it leads straight to the 8.3× overcount.

There are two ways to make it mandatory:

| | Where it lives | Guarantee |
|---|---|---|
| "You MUST read /skills/query_database/SKILL.md first" in the system prompt | prompt | a strong suggestion |
| Middleware that blocks the tool until the skill is read | code | enforced on every call |

I went with the middleware, using a simple convention: **the skill for tool `<name>` lives at `/skills/<name>/SKILL.md`**. That's why the SQL skill's folder is `query_database/`. With that convention, "which skill governs this tool?" needs no lookup table.

### Where to hook in: `after_model`

A LangChain middleware can intercept at several points. The two that matter here:

| Hook | Runs | To block a call it... |
|---|---|---|
| `wrap_tool_call` | around each tool execution, inside the tools node | returns a fake `ToolMessage` instead of calling the handler |
| `after_model` | right after the model responds, before the tools node | writes `ToolMessage`s into state and `jump_to="model"` |

`after_model` is the earlier point. A blocked call never reaches the tools node at all, and never reaches any other `after_model` hook either. That matters once there's a human-in-the-loop approval step: you don't want a person asked to approve a call that's about to be rejected anyway.

```python
# agent/middleware.py
class SkillEnforcerMiddleware(AgentMiddleware):
    """Blocks enforced tool calls (in after_model) until the tool's skill is read."""

    def __init__(self, skills_dir: Path, target_skills: str | list[str] | None = None) -> None:
        super().__init__()
        self.skills_dir = Path(skills_dir)
        if target_skills is None:
            self._targets: set[str] | None = None      # enforce every tool that has a SKILL.md
        elif isinstance(target_skills, str):
            self._targets = {target_skills}
        else:
            self._targets = set(target_skills)

    def _skill_virtual_path(self, tool_name: str) -> str:
        # The path the *agent* sees, routed by CompositeBackend to skills/ on disk.
        return f"/skills/{tool_name}/SKILL.md"

    def _skill_was_read(self, tool_name: str, messages: list) -> bool:
        skill_path = self._skill_virtual_path(tool_name)
        for msg in messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                if tc["name"] == "read_file" and skill_path in self._get_path_arg(tc["args"]):
                    return True
        return False

    def _check_skills(self, state: AgentState) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        blocked = {
            tc["id"]
            for tc in last_ai_msg.tool_calls
            if self._is_enforced(tc["name"]) and not self._skill_was_read(tc["name"], messages)
        }
        if not blocked:
            return None

        replies = [
            self._block_message(tc["name"], tc["id"])
            if tc["id"] in blocked
            else self._skipped_message(tc["name"], tc["id"])
            for tc in last_ai_msg.tool_calls
        ]
        return {"messages": replies, "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._check_skills(state)
```

(`_is_enforced`, `_get_path_arg` and the two message builders are small helpers; see `agent/middleware.py` for the full class.)

`@hook_config(can_jump_to=["model"])` declares the jump up front so LangGraph adds the extra edge: from this hook straight back to the model node, skipping the tools node.

```python
middleware = [SkillEnforcerMiddleware(cfg.agent.skills_dir, target_skills="query_database")]
```

`after_model` hooks run in **reverse** list order, so the enforcer goes **last** in the list. That way it runs first and checks the model's calls before anything else sees them.

The block isn't a crash. It's an **observation**, just like a SQL error. The model reads `Skill check failed: ... Call read_file with path='/skills/query_database/SKILL.md' first, then retry.`, reads the skill, and retries. We didn't take control away from the ReAct loop; we fed it a better observation.

### The gotcha: parallel tool calls

Models often emit several tool calls in one turn. Suppose the turn is `[ls("/"), query_database(...)]`. The enforcer blocks `query_database` and jumps back to the model, which skips the tools node **for the whole turn**, so `ls` never runs either. If only the blocked call gets a `ToolMessage`, the history now holds a `tool_use` with no matching `tool_result`, and the next model request is rejected by the API. deepagents' `PatchToolCallsMiddleware` doesn't save you here: it only repairs dangling calls in `before_agent`, at the start of a run.

So on a block, **every** call in the turn gets a reply: the blocked ones get the skill-check error, and the others get "Not run: another call in this turn failed its skill check." The reverse case needs no special handling: if the model reads the skill *and* queries in the same turn, `_skill_was_read` sees the `read_file` in that same message and lets both through, because the tools node runs them together.

### Testing it

`_check_skills` is plain Python over a list of messages, so the rules can be tested without a model:

```python
def test_query_is_blocked_until_the_skill_is_read():
    result = check(AIMessage(content="", tool_calls=[query()]))
    assert result["jump_to"] == "model"
    [reply] = result["messages"]
    assert reply.tool_call_id == "q1" and reply.status == "error"


def test_reading_the_skill_in_the_same_turn_counts():
    assert check(AIMessage(content="", tool_calls=[read(SKILL), query()])) is None


def test_every_call_in_a_blocked_turn_gets_a_result():
    turn = AIMessage(content="", tool_calls=[call("ls", {"path": "/"}, "l1"), query()])
    replies = check(turn)["messages"]
    assert {r.tool_call_id for r in replies} == {"l1", "q1"}
```

To check the wiring as well (that `jump_to` really loops back, and that `/skills/` really routes to disk), I ran the full graph with a **scripted fake model** (`FakeMessagesListChatModel`) that deliberately queries first, in parallel with an `ls`. The tools, backend and Postgres are all real:

```text
AI   [('ls', 'a'), ('query_database', 'b')]
TOOL a Not run: another call in this turn failed its skill check. Re-issue 'ls' if you still need it.
TOOL b Skill check failed: you must read the skill for 'query_database' before using it. Call read_file with path='/s…
AI   [('read_file', 'c')]
TOOL c @@ lines 1-5 of 55 | next offset 5 @@ --- name: query_database description: Schema and counting rules for the …
AI   [('query_database', 'd')]
TOOL d [{"n": 6351}]
AI   6351 videos.
```

Blocked, sent back, skill read through the `/skills/` route, query retried against the real database.

> Naming the folder after the tool has one cost: the Agent Skills spec wants lowercase-and-hyphen names, so deepagents logs a warning for `query_database`. It still loads the skill.

And with a real model:

```text
<!-- TODO: paste output of
     uv run --env-file .env python scripts/stages.py forced -->
```

## Keeping query results out of the context window

The agent writes whatever SQL the question needs, and everything that SQL returns becomes a tool result. A tool result doesn't just cost tokens once. It stays in the conversation history, and the API is stateless, so **every later model call sends it again**. One oversized result early in a run gets paid for on every turn after it.

So the tool needs a limit. The question is: a limit on what?

### Rows are the wrong unit

The datasource already had `max_rows: 1000`. But what fills the context is size, not row count:

| Query | Rows | ~Tokens |
|---|---:|---:|
| views by category, aggregated in SQL | 16 | 224 |
| top 5 videos | 5 | 134 |
| 5 sample rows, all columns | 5 | 820 |
| 1,000 rows × 3 columns | 1,000 | 19,515 |
| 1,000 rows × all columns | 1,000 | 165,956 |

The last two have the same row count and differ in size by 8×. A row limit can't tell them apart.

### A size budget, not a rule about SQL

The fix is to limit the result's **size** and say nothing about how the query should get there:

```yaml
# config.yml
agent:
  # A tool result bigger than this is refused rather than put into the model's
  # context. Approximate: ~4 characters per token.
  max_result_tokens: 5000
```

```python
# agent/tools.py
def format_result(rows: list[dict], max_tokens: int) -> str:
    """Rows as JSON, or a refusal saying how big they were."""
    content = json.dumps(rows, ensure_ascii=False)
    tokens = estimate_tokens(content)
    if tokens <= max_tokens:
        return content
    columns = len(rows[0])
    return (
        f"ERROR: result is ~{tokens:,} tokens ({len(rows):,} rows x {columns} columns), "
        f"over the limit of {max_tokens:,}. Nothing was returned; narrow the query "
        f"to what the answer needs."
    )
```

Three choices in there:

- **The tool doesn't dictate how to shrink the result.** The model can select fewer columns, add a `LIMIT`, or aggregate, whichever fits the question. "Show me five sample rows" is a perfectly good query, and it passes at 820 tokens.
- **Refuse, don't truncate.** Cutting the result at the budget would be easy, but a truncated result looks complete. The model would answer from part of the data without knowing it.
- **The refusal reports the size.** `827 rows x 3 columns` tells the model which way to cut. And the refusal itself costs about 50 tokens instead of 25,000.

`max_rows` stays, but its job has changed. It now protects the database and the Python process: its cursor stops fetching at 1,001 rows, so `SELECT * FROM videos` never pulls 40,949 rows into memory. Context size is the token budget's job.

### Why not let deepagents handle it?

deepagents has its own safeguard: any tool result over about 20,000 tokens is saved to a file under `/large_tool_results/`, and only a preview enters the context. The model can then page through the file with `read_file`.

That's the right default for a tool whose result is expensive to reproduce. For SQL it's the wrong trade-off. Paging through a saved result takes several turns, and every page stays in the history. Writing a narrower query costs one cheap call. So the tool refuses at 5,000 tokens, well before deepagents' 20,000, and eviction remains the backstop for every other tool.

### What the model does with a refusal

A question that can't fit, run with Sonnet 4.6:

```text
Q: List all the Music videos.

[1] ACT     read_file({"file_path": "/skills/query_database/SKILL.md", "limit": 1000})
    OBSERVE @@ lines 1-43 of 43 @@ --- name: query_database ...
[2] ACT     query_database({"sql": "SELECT DISTINCT video_id, title, channel_title
                                    FROM videos WHERE category_name = 'Music' ORDER BY title"})
    OBSERVE ERROR: result is ~25,178 tokens (827 rows x 3 columns), over the limit of 5,000.
            Nothing was returned; narrow the query to what the answer needs.
[3] ACT     query_database({"sql": "SELECT DISTINCT video_id, title, channel_title
                                    FROM videos WHERE category_name = 'Music' ORDER BY title
                                    LIMIT 100"})
    OBSERVE [{"video_id": "uKkbfxTHJGI", "title": "10$ Drum - Faded ( Alan Walker )", ...}, ...]

=== ANSWER ===
There are **827 distinct Music videos** in the dataset. Here's the first 100 listed
alphabetically (by title): ...

--- model turns: 4, tokens in/out: 24044/904
```

The refusal is just another observation. The model read it, added `LIMIT 100`, and told the user the list was partial. Nothing in the tool told it to use `LIMIT`; it picked that because the question asked for a list, not a summary.

**But the answer is wrong, and not because of the budget.** There are 801 Music videos, not 827. `DISTINCT video_id, title, channel_title` counts a video once per title it had, and 23 Music videos were retitled while trending. The budget did its job, keeping 25,000 tokens out of the context. Whether the query is *correct* is a separate problem. That's what the skill is for, and it's why the next section compares runs with and without it.

A single run shows the mechanism works, not that the model always recovers this well. The loop is bounded either way: LangGraph stops a run at its recursion limit.

> **A caveat on the estimate:** ~4 characters per token (the same estimate deepagents uses) is close for English and undercounts for CJK text, where one character is often a token or more. This dataset's titles are mostly English. For exact counts, Anthropic's token-counting API is an option, at the cost of an extra call per query.

## Results

<!-- TODO: fill from the four runs -->

| Stage | Read the skill? | SQL pattern | Answer | Model turns | Tokens in / out |
|---|---|---|---|---:|---:|
| bare | — | — | | | |
| tools | — | | | | |
| skills | | | | | |
| forced | yes (enforced) | | | | |
| *ground truth* | | one row per video, median | Gaming, 1.32M median (Music leads on total: 4.82B) | | |

## Takeaways

- **The loop is free; the observations are the product.** `create_deep_agent` gives you ReAct out of the box. What you actually design is what the agent *observes*: tool docstrings, error messages as text, and refusals that say what to do next.
- **Tools give an agent access; skills give it judgement.** `query_database` let the agent get *a* number. Only the skill carried the fact that one row is not one video.
- **Budget tool results by size, not by rule.** Every tool result is resent on every later turn. Cap its size, refuse rather than truncate, and report the size so the model can narrow its own query.
- **Put guarantees in code.** A description or system prompt makes a skill likely to be read. Middleware makes it certain, and you can unit-test it.
- **Keep safety below the agent.** Read-only access is a Postgres role, and skills are write-protected by a backend permission. Neither depends on the model behaving.

Next time: <!-- TODO -->

You are checking whether a data-analysis agent's final answer is grounded:
every factual claim it makes about the data must be supported by something the
agent actually observed in a tool result (query result, file, program output).

This is not a correctness check. If the agent's code computed the wrong thing
and the answer faithfully reports it, the answer is still grounded. Do not
re-check the math. Only ask: did the agent see this before saying it?

## Earlier in the conversation
(Tool results from earlier turns also count as observed; they are included
below under "Earlier tool results" if any.)
$conversation

## Question
$question

## What the agent observed
$evidence

## Final answer
$answer

## Numbers code could not find in any observation
$ungrounded

These are only candidates. A number is fine if it is plainly derived from
observed values (a sum, difference, ratio or rounding of numbers the agent
saw), or restates the question. It is unsupported if nothing observed leads to it.

## Your task
List each claim in the final answer that is not supported by the observations.
Include non-numeric claims too: names, rankings, "every day", "none", "most",
causes, trends.

- pass: no unsupported claims.
- fail: at least one unsupported claim.
- unknown: the observations are cut off or missing in a way that makes it
  impossible to tell.

Give a short overall reason. For each unsupported claim, quote it and say why
it is unsupported.

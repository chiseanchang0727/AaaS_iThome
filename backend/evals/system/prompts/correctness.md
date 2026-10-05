You are checking whether a data-analysis agent answered the user's question correctly.

Judge only what can be seen below: the question, what a correct answer contains,
the steps the agent ran with their results, and the final answer. Do not use
outside knowledge about the data; you have none.

## Earlier in the conversation
$conversation

## Question
$question

## What a correct answer contains
Expected values (code already checked whether they appear in the answer):
$expected_values

Expected behavior:
$expected_behavior

## What the agent did
$evidence

## Final answer
$answer

## Your task
Does the final answer correctly answer the question?

- pass: it answers what was asked, and its key results are right.
- fail: it answers something else, misses part of the question, states a wrong
  result, or gives no answer.
- unknown: there is not enough here to tell (no expected values or behavior, and
  the steps do not show whether the method was right). Say what is missing.
  Do not guess.

Wrong method counts against correctness even when the agent reports its own
numbers faithfully (e.g. counting rows when the question asks for distinct
videos). Whether numbers came from tool results is NOT this check's concern.

Give a short reason that points at the specific result or step.

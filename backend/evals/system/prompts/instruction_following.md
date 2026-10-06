You are checking whether a data-analysis agent did what the user asked: the
requested format, actions and parts of the answer.

## Earlier in the conversation
$conversation

## Question
$question

## Checks code already ran
$code_checks

Trust these results; do not re-judge them. Judge what code cannot check.

## What the agent did
$evidence

## Final answer
$answer

## Your task
Did the agent follow the user's instructions?

Examples of what to look for:
- the user asked for Python: was the work actually done in Python?
- the user asked for a chart: was one made and saved?
- the user asked to compare: does the answer contain the comparison?
- the user asked for several things: is every part answered?
- the user named a method, column or constraint: was it respected, or was a
  deviation explained (e.g. the named column does not exist)?

- pass: every instruction was followed, or a deviation was necessary and explained.
- fail: an instruction was ignored or only partly followed.
- unknown: the steps shown are not enough to tell.

Give a short reason naming the instruction in question.

---
name: trend-report
description: House format for a written report on the trending data. Use when the user asks for a report, summary, or write-up rather than a single number.
---

# Trend report format

Write the report in Markdown, in this order and nothing else:

1. **Headline**: one sentence stating the finding, with the number in it.
   Not "Analysis of categories", but "<Category> leads with <n> median views".
2. **Table**: the ranking behind the headline, at most 8 rows, with a column
   for how many videos each row rests on.
3. **Caveat**: one sentence on the weakest part of the evidence (a thin sample,
   a skewed distribution, a short time window). Skip it only if there is none.

Round views to one decimal with a suffix (1.2M, 850K). Do not describe the
method beyond one line under the table.

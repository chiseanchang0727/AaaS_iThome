"""Saved analyses: a recipe to make an analysis's output again from the data as it is now.

The agent saves one with its save_analysis tool (agent/save_analysis.py) when
the user asks; the Analyses page runs it again with no model involved:

    inputs   where the data comes from: a SQL query (exported to a Parquet
             file, as export_query does) or an uploaded file dataset
    script   Python that reads the inputs from $DATA_DIR and writes the
             outputs to $OUTPUT_DIR
    outputs  the files it makes (HTML for now)

Both the test run when saving and every later run go through runner.run, in
an empty scratch folder, so a recipe that saved also runs.
"""

from .models import Analysis, DatasetInput, Output, QueryInput, RunRecord
from .store import AnalysisStore

__all__ = ["Analysis", "AnalysisStore", "DatasetInput", "Output", "QueryInput", "RunRecord"]

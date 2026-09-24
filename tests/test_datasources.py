import datetime as dt
import uuid
from decimal import Decimal

import pytest

from datasources.postgres import _jsonable


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("250250000"), 250250000),       # SUM(bigint) stays exact
        (Decimal("10.50"), 10.5),
        (dt.date(2026, 9, 21), "2026-09-21"),
        (uuid.UUID(int=0), "00000000-0000-0000-0000-000000000000"),
        (b"\x01\x02", "0102"),
        ("plain", "plain"),
        (None, None),
    ],
)
def test_pg_types_survive_json_encoding(value, expected):
    assert _jsonable(value) == expected


def test_large_sums_keep_full_precision():
    """float() would round this; the agent would see a wrong total."""
    big = Decimal(2**53 + 1)
    assert _jsonable(big) == 2**53 + 1

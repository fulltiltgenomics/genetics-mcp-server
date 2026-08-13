"""Unit tests for SQL literal validation.

The db-api /query endpoint takes SQL as a string and offers no parameter binding, so these
helpers are the only place a caller-supplied value can be checked before it reaches
BigQuery. Under the SDK those values are composed by a script, not chosen by a typed tool
schema, which is what makes the check load-bearing.
"""

import pytest

from genetics_mcp_server.tools.sql_safety import (
    SqlValueError,
    quote_literal,
    quote_literal_list,
    sql_float,
    sql_int,
)


@pytest.mark.parametrize(
    "value",
    ["APOE", "HLA-DRB1", "C4orf54", "deCODE_asmQTL_CpG", "finngen", "IGH@", "MIR100HG"],
)
def test_real_symbols_and_resource_ids_are_accepted(value):
    assert quote_literal(value, name="gene") == f"'{value}'"


def test_surrounding_whitespace_is_trimmed_not_rejected():
    assert quote_literal("  APOE  ", name="gene") == "'APOE'"


@pytest.mark.parametrize(
    "payload",
    [
        "APOE' OR '1'='1",
        "APOE'); DROP TABLE x; --",
        "APOE' UNION ALL SELECT * FROM `other.secret` --",
        "APOE\\' OR TRUE --",
        'APOE" OR "1"="1',
        "APOE`",
        "APOE\nUNION SELECT 1",
        "APOE\x00",
        "APOE OR TRUE",
        "(SELECT 1)",
        "a" * 129,
        "",
        "   ",
    ],
)
def test_injection_attempts_are_rejected(payload):
    with pytest.raises(SqlValueError):
        quote_literal(payload, name="gene")


def test_comment_markers_are_harmless_inside_the_literal():
    """`-` has to stay allowed for symbols like HLA-DRB1. A `--` that cannot escape the
    quotes is not a comment, it is two characters of a string that will match no row."""
    assert quote_literal("APOE--comment", name="gene") == "'APOE--comment'"


def test_non_string_is_rejected():
    with pytest.raises(SqlValueError):
        quote_literal(123, name="gene")


def test_literal_list_quotes_each_item():
    assert quote_literal_list(["a", "b"], name="resources") == "'a', 'b'"


def test_literal_list_rejects_one_bad_item():
    with pytest.raises(SqlValueError):
        quote_literal_list(["ok", "bad' OR 1=1"], name="resources")


def test_literal_list_rejects_empty():
    with pytest.raises(SqlValueError):
        quote_literal_list([], name="resources")


def test_sql_int_coerces_and_range_checks():
    assert sql_int(500, name="window", minimum=0, maximum=1000) == "500"
    with pytest.raises(SqlValueError):
        sql_int(5000, name="window", minimum=0, maximum=1000)
    with pytest.raises(SqlValueError):
        sql_int(-1, name="window", minimum=0)


@pytest.mark.parametrize("bad", ["500 OR TRUE", "500", True, None, 1.5])
def test_sql_int_rejects_non_integers(bad):
    with pytest.raises(SqlValueError):
        sql_int(bad, name="window")


def test_sql_float_range_and_finiteness():
    assert float(sql_float(0.1, name="min_pip", minimum=0.0, maximum=1.0)) == 0.1
    for bad in (float("nan"), float("inf"), 1.5, "0.1"):
        with pytest.raises(SqlValueError):
            sql_float(bad, name="min_pip", minimum=0.0, maximum=1.0)

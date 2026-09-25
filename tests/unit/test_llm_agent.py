"""Agent plumbing: prompt blinding, strict parsing, four-class accounting.

No model is called. Every seat used here is a recorded one.
"""

import pytest

from livelane.agent.base import AgentContext
from livelane.agent.llm import (SYSTEM_PROMPT, LLMAgent, RecordedSeat, Usage,
                                build_user_prompt, parse_claude_json,
                                parse_proposal)

CTX = AgentContext(
    design="picorv32", top="picorv32", iteration=3,
    files=["picorv32.v"],
    qor={"top": "picorv32", "cells": 6691, "area_um2": 75663.8,
         "max_delay_ns": 9.4, "valid": True, "message": ""},
    timing={}, history=[{"iteration_index": 1, "accepted": 1}],
    read_file=lambda p: "module picorv32 #(\n// body\nendmodule\n")


def test_prompt_never_names_a_tool_or_the_treatment():
    """The agent may see the OBJECTIVE but never the apparatus.

    "worst-case path delay" is the thing being optimised and is identical in
    both lanes, so the word "delay" is expected. What must never appear is a
    tool name, a lane identifier, or anything about injected latency or
    wall-clock, those would tell the agent which arm it is in.
    """
    lowered = (SYSTEM_PROMPT + "\n" + build_user_prompt(CTX)).lower()
    for banned in ("yosys", "abc ", "opensta", "livehd", "lhd ", "verilator",
                   "eqy", "injected", "latency", "wall-clock", "wall clock",
                   "lane s", "lane l", "arm ", "i(0)", "i(600)", "seconds of"):
        assert banned not in lowered, f"prompt leaks {banned!r}"


def test_prompt_does_show_the_objective():
    """The one thing it must say: shorter path delay, area guarded."""
    lowered = (SYSTEM_PROMPT + "\n" + build_user_prompt(CTX)).lower()
    assert "path delay" in lowered and "area" in lowered


def test_prompt_carries_what_the_agent_needs():
    u = build_user_prompt(CTX)
    assert "picorv32" in u and "6691" in u and "75663.8" in u


def test_parses_a_well_formed_edit():
    p = parse_proposal('{"file":"a.v","old":"x","new":"y","note":"n"}')
    assert p and (p.path, p.old, p.new) == ("a.v", "x", "y")


def test_parses_an_edit_wrapped_in_prose():
    p = parse_proposal('Sure! Here is my edit:\n'
                       '{"file":"a.v","old":"x","new":"y","note":"n"}\nHope that helps.')
    assert p and p.path == "a.v"


def test_stop_is_not_an_edit():
    assert parse_proposal('{"stop": true}') is None


@pytest.mark.parametrize("bad", [
    "",                                   # empty reply
    "no json at all",                     # no object
    "{not valid json",                    # malformed
    '{"file":"a.v","old":"x"}',           # missing 'new'
    '{"file":"a.v","old":"x","new":1}',   # wrong type
    '{"file":"a.v","old":"x","new":"x"}', # a no-op edit
])
def test_unusable_replies_are_rejected_not_guessed(bad):
    assert parse_proposal(bad) is None


def test_four_class_usage_from_the_claude_envelope():
    text, u = parse_claude_json({
        "result": '{"file":"a.v","old":"x","new":"y"}',
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 100, "output_tokens": 40,
                  "cache_read_input_tokens": 9000,
                  "cache_creation_input_tokens": 500},
    }, "claude-sonnet-5", 1.5)
    assert (u.tokens_in, u.tokens_out, u.cache_read, u.cache_write) == (100, 40, 9000, 500)
    assert u.cost_usd == 0.0123 and u.cost_source == "reported"
    # Cache-read dominates billed input; two-class accounting would be ~90x off.
    assert u.billed_input == 9600
    assert parse_proposal(text) is not None


def test_missing_cost_is_flagged_not_invented():
    _, u = parse_claude_json({"result": "hi", "usage": {"input_tokens": 5}},
                             "m", 0.1)
    assert u.cost_usd is None and u.cost_source == "unknown"


def test_agent_records_one_turn_per_iteration():
    seat = RecordedSeat(transcript=[
        '{"file":"a.v","old":"x","new":"y","note":"first"}',
        'garbage that is not json',
        '{"stop": true}',
    ])
    agent = LLMAgent(seat=seat)
    assert agent.propose(CTX) is not None
    assert agent.propose(CTX) is None      # malformed
    assert agent.propose(CTX) is None      # stop
    s = agent.usage_summary()
    assert s["turns"] == 3
    assert s["malformed_replies"] == 1     # the stop is not malformed


def test_recorded_seat_costs_nothing_which_is_the_replay_story():
    agent = LLMAgent(seat=RecordedSeat(transcript=['{"stop": true}']))
    agent.propose(CTX)
    assert agent.total_cost_usd == 0.0

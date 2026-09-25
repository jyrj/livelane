"""Unit tests for partition-level proof caching. No EDA tool required.

The property that matters is not speed, it is that a cache hit can never admit
something a fresh proof would have refused. Two ways that breaks:

* **Over-normalisation.** If canonicalisation erases a real difference, two
  behaviourally different netlists hash equal and the second one inherits the
  first one's proof. That is a false accept, and in a gate it means shipping a
  functional bug.
* **Under-normalisation.** If it keeps a global counter, nothing ever matches
  and the cache is dead weight. Measured on picorv32: hashing the obligation as
  written gave 0% reuse because a one-line edit moved ``autoidx`` on all 590
  partitions.

These tests pin both directions, plus the verdict vocabulary, refuted and
undecided must stay distinct, because only the first is evidence that the
designs differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chia_livelane.formal.proof_cache import (CacheStats, PartitionProofCache,
                                              _partition_verdict)


class TestCanonicalisation:
    def test_autoidx_is_normalised(self):
        # Yosys's global next-free-id moves whenever elaboration created a
        # different number of objects ANYWHERE, including in untouched modules.
        a = b"autoidx 12359\nmodule \\foo\nend\n"
        b = b"autoidx 12361\nmodule \\foo\nend\n"
        assert PartitionProofCache.canonicalise(a) == PartitionProofCache.canonicalise(b)

    def test_generated_serials_are_renumbered(self):
        a = b"wire $auto$insbuf.cc:97:execute$3972\nwire $auto$insbuf.cc:97:execute$3973\n"
        b = b"wire $auto$insbuf.cc:97:execute$3974\nwire $auto$insbuf.cc:97:execute$3975\n"
        assert PartitionProofCache.canonicalise(a) == PartitionProofCache.canonicalise(b)

    def test_renumbering_keeps_distinct_wires_distinct(self):
        # THE false-accept guard. Deleting the serial instead of renumbering
        # would map these two onto one name and make the pair hash equal.
        two = b"wire $auto$x.cc:1:f$1\nwire $auto$x.cc:1:f$2\n"
        one = b"wire $auto$x.cc:1:f$1\nwire $auto$x.cc:1:f$1\n"
        assert PartitionProofCache.canonicalise(two) != PartitionProofCache.canonicalise(one)

    def test_source_directory_is_normalised_but_basename_is_not(self):
        a = b'attribute \\src "/tmp/run1/Alu.sv:2.8"\n'
        b = b'attribute \\src "/var/other/Alu.sv:2.8"\n'
        c = b'attribute \\src "/tmp/run1/Other.sv:2.8"\n'
        assert PartitionProofCache.canonicalise(a) == PartitionProofCache.canonicalise(b)
        assert PartitionProofCache.canonicalise(a) != PartitionProofCache.canonicalise(c)

    def test_real_logic_differences_survive(self):
        a = b"cell $ge $x\nend\n"
        b = b"cell $gt $x\nend\n"
        assert PartitionProofCache.canonicalise(a) != PartitionProofCache.canonicalise(b)

    def test_key_differs_for_different_obligations(self, tmp_path):
        p, q = tmp_path / "a.il", tmp_path / "b.il"
        p.write_bytes(b"autoidx 1\ncell $ge $x\nend\n")
        q.write_bytes(b"autoidx 9\ncell $gt $x\nend\n")
        assert PartitionProofCache.key_for(p) != PartitionProofCache.key_for(q)

    def test_key_matches_across_a_counter_shift(self, tmp_path):
        p, q = tmp_path / "a.il", tmp_path / "b.il"
        p.write_bytes(b"autoidx 1\nwire $auto$f.cc:1:g$10\n")
        q.write_bytes(b"autoidx 9\nwire $auto$f.cc:1:g$99\n")
        assert PartitionProofCache.key_for(p) == PartitionProofCache.key_for(q)


class TestStore:
    def test_miss_then_hit_roundtrip(self, tmp_path):
        cache = PartitionProofCache(tmp_path / "store")
        status = tmp_path / "status"
        status.write_text("PASS\n")
        assert cache.get("k1") is None
        cache.put("k1", status, {"verdict": "pass"})
        assert cache.get("k1")["verdict"] == "pass"
        dest = tmp_path / "out" / "status"
        assert cache.install("k1", dest) is True
        assert dest.read_text() == "PASS\n"

    def test_installing_an_absent_key_is_a_miss_not_an_error(self, tmp_path):
        cache = PartitionProofCache(tmp_path / "store")
        assert cache.install("nope", tmp_path / "x" / "status") is False

    def test_a_corrupt_entry_reads_as_a_miss(self, tmp_path):
        cache = PartitionProofCache(tmp_path / "store")
        status = tmp_path / "status"; status.write_text("PASS\n")
        cache.put("k", status, {"verdict": "pass"})
        (cache._dir("k") / "meta.json").write_text("{not json")
        assert cache.get("k") is None   # never a crash, never a hit


class TestVerdictVocabulary:
    @pytest.mark.parametrize("marker,expected", [
        ("PASS", "proven"), ("FAIL", "refuted"), ("UNKNOWN", "undecided"),
        ("ERROR", "error"), ("TIMEOUT", "timeout"),
    ])
    def test_sby_markers_map_to_the_gate_vocabulary(self, tmp_path, marker, expected):
        base = tmp_path / "strategies" / "p" / "s" / "p"
        base.mkdir(parents=True)
        (base / marker).touch()
        assert _partition_verdict(tmp_path, "p", "s") == expected

    def test_refuted_and_undecided_are_not_collapsed(self, tmp_path):
        # eqy reports both with the same summary line and the same exit code.
        # Conflating them rejects valid edits and inflates the rejection rate.
        for name, marker in (("a", "FAIL"), ("b", "UNKNOWN")):
            d = tmp_path / "strategies" / name / "s" / name
            d.mkdir(parents=True)
            (d / marker).touch()
        assert _partition_verdict(tmp_path, "a", "s") == "refuted"
        assert _partition_verdict(tmp_path, "b", "s") == "undecided"

    def test_a_missing_marker_is_an_error_not_a_pass(self, tmp_path):
        (tmp_path / "strategies" / "p" / "s" / "p").mkdir(parents=True)
        assert _partition_verdict(tmp_path, "p", "s") == "error"


def test_cache_stats_serialises():
    s = CacheStats(partitions_total=590, hits=435, misses=155, reuse_pct=73.7,
                   verdict="refuted")
    d = s.as_dict()
    assert d["hits"] == 435 and d["verdict"] == "refuted"


class TestStrategyLadder:
    """A config may declare a cheap strategy first and a heavy one as fallback.

    eqy wires them as a make chain: the first rung has no prerequisites, later
    rungs depend on the previous rung's status and short-circuit when it says
    PASS or FAIL. Two different lists of targets therefore matter, and mixing
    them up fails quietly in both directions.
    """

    def _mk(self, wd, text):
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "strategies.mk").write_text(text)
        return wd

    def test_first_rung_only_is_cacheable(self, tmp_path) -> None:
        # Only a target with NO prerequisites can be satisfied by dropping a
        # status file in place; make would rebuild any rule whose prerequisite
        # is missing or newer. `_strategy_targets` must therefore skip rung 2.
        from chia_livelane.formal.proof_cache import _strategy_targets
        wd = self._mk(tmp_path / "wd", (
            "strategies/p1/fast/status:\n\t@echo run\n\n"
            "strategies/p1/slow/status: strategies/p1/fast/status\n\t@echo run\n"))
        got = _strategy_targets(wd)
        assert [(p, s) for p, s, _ in got] == [("p1", "fast")]

    def test_final_rung_is_where_the_verdict_lives(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import _final_targets
        wd = tmp_path / "wd"
        wd.mkdir(parents=True)
        (wd / "summary_targets.list").write_text(
            "strategies/p1/slow/status\nstrategies/p2/fast/status\n")
        got = _final_targets(wd)
        assert [(p, s) for p, s, _ in got] == [("p1", "slow"), ("p2", "fast")]

    def test_partition_names_with_dots_survive(self, tmp_path) -> None:
        # Partition names are hierarchical signal paths like
        # `ibex_core.id_stage_i.instr_executing`. Splitting on the wrong
        # separator silently loses the partition.
        from chia_livelane.formal.proof_cache import _final_targets
        wd = tmp_path / "wd"
        wd.mkdir(parents=True)
        (wd / "summary_targets.list").write_text(
            "strategies/ibex_core.id_stage_i.instr_executing/smt/status\n")
        got = _final_targets(wd)
        assert got[0][0] == "ibex_core.id_stage_i.instr_executing"
        assert got[0][1] == "smt"

    def test_missing_list_is_empty_not_an_error(self, tmp_path) -> None:
        # Absent list means single-strategy config or a failed setup; callers
        # fall back to the first rung, which is correct in both cases.
        from chia_livelane.formal.proof_cache import _final_targets
        wd = tmp_path / "wd"
        wd.mkdir(parents=True)
        assert _final_targets(wd) == []

    def test_junk_lines_are_ignored(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import _final_targets
        wd = tmp_path / "wd"
        wd.mkdir(parents=True)
        (wd / "summary_targets.list").write_text(
            "\n# comment\nstrategies/p1/smt/status\nnot-a-target\n"
            "strategies/broken\n")
        assert [(p, s) for p, s, _ in _final_targets(wd)] == [("p1", "smt")]

    def test_single_strategy_lists_agree(self, tmp_path) -> None:
        # With one strategy the two lists must name the same target, or the
        # ladder change would alter results for configs that have no ladder.
        from chia_livelane.formal.proof_cache import (_final_targets,
                                                      _strategy_targets)
        wd = self._mk(tmp_path / "wd", "strategies/p1/smt/status:\n\t@echo run\n")
        (wd / "summary_targets.list").write_text("strategies/p1/smt/status\n")
        assert [(p, s) for p, s, _ in _strategy_targets(wd)] == \
               [(p, s) for p, s, _ in _final_targets(wd)]


class TestNullProofCache:
    """The `full` denominator must genuinely cache nothing.

    If this class ever reused or stored anything, every published speedup
    would quietly divide by a partly-warm baseline, and it would look like a
    smaller, more modest, more believable number, not like a bug.
    """

    def test_never_installs(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import NullProofCache
        assert NullProofCache().install("any-key", tmp_path / "status") is False

    def test_never_returns_a_hit(self) -> None:
        from chia_livelane.formal.proof_cache import NullProofCache
        c = NullProofCache()
        c.put("k", None, {})
        assert c.get("k") is None, "a put must not become a later hit"

    def test_stays_empty(self) -> None:
        from chia_livelane.formal.proof_cache import NullProofCache
        c = NullProofCache()
        for i in range(5):
            c.put(f"k{i}", None, {})
        assert len(c) == 0

    def test_matches_the_real_cache_api(self) -> None:
        # incremental_check calls install/get/put/len. A signature drift would
        # surface as a TypeError only on the `full` run, i.e. only when
        # measuring the denominator, which is the run least often exercised.
        from chia_livelane.formal.proof_cache import (NullProofCache,
                                                      PartitionProofCache)
        for name in ("install", "get", "put"):
            assert hasattr(NullProofCache, name), name
            assert callable(getattr(PartitionProofCache, name, None)), name


class TestLadderTargets:
    """All rungs of a partition's ladder, so a hit can satisfy the whole chain.

    Getting this wrong is a performance bug in one direction (make re-runs the
    rungs, as before) and a correctness bug in the other (a rung satisfied for
    the wrong partition would report someone else's verdict), so the mapping
    from rule line to partition must be exact.
    """

    def _mk(self, tmp_path, text):
        wd = tmp_path / "wd"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "strategies.mk").write_text(text)
        return wd

    def test_collects_every_rung_in_order(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import _ladder_targets
        wd = self._mk(tmp_path, (
            "strategies/p1/fast/status:\n\t@echo run\n\n"
            "strategies/p1/slow/status: strategies/p1/fast/status\n\t@echo x\n"))
        got = _ladder_targets(wd)
        assert [p.parent.name for p in got["p1"]] == ["fast", "slow"]

    def test_single_rung_has_nothing_extra_to_satisfy(self, tmp_path) -> None:
        # With one strategy the optimisation must be a no-op, or it would
        # change behaviour for every configuration that has no ladder.
        from chia_livelane.formal.proof_cache import _ladder_targets
        wd = self._mk(tmp_path, "strategies/p1/smt/status:\n\t@echo run\n")
        assert len(_ladder_targets(wd)["p1"]) == 1

    def test_partitions_are_kept_apart(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import _ladder_targets
        wd = self._mk(tmp_path, (
            "strategies/p1/fast/status:\n\t@echo run\n\n"
            "strategies/p2/fast/status:\n\t@echo run\n\n"
            "strategies/p2/slow/status: strategies/p2/fast/status\n\t@echo x\n"))
        got = _ladder_targets(wd)
        assert len(got["p1"]) == 1 and len(got["p2"]) == 2

    def test_hierarchical_partition_names_survive(self, tmp_path) -> None:
        # Partition names are dotted signal paths; a greedy or wrong split
        # would file rungs under the wrong partition.
        from chia_livelane.formal.proof_cache import _ladder_targets
        name = "ibex_core.id_stage_i.instr_executing"
        wd = self._mk(tmp_path, (
            f"strategies/{name}/fast/status:\n\t@echo run\n\n"
            f"strategies/{name}/slow/status: strategies/{name}/fast/status\n"
            "\t@echo x\n"))
        got = _ladder_targets(wd)
        assert list(got) == [name]
        assert [p.parent.name for p in got[name]] == ["fast", "slow"]

    def test_missing_makefile_is_empty_not_an_error(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import _ladder_targets
        wd = tmp_path / "wd"
        wd.mkdir()
        assert _ladder_targets(wd) == {}

    def test_paths_point_at_status_files(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import _ladder_targets
        wd = self._mk(tmp_path, "strategies/p1/smt/status:\n\t@echo run\n")
        p = _ladder_targets(wd)["p1"][0]
        assert p.name == "status"
        assert p == wd / "strategies" / "p1" / "smt" / "status"


class TestLadderIdentityInTheKey:
    """A cached result may only be reused under the SAME ladder.

    The bug this pins: with ladder [sat, smt], a partition that `sat` leaves
    UNKNOWN and `smt` proves was stored under a key naming only `sat`. Swapping
    or dropping the second rung then still produced a HIT, reusing a result
    that only the removed rung had ever established. It got faster and kept
    saying "proven", which is the worst possible way for this to fail.
    """

    def _rungs(self, tmp_path, part, names):
        return [tmp_path / "strategies" / part / n / "status" for n in names]

    def test_every_rung_is_named(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import ladder_id
        assert ladder_id(self._rungs(tmp_path, "p", ["sat", "smt"])) == "sat|smt"

    def test_changing_a_later_rung_changes_the_identity(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import ladder_id
        a = ladder_id(self._rungs(tmp_path, "p", ["sat", "smt"]))
        b = ladder_id(self._rungs(tmp_path, "p", ["sat", "pdr"]))
        assert a != b, "a different second rung must not reuse the first's proofs"

    def test_dropping_a_rung_changes_the_identity(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import ladder_id
        a = ladder_id(self._rungs(tmp_path, "p", ["sat", "smt"]))
        b = ladder_id(self._rungs(tmp_path, "p", ["sat"]))
        assert a != b, "removing the rung that proved it must invalidate"

    def test_adding_a_rung_changes_the_identity(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import ladder_id
        a = ladder_id(self._rungs(tmp_path, "p", ["smt"]))
        b = ladder_id(self._rungs(tmp_path, "p", ["smt", "induct"]))
        assert a != b

    def test_order_is_significant(self, tmp_path) -> None:
        # [sat, smt] and [smt, sat] answer different questions: eqy reports the
        # verdict of whichever rung settles a partition first.
        from chia_livelane.formal.proof_cache import ladder_id
        a = ladder_id(self._rungs(tmp_path, "p", ["sat", "smt"]))
        b = ladder_id(self._rungs(tmp_path, "p", ["smt", "sat"]))
        assert a != b

    def test_single_rung_identity_is_just_its_name(self, tmp_path) -> None:
        # Must match what a single-strategy config produced before ladders
        # existed, or every existing single-rung cache entry silently misses.
        from chia_livelane.formal.proof_cache import ladder_id
        assert ladder_id(self._rungs(tmp_path, "p", ["smt"])) == "smt"


class TestQuestionDigest:
    """A cached proof is only reusable for the SAME QUESTION.

    The bug this pins was live and silent: `ladder_id` encoded the strategy's
    NAME, so two configs both naming a strategy `smt` collided even when one
    asked `depth 10, smtbmc yices` and the other `depth 40, smtbmc z3`. A
    bounded depth-10 result was served as the answer to a depth-40 question.
    The run got faster and still said "proven", the failure mode that does
    not announce itself.
    """

    def _cfg(self, tmp_path, body, name="smt"):
        c = tmp_path / f"{abs(hash(body)) % 99999}.eqy"
        c.write_text(f"[gold]\nread_verilog a.v\n\n[gate]\nread_verilog b.v\n\n"
                     f"[collect *]\n\n[strategy {name}]\n{body}")
        return c

    def test_depth_change_changes_the_question(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import question_digest
        a = question_digest(self._cfg(tmp_path, "use sby\ndepth 10\n"))
        b = question_digest(self._cfg(tmp_path, "use sby\ndepth 40\n"))
        assert a != b, "a bounded depth is part of what was proven"

    def test_engine_change_changes_the_question(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import question_digest
        a = question_digest(self._cfg(tmp_path, "use sby\nengine smtbmc yices\n"))
        b = question_digest(self._cfg(tmp_path, "use sby\nengine smtbmc z3\n"))
        assert a != b, "a proof is only valid for the prover that produced it"

    def test_same_question_same_digest(self, tmp_path) -> None:
        # The fix must not void every legitimate hit.
        from chia_livelane.formal.proof_cache import question_digest
        body = "use sby\nengine smtbmc yices\ndepth 10\n"
        assert question_digest(self._cfg(tmp_path, body)) == \
               question_digest(self._cfg(tmp_path, body))

    def test_strategy_name_alone_is_not_enough(self, tmp_path) -> None:
        # Both are named `smt`; only the body differs. This is exactly the
        # collision that produced the false hit.
        from chia_livelane.formal.proof_cache import question_digest
        a = question_digest(self._cfg(tmp_path, "use sby\ndepth 10\n", "smt"))
        b = question_digest(self._cfg(tmp_path, "use sby\ndepth 99\n", "smt"))
        assert a != b

    def test_ladder_order_changes_the_question(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import question_digest
        c1 = tmp_path / "l1.eqy"
        c1.write_text("[collect *]\n\n[strategy fast]\nuse sat\ndepth 3\n\n"
                      "[strategy smt]\nuse sby\ndepth 10\n")
        c2 = tmp_path / "l2.eqy"
        c2.write_text("[collect *]\n\n[strategy smt]\nuse sby\ndepth 10\n\n"
                      "[strategy fast]\nuse sat\ndepth 3\n")
        assert question_digest(c1) != question_digest(c2), \
            "eqy reports the verdict of whichever rung settles a partition first"

    def test_non_strategy_text_is_ignored(self, tmp_path) -> None:
        # The workdir path and the source list are not the question. Including
        # them would void the store on every run for no soundness gain.
        from chia_livelane.formal.proof_cache import question_digest
        c1 = tmp_path / "x1.eqy"
        c1.write_text("[gold]\nread_verilog /one/a.v\n\n[collect *]\n\n"
                      "[strategy smt]\nuse sby\ndepth 10\n")
        c2 = tmp_path / "x2.eqy"
        c2.write_text("[gold]\nread_verilog /two/b.v\n\n[collect *]\n\n"
                      "[strategy smt]\nuse sby\ndepth 10\n")
        assert question_digest(c1) == question_digest(c2)

    def test_missing_cfg_does_not_raise(self, tmp_path) -> None:
        from chia_livelane.formal.proof_cache import question_digest
        assert isinstance(question_digest(tmp_path / "gone.eqy"), str)

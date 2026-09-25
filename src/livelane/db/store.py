"""SQLite-backed variant tree and measurement ledger.

The harness owns this file exclusively.  The agent may only ever *read* it, and
only through the ``history()`` tool, it never writes a score for itself.  That
separation is the reason a reviewer can believe the numbers, so it is enforced
here by having no agent-facing write path at all.

WAL mode is on because the agent's read-only ``history()`` query runs while the
harness is still writing the current iteration; in the default rollback journal
that reader would block or see a locked database.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

SCHEMA_VERSION = "1"
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _jdump(obj: Any) -> str | None:
    return None if obj is None else json.dumps(obj, sort_keys=True)


@dataclass
class RunHandle:
    """A live run. Holds the monotonic origin for every wall_offset_s."""

    run_id: str
    store: "VariantStore"
    t0: float

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


class VariantStore:
    def __init__(self, path: str | Path, *, verbose: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA_PATH.read_text())
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', ?)",
            (SCHEMA_VERSION,),
        )
        got = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key='version'"
        ).fetchone()[0]
        if got != SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} was created by schema version {got}, this code is "
                f"{SCHEMA_VERSION}. Refusing to mix measurements across schemas."
            )

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # --- runs ---------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        *,
        design: str,
        lane: str,
        arm: str,
        model: str,
        seed: int,
        arm_delay_s: float = 0.0,
        arm_delay_mode: str = "additive",
        arm_delay_virtual: bool = False,
        budget_wall_s: float | None = None,
        budget_iters: int | None = None,
        liberty_path: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        note: str | None = None,
    ) -> RunHandle:
        with self.tx() as c:
            c.execute(
                """INSERT INTO runs (run_id, design, lane, arm, arm_delay_s,
                       arm_delay_mode, arm_delay_virtual, model, seed, started_at,
                       budget_wall_s, budget_iters, status, liberty_path,
                       provenance_json, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'running', ?,?,?)""",
                (run_id, design, lane, arm, arm_delay_s, arm_delay_mode,
                 int(arm_delay_virtual), model, seed, _utcnow(), budget_wall_s,
                 budget_iters, liberty_path, _jdump(provenance), note),
            )
        if self.verbose:
            tag = " [VIRTUAL DELAY -- not a latency measurement]" if arm_delay_virtual else ""
            print(f"  [db] run {run_id}: {design} lane={lane} arm={arm} "
                  f"model={model} seed={seed}{tag}", flush=True)
        return RunHandle(run_id=run_id, store=self, t0=time.monotonic())

    def finish_run(self, run_id: str, status: str = "done") -> None:
        with self.tx() as c:
            c.execute("UPDATE runs SET ended_at=?, status=? WHERE run_id=?",
                      (_utcnow(), status, run_id))

    def record_provenance(self, run_id: str, prov: Mapping[str, Any]) -> None:
        tools = prov.get("tools", {}) or {}
        repos = prov.get("repos", {}) or {}

        def sha(name: str) -> str | None:
            r = repos.get(name)
            return r.get("sha") if isinstance(r, Mapping) else None

        host = prov.get("host", {}) or {}
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO provenance (run_id, host, livehd_git_sha,
                       chia_revision, lhdsuite_git_sha, yosys_version, yosys_git_sha,
                       verilator_version, opensta_version, pdk_version, liberty_sha256,
                       warm_run_index, captured_at, full_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, host.get("hostname"), sha("livehd"), sha("chia"),
                 sha("lhdsuite"), tools.get("yosys"), tools.get("yosys_git_sha"),
                 tools.get("verilator"), tools.get("sta"),
                 (prov.get("notes") or {}).get("pdk_version"),
                 (prov.get("notes") or {}).get("liberty_sha256"),
                 prov.get("warm_run_index"), _utcnow(), _jdump(prov)),
            )

    # --- variants -----------------------------------------------------------

    def add_variant(
        self,
        run: RunHandle,
        *,
        iteration_index: int,
        parent_id: int | None = None,
        diff: str | None = None,
        edit_note: str | None = None,
        verilog_sha256: str | None = None,
        wall_offset_s: float | None = None,
        **fields: Any,
    ) -> int:
        cols: dict[str, Any] = {
            "run_id": run.run_id,
            "parent_id": parent_id,
            "iteration_index": iteration_index,
            "created_at": _utcnow(),
            "wall_offset_s": run.elapsed() if wall_offset_s is None else wall_offset_s,
            "diff": diff,
            "edit_note": edit_note,
            "verilog_sha256": verilog_sha256,
        }
        for k, v in fields.items():
            cols[k] = _jdump(v) if k.endswith("_json") and not isinstance(v, (str, type(None))) else v
        names = ", ".join(cols)
        holes = ", ".join("?" * len(cols))
        with self.tx() as c:
            cur = c.execute(f"INSERT INTO variants ({names}) VALUES ({holes})",
                            tuple(cols.values()))
            vid = cur.lastrowid
        if self.verbose:
            print(f"  [db] variant {vid} (iter {iteration_index}, parent {parent_id}) "
                  f"@ t+{cols['wall_offset_s']:.1f}s", flush=True)
        return int(vid)

    def update_variant(self, variant_id: int, **fields: Any) -> None:
        if not fields:
            return
        vals = {
            k: (_jdump(v) if k.endswith("_json") and not isinstance(v, (str, type(None))) else v)
            for k, v in fields.items()
        }
        sets = ", ".join(f"{k}=?" for k in vals)
        with self.tx() as c:
            c.execute(f"UPDATE variants SET {sets} WHERE id=?",
                      (*vals.values(), variant_id))

    def add_iteration(self, run: RunHandle, *, iteration_index: int,
                      variant_id: int | None = None, **fields: Any) -> int:
        cols: dict[str, Any] = {
            "run_id": run.run_id,
            "variant_id": variant_id,
            "iteration_index": iteration_index,
            "wall_offset_s": fields.pop("wall_offset_s", run.elapsed()),
        }
        cols.update(fields)
        names = ", ".join(cols)
        holes = ", ".join("?" * len(cols))
        with self.tx() as c:
            cur = c.execute(f"INSERT INTO iterations ({names}) VALUES ({holes})",
                            tuple(cols.values()))
        return int(cur.lastrowid)

    def add_tool_run(self, run_id: str | None, tool_run: Any,
                     variant_id: int | None = None) -> int:
        """Persist a :class:`livelane.harness.run.ToolRun`."""
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO tool_runs (run_id, variant_id, label, tool, argv_json,
                       cwd, returncode, timed_out, wall_s, cpu_user_s, cpu_sys_s,
                       peak_rss_kb, stdout_path, stderr_path, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, variant_id, tool_run.label,
                 Path(tool_run.argv[0]).name if tool_run.argv else None,
                 json.dumps(tool_run.argv), tool_run.cwd, tool_run.returncode,
                 int(tool_run.timed_out), tool_run.wall_s, tool_run.cpu_user_s,
                 tool_run.cpu_sys_s, tool_run.peak_rss_kb, tool_run.stdout_path,
                 tool_run.stderr_path, _utcnow()),
            )
        return int(cur.lastrowid)

    # --- read-only queries --------------------------------------------------

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, tuple(params)))

    def history(self, run_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """The agent-visible view: what was tried and what it scored.

        Deliberately narrow. It exposes no column the agent could use to infer
        which lane it is in, no timings, no tool names, no delay.
        """
        rows = self._conn.execute(
            """SELECT iteration_index, parent_id, edit_note, functional_pass,
                      lec_verdict, accepted, qor_cells, qor_area_um2,
                      qor_max_delay_ns, qor_slack_ns
               FROM variants WHERE run_id=?
               ORDER BY iteration_index DESC LIMIT ?""",
            (run_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def best_so_far(self, run_id: str, metric: str = "qor_max_delay_ns",
                    lower_is_better: bool = True) -> list[tuple[float, float]]:
        """(wall_offset_s, best-so-far metric) over accepted variants, Figure 1."""
        if metric not in {"qor_max_delay_ns", "qor_area_um2", "qor_cells",
                          "qor_slack_ns", "judge_max_delay_ns", "judge_area_um2"}:
            raise ValueError(f"refusing to interpolate unknown metric {metric!r}")
        rows = self._conn.execute(
            f"""SELECT wall_offset_s, {metric} FROM variants
                WHERE run_id=? AND accepted=1 AND {metric} IS NOT NULL
                ORDER BY wall_offset_s""",
            (run_id,),
        ).fetchall()
        out: list[tuple[float, float]] = []
        best: float | None = None
        for t, v in rows:
            if best is None or (v < best if lower_is_better else v > best):
                best = v
            out.append((t, best))
        return out

    def run_summary(self, run_id: str) -> dict[str, Any]:
        r = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if r is None:
            raise KeyError(run_id)
        agg = self._conn.execute(
            """SELECT COUNT(*) AS variants,
                      SUM(accepted) AS accepted,
                      SUM(CASE WHEN lec_verdict='refuted' THEN 1 ELSE 0 END) AS lec_refuted,
                      SUM(CASE WHEN functional_pass=0 THEN 1 ELSE 0 END) AS func_failed,
                      MAX(wall_offset_s) AS last_t
               FROM variants WHERE run_id=?""",
            (run_id,),
        ).fetchone()
        cost = self._conn.execute(
            """SELECT COALESCE(SUM(cost_usd),0) AS usd,
                      COALESCE(SUM(evaluator_cpu_s),0) AS cpu_s,
                      COALESCE(SUM(t_delay_ms),0)/1000.0 AS delay_s,
                      COUNT(*) AS iterations
               FROM iterations WHERE run_id=?""",
            (run_id,),
        ).fetchone()
        return {**dict(r), **dict(agg), **dict(cost)}


if __name__ == "__main__":
    import tempfile

    print("=== VariantStore self-test ===")
    with tempfile.TemporaryDirectory() as td:
        s = VariantStore(Path(td) / "livelane.db")
        run = s.start_run("t-1", design="picorv32", lane="S", arm="I(30)",
                          model="gemini-flash", seed=1, arm_delay_s=30.0,
                          budget_wall_s=7200)

        from livelane.harness.provenance import capture
        prov = capture("t-1", repos={"livehd": Path.cwd() / "thirdparty" / "livehd"})
        s.record_provenance("t-1", json.loads(prov.to_json()))
        got = s.query("SELECT livehd_git_sha, yosys_git_sha FROM provenance WHERE run_id='t-1'")[0]
        assert got["livehd_git_sha"] == "ae0995d8dfd34b84b5d0f8b4f87b0138e0442272", dict(got)
        print(f"    provenance: livehd={got['livehd_git_sha'][:8]} yosys={got['yosys_git_sha'][:8]}")

        # A tree: seed -> two children, one accepted and improving, one refuted by LEC.
        root = s.add_variant(run, iteration_index=0, wall_offset_s=0.0,
                             accepted=1, functional_pass=1, lec_verdict="skipped",
                             qor_max_delay_ns=10.0, qor_area_um2=75664.0, qor_cells=6691)
        good = s.add_variant(run, iteration_index=1, parent_id=root, wall_offset_s=40.0,
                             accepted=1, functional_pass=1, lec_verdict="proven",
                             qor_max_delay_ns=9.1, qor_area_um2=76000.0, qor_cells=6720,
                             edit_note="retime the shifter")
        bad = s.add_variant(run, iteration_index=2, parent_id=root, wall_offset_s=80.0,
                            accepted=0, functional_pass=1, lec_verdict="refuted",
                            qor_max_delay_ns=8.0, edit_note="dropped a reset term")

        # A non-improving accepted variant must NOT move best-so-far.
        s.add_variant(run, iteration_index=3, parent_id=good, wall_offset_s=120.0,
                      accepted=1, functional_pass=1, lec_verdict="proven",
                      qor_max_delay_ns=9.6)

        bsf = s.best_so_far("t-1")
        assert bsf == [(0.0, 10.0), (40.0, 9.1), (120.0, 9.1)], bsf
        assert all(bsf[i][1] >= bsf[i + 1][1] for i in range(len(bsf) - 1)), "must be monotone"
        print(f"    best_so_far excludes the LEC-refuted variant: {bsf}")

        s.add_iteration(run, iteration_index=1, variant_id=good, wall_offset_s=40.0,
                        t_llm_ms=8000, t_tool_ms=2810, t_delay_ms=30000,
                        injected_delay_s=30.0, delay_mode="additive",
                        tokens_in=1200, tokens_out=300, tokens_cache_read=9000,
                        tokens_cache_write=500, cost_usd=0.004,
                        cost_source="reported", evaluator_cpu_s=2.6)

        from livelane.harness.run import run_tool
        tr = run_tool(["/bin/sh", "-c", "true"], label="yosys-synth", verbose=False)
        s.add_tool_run("t-1", tr, variant_id=good)

        # The agent view must not leak the lane.
        h = s.history("t-1", limit=5)
        leaky = {"wall_offset_s", "injected_delay_s", "qor_source", "lec_backend"}
        assert not (set(h[0]) & leaky), f"history leaks lane-identifying columns: {set(h[0]) & leaky}"
        print(f"    history() exposes only {sorted(h[0])}")

        summ = s.run_summary("t-1")
        assert summ["variants"] == 4 and summ["accepted"] == 3 and summ["lec_refuted"] == 1
        assert abs(summ["delay_s"] - 30.0) < 1e-6 and abs(summ["usd"] - 0.004) < 1e-9
        print(f"    summary: {summ['variants']} variants, {summ['accepted']} accepted, "
              f"{summ['lec_refuted']} LEC-refuted, ${summ['usd']:.4f}, {summ['delay_s']:.0f}s delay")

        try:
            s.best_so_far("t-1", metric="area; DROP TABLE variants")
            raise AssertionError("unvalidated metric name accepted")
        except ValueError as e:
            print(f"    metric allow-list holds: {e}")

        s.finish_run("t-1")
        s.close()
    print("=== all VariantStore self-tests passed ===")

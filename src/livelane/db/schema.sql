-- LiveLane variant tree and measurement ledger.
--
-- Design notes that matter for the science:
--  * QoR numbers are stored BOTH as the raw tool JSON (auditable) and as typed,
--    indexed columns (queryable).  Analysis must never re-parse JSON to answer
--    "best-so-far vs wall-clock", or the plots stop being reproducible cheaply.
--  * wall_offset_s is seconds since the run started, recorded on every row.  It
--    is the x-axis of Figure 1 and is therefore not derivable-after-the-fact.
--  * Every arm's injected delay is stored per-iteration, not just per-run, so an
--    interrupted or misconfigured delay can be found and excluded rather than
--    silently averaging into the headline.
--  * The equivalence verdict has room for TWO backends, because a disagreement
--    between eqy and circt-lec is itself a reportable result.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per (design, arm, model, seed) experimental cell execution.
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    design            TEXT NOT NULL,
    lane              TEXT NOT NULL CHECK (lane IN ('S', 'L')),
    arm               TEXT NOT NULL,              -- 'I(0)', 'I(30)', ..., 'L'
    arm_delay_s       REAL NOT NULL DEFAULT 0.0,
    arm_delay_mode    TEXT NOT NULL DEFAULT 'additive',
    arm_delay_virtual INTEGER NOT NULL DEFAULT 0, -- 1 => replay, NOT a latency measurement
    model             TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    budget_wall_s     REAL,
    budget_iters      INTEGER,
    status            TEXT NOT NULL DEFAULT 'running',
    liberty_path      TEXT,
    provenance_json   TEXT,
    note              TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_cell ON runs (design, arm, model, seed);

-- One row per candidate design variant produced by the agent.
CREATE TABLE IF NOT EXISTS variants (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    parent_id           INTEGER REFERENCES variants (id),
    iteration_index     INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    wall_offset_s       REAL NOT NULL,            -- x-axis of the money chart

    -- provenance of the edit itself
    diff                TEXT,
    edit_note           TEXT,
    verilog_sha256      TEXT,

    -- gates (harness-computed; the agent never writes these)
    functional_pass     INTEGER,                  -- Verilator oracle, shared by all arms
    lec_verdict         TEXT,                     -- proven | refuted | error | skipped
    lec_backend         TEXT,
    lec_wall_s          REAL,
    lec_crosscheck      TEXT,                     -- second opinion; disagreement is a result
    lec_crosscheck_backend TEXT,
    accepted            INTEGER NOT NULL DEFAULT 0,

    -- in-lane QoR (lane S: yosys+abc+opensta; lane L: lhd synth)
    qor_source          TEXT,
    qor_cells           INTEGER,
    qor_area_um2        REAL,
    qor_max_delay_ns    REAL,
    qor_slack_ns        REAL,
    qor_json            TEXT,
    timing_json         TEXT,

    -- neutral judge: ONE common flow re-scores every final, in every arm
    judge_cells         INTEGER,
    judge_area_um2      REAL,
    judge_max_delay_ns  REAL,
    judge_slack_ns      REAL,
    judge_json          TEXT,

    status              TEXT NOT NULL DEFAULT 'ok',
    note                TEXT
);
CREATE INDEX IF NOT EXISTS idx_variants_run     ON variants (run_id, wall_offset_s);
CREATE INDEX IF NOT EXISTS idx_variants_parent  ON variants (parent_id);
CREATE INDEX IF NOT EXISTS idx_variants_accept  ON variants (run_id, accepted, wall_offset_s);

-- One row per agent turn: the time split and the four-class token accounting.
CREATE TABLE IF NOT EXISTS iterations (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    variant_id           INTEGER REFERENCES variants (id),
    iteration_index      INTEGER NOT NULL,
    wall_offset_s        REAL NOT NULL,

    -- the time split (Figure 2)
    t_llm_ms             REAL NOT NULL DEFAULT 0,
    t_tool_ms            REAL NOT NULL DEFAULT 0,
    t_orch_ms            REAL NOT NULL DEFAULT 0,
    t_gate_ms            REAL NOT NULL DEFAULT 0,
    t_delay_ms           REAL NOT NULL DEFAULT 0,

    -- the treatment, recorded per iteration so misfires are findable
    injected_delay_s     REAL NOT NULL DEFAULT 0,
    delay_mode           TEXT,
    delay_interrupted    INTEGER NOT NULL DEFAULT 0,

    -- four-class tokens: two-class pricing is wrong by up to 10x
    tokens_in            INTEGER NOT NULL DEFAULT 0,
    tokens_out           INTEGER NOT NULL DEFAULT 0,
    tokens_cache_read    INTEGER NOT NULL DEFAULT 0,
    tokens_cache_write   INTEGER NOT NULL DEFAULT 0,
    cost_usd             REAL,
    cost_source          TEXT,                    -- reported | listprice

    -- evaluator compute, priced separately from tokens
    evaluator_cpu_s      REAL NOT NULL DEFAULT 0,
    evaluator_peak_rss_kb INTEGER NOT NULL DEFAULT 0,

    note                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations (run_id, iteration_index);

-- Every instrumented external tool invocation. Feeds the CPU-hours column and
-- lets any published wall-clock be traced back to a specific argv.
CREATE TABLE IF NOT EXISTS tool_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT REFERENCES runs (run_id) ON DELETE CASCADE,
    variant_id     INTEGER REFERENCES variants (id),
    label          TEXT,
    tool           TEXT,
    argv_json      TEXT NOT NULL,
    cwd            TEXT,
    returncode     INTEGER,
    timed_out      INTEGER NOT NULL DEFAULT 0,
    wall_s         REAL NOT NULL,
    cpu_user_s     REAL NOT NULL DEFAULT 0,
    cpu_sys_s      REAL NOT NULL DEFAULT 0,
    peak_rss_kb    INTEGER NOT NULL DEFAULT 0,
    stdout_path    TEXT,
    stderr_path    TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_runs_run ON tool_runs (run_id, label);

-- Full provenance blob per run, plus the fields we filter on.
CREATE TABLE IF NOT EXISTS provenance (
    run_id            TEXT PRIMARY KEY REFERENCES runs (run_id) ON DELETE CASCADE,
    host              TEXT,
    livehd_git_sha    TEXT,
    chia_revision     TEXT,
    lhdsuite_git_sha  TEXT,
    yosys_version     TEXT,
    yosys_git_sha     TEXT,
    verilator_version TEXT,
    opensta_version   TEXT,
    pdk_version       TEXT,
    liberty_sha256    TEXT,
    -- LiveHD re-salts caches on rebuild: warm run #1 misses, #2 is the real number.
    warm_run_index    INTEGER,
    captured_at       TEXT NOT NULL,
    full_json         TEXT NOT NULL
);

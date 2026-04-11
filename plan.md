# oxRL Self-Healing Guardian — Implementation Plan

## Goal

`oxrl train` spawns a guardian process that watches the experiment in real-time,
detects failures as they happen (not post-mortem), fixes what it can using existing
bugfixer logic, web-searches for fixes when bugfixer doesn't know, and files a
GitHub issue for systematic problems. No babysitting.

## Architecture

```
oxrl train --model Qwen/Qwen2.5-7B --task math
  │
  ├── Training Process (main_rl.py)
  │     writes → train.log (continuously)
  │
  └── Guardian Process (guardian.py)
        │
        │  MONITOR (every tick):
        │    tail train.log for new lines → pattern match known errors
        │    PID alive?
        │    disk full?
        │
        │  ON FAILURE DETECTED:
        │    1. classify → bugfixer.classify_failure()
        │    2. known fix exists?
        │    │   YES → bugfixer.get_fix() → apply_config_fix() → restart
        │    │   NO  → _research_fix(error_msg) → web search → try fix → restart
        │    3. still broken after 3 attempts? → bug_reporter.submit_github_issue()
        │
        └── writes → guardian.log
```

One file. One class. Calls existing modules. Searches the web when stuck.

## Law of Demeter

```
trainer.py  →  guardian.py  →  bugfixer      (classify, fix, patch config)
                            →  bug_reporter  (file issue)
                            →  web search    (research unknown errors)
                            →  train.log     (tail / read)
                            →  config.yaml   (read / write via bugfixer)
```

- guardian receives paths and PIDs from trainer, not objects
- guardian calls bugfixer's public API, never reaches into its internals
- trainer never talks to bugfixer directly — guardian is the boundary
- web search results flow into bugfixer's config patching, not a separate path

## New Files

```
oxrl/guardian.py       # The guardian. One file.
tests/test_guardian.py # Tests.
```

## Modified Files

```
oxrl/trainer.py        # Spawn guardian after launching training subprocess
oxrl/cli.py            # Add --no-guardian flag
```

---

## Task 1: `oxrl/guardian.py`

### Core loop

```python
class Guardian:
    def __init__(self, pid, config_path, log_path, experiment_id, model_info):
        self.pid = pid
        self.config_path = config_path
        self.log_path = log_path
        self.experiment_id = experiment_id
        self.model_info = model_info
        self.max_heal_attempts = 3
        self.heal_count = 0
        self._heal_history = []       # ["oom → halved batch size", ...]
        self._stop = False
        self._log_offset = 0          # byte offset — where we last read train.log
        self._last_log_activity = time.time()

    def run(self):
        """Main loop. Tail log + check health every tick.

        Wrapped in try/except so guardian never dies from its own bugs.
        If the guardian's own code throws, it logs the error and keeps going.
        Only stops on: clean training exit, exhausted heal attempts, or stop().
        """
        self._log(f"Guardian started for PID {self.pid}")
        while not self._stop:
            try:
                self._tick()
            except _TrainingDone:
                break
            except Exception as e:
                # Guardian must not die. Log and continue.
                self._log(f"Guardian internal error (continuing): {e}")
            time.sleep(5)

        self._print_summary()

    def _tick(self):
        """One iteration of the main loop."""
        # 1. Tail new log lines, check for errors in real-time
        problem = self._tail_log()

        # 2. If log shows a problem, act now (don't wait for process to die)
        if problem:
            self._handle_failure(problem)
            return

        # 3. Process already dead?
        if not self._pid_alive():
            failure = bugfixer.classify_failure(self.log_path)
            if failure == "unknown" and self._exit_code() == 0:
                raise _TrainingDone()  # clean exit
            self._handle_failure(failure)
            return

        # 4. Stall detection: PID alive but no log output for 10 min
        if self._stalled():
            self._kill_and_cleanup()
            self._handle_failure("timeout")

        # 5. Disk space
        if self._disk_low():
            self._run_gc()

    def stop(self):
        self._stop = True

    def _print_summary(self):
        """Print what happened so the user sees it when training finishes."""
        if self.heal_count == 0:
            return  # nothing interesting happened
        print(f"\n--- Guardian Summary (experiment {self.experiment_id}) ---")
        print(f"  Interventions: {self.heal_count}")
        for entry in self._heal_history:
            print(f"  - {entry}")
        print("---\n")


class _TrainingDone(Exception):
    """Sentinel: training exited cleanly."""
    pass
```

### Real-time log monitoring

```python
# Patterns that mean "problem happening right now"
LOG_PATTERNS = {
    "oom":       re.compile(r"CUDA out of memory|OutOfMemoryError"),
    "nan_loss":  re.compile(r"loss.*nan|NaN"),
    "vllm_load": re.compile(r"Cannot load model|Error loading model"),
    "ray_crash": re.compile(r"RayActorError|ActorDiedError"),
    "nccl":      re.compile(r"NCCL.*error|NCCL.*timeout", re.I),
}

def _tail_log(self) -> Optional[str]:
    """Read new bytes from train.log since last check. Return failure type or None."""
    try:
        size = os.path.getsize(self.log_path)
    except OSError:
        return None
    if size <= self._log_offset:
        return None

    with open(self.log_path, "r", errors="replace") as f:
        f.seek(self._log_offset)
        new_text = f.read()
    self._log_offset = size
    self._last_log_activity = time.time()

    for failure_type, pattern in self.LOG_PATTERNS.items():
        if pattern.search(new_text):
            return failure_type
    return None
```

### Failure handling: known fix → bugfixer, unknown → web search

```python
def _handle_failure(self, failure_type: str):
    """Try to heal. Known fix first, then web search, then give up and report."""
    if self.heal_count >= self.max_heal_attempts:
        self._report(failure_type)
        self._stop = True
        return

    # Step 1: Try bugfixer (known fixes)
    fix = bugfixer.get_fix(failure_type, self.model_info)

    if fix["action"] != "skip":
        # Known fix — apply it
        bugfixer.apply_config_fix(fix, self.model_info)
        self._heal_history.append(f"{failure_type} → {fix['reason']}")
        self._log(f"Applied known fix for {failure_type}: {fix['reason']}")
    else:
        # Step 2: Unknown/unfixable — research a fix via web search
        error_context = self._get_error_context()
        researched_fix = self._research_fix(failure_type, error_context)

        if researched_fix:
            self._apply_researched_fix(researched_fix)
            self._heal_history.append(f"{failure_type} → web-searched fix: {list(researched_fix.keys())}")
            self._log(f"Applied researched fix for {failure_type}")
        else:
            # Nothing worked — report and stop
            self._report(failure_type)
            self._stop = True
            return

    # Clean up orphan processes (Ray workers, vLLM engines) before restart
    if self._pid_alive():
        self._kill_and_cleanup()
    self.pid = self._restart_training()
    self.heal_count += 1
```

### Web search for unknown fixes

```python
def _get_error_context(self) -> str:
    """Last 30 lines of train.log — the error traceback."""
    with open(self.log_path, "r", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-30:])

def _research_fix(self, failure_type: str, error_context: str) -> Optional[dict]:
    """Web search the error, return config changes or None.

    Searches for the error message, parses results for actionable config
    changes (batch size, LR, DeepSpeed settings, etc.)

    Returns dict of dotted-key config changes like bugfixer uses:
        {"train.lr": 1e-7, "deepspeed.zero_optimization.stage": 2}
    or None if nothing actionable found.
    """
    # Build search query from the most specific error line
    error_lines = error_context.strip().split("\n")
    # Use the last Exception/Error line as the query
    query_line = next(
        (l for l in reversed(error_lines) if "Error" in l or "Exception" in l),
        error_lines[-1] if error_lines else failure_type,
    )
    query = f"oxRL DeepSpeed vLLM fix: {query_line[:200]}"

    try:
        results = self._web_search(query)
    except Exception as e:
        self._log(f"Web search failed: {e}")
        return None

    return self._parse_search_results(results, failure_type)

def _web_search(self, query: str) -> list[str]:
    """Search the web. Returns list of relevant text snippets.

    Uses requests to hit a search API. Falls back to None on failure.
    Implementations can use:
      - Google Custom Search API (if GOOGLE_API_KEY set)
      - DuckDuckGo instant answers (no key needed)
      - GitHub issue search on common repos (DeepSpeed, vLLM, transformers)
    """
    import requests

    # DuckDuckGo instant answer (no API key needed)
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1},
        timeout=10,
    )
    data = resp.json()
    snippets = []
    if data.get("AbstractText"):
        snippets.append(data["AbstractText"])
    for r in data.get("RelatedTopics", [])[:5]:
        if isinstance(r, dict) and r.get("Text"):
            snippets.append(r["Text"])

    # Also search GitHub issues for DeepSpeed/vLLM/transformers
    for repo in ["microsoft/DeepSpeed", "vllm-project/vllm"]:
        try:
            gh_resp = requests.get(
                f"https://api.github.com/search/issues",
                params={"q": f"{query_line[:100]} repo:{repo}", "per_page": 3},
                timeout=10,
            )
            for item in gh_resp.json().get("items", []):
                snippets.append(f"{item['title']}: {item['body'][:300]}")
        except Exception:
            continue

    return snippets

def _parse_search_results(self, snippets: list[str], failure_type: str) -> Optional[dict]:
    """Extract actionable config changes from search result text.

    Looks for common fix patterns in the snippets:
      - "reduce batch size" / "batch_size" → halve batch sizes
      - "reduce learning rate" / "lower lr" → reduce LR
      - "ZeRO stage 2" / "disable stage 3" → change ZeRO stage
      - "offload" / "cpu offload" → enable CPU offload
      - "gradient checkpointing" → enable gradient checkpointing
      - "trust_remote_code" → set trust_remote_code=True
      - specific numeric suggestions (lr=X, bs=Y)
    """
    combined = " ".join(snippets).lower()
    changes = {}

    # Load current config to compute relative changes
    import yaml
    with open(self.config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if "batch size" in combined or "batch_size" in combined:
        bs = cfg.get("train", {}).get("train_batch_size_per_gpu", 2)
        changes["train.train_batch_size_per_gpu"] = max(1, bs // 2)

    if "learning rate" in combined or "reduce lr" in combined:
        lr = cfg.get("train", {}).get("lr", 1e-6)
        changes["train.lr"] = lr / 10.0

    if "zero stage 2" in combined or "disable stage 3" in combined:
        changes["deepspeed.zero_optimization.stage"] = 2

    if "cpu offload" in combined or "offload optimizer" in combined:
        changes["deepspeed.zero_optimization.offload_optimizer"] = {
            "device": "cpu", "pin_memory": True
        }

    if "gradient checkpointing" in combined or "activation checkpointing" in combined:
        changes["model.gradient_checkpointing"] = True

    if "trust_remote_code" in combined:
        changes["model.trust_remote_code"] = True

    return changes if changes else None

def _apply_researched_fix(self, changes: dict):
    """Apply web-researched config changes using bugfixer's patching."""
    fix = {
        "action": "adjust_config",
        "changes": changes,
        "reason": "Fix researched via web search",
    }
    bugfixer.apply_config_fix(fix, self.model_info)
```

### Checkpoint resume on restart

```python
def _find_latest_checkpoint(self) -> Optional[str]:
    """Find the latest checkpoint dir for this experiment.

    Scans the experiment's checkpoint directory for dirs like
    'step_100/', 'step_200/' and returns the highest-numbered one.
    Returns None if no checkpoints exist (training will start fresh).
    """
    ckpt_root = os.path.join(os.path.dirname(self.config_path), "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    steps = []
    for name in os.listdir(ckpt_root):
        m = re.match(r"step_(\d+)", name)
        if m:
            steps.append((int(m.group(1)), os.path.join(ckpt_root, name)))
    return max(steps, key=lambda x: x[0])[1] if steps else None

def _restart_training(self) -> int:
    """Launch training subprocess with current config. Resume from checkpoint if available."""
    cmd = [
        sys.executable, "-m", "oxrl.main_rl",
        "--config-file", self.config_path,
        "--experiment_id", self.experiment_id,
    ]
    ckpt = self._find_latest_checkpoint()
    if ckpt:
        cmd.extend(["--resume-from", ckpt])
        self._log(f"Resuming from checkpoint: {ckpt}")

    proc = subprocess.Popen(cmd, start_new_session=True)
    return proc.pid
```

### Systematic failure detection (cross-experiment)

All guardians append to one shared file: `~/.oxrl/failure_ledger.jsonl`.
One line per failure. Before filing a GitHub issue, check if the same
`failure_type` has hit N+ different experiments recently — that's systematic.

```python
FAILURE_LEDGER = os.path.expanduser("~/.oxrl/failure_ledger.jsonl")

def _record_failure(self, failure_type: str):
    """Append this failure to the shared ledger."""
    os.makedirs(os.path.dirname(self.FAILURE_LEDGER), exist_ok=True)
    entry = json.dumps({
        "ts": datetime.now().isoformat(),
        "experiment_id": self.experiment_id,
        "model": self.model_info.get("model_name"),
        "failure_type": failure_type,
        "config_path": self.config_path,
    })
    with open(self.FAILURE_LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(entry + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

def _is_systematic(self, failure_type: str, threshold: int = 3) -> bool:
    """Same failure_type across threshold+ different experiments in the last 24h."""
    if not os.path.isfile(self.FAILURE_LEDGER):
        return False
    cutoff = datetime.now() - timedelta(hours=24)
    experiment_ids = set()
    with open(self.FAILURE_LEDGER) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("failure_type") != failure_type:
                continue
            if datetime.fromisoformat(entry["ts"]) < cutoff:
                continue
            experiment_ids.add(entry["experiment_id"])
    return len(experiment_ids) >= threshold

ISSUE_COOLDOWN_SEC = 3600  # Don't file more than 1 issue per failure_type per hour

def _report(self, failure_type: str):
    """Record failure, detect systematic issues, file GitHub issue (rate-limited)."""
    self._record_failure(failure_type)

    # Rate limit: check if we already filed an issue for this failure_type recently
    if self._recently_reported(failure_type):
        self._log(f"Skipping issue for {failure_type} — reported within last hour")
        return

    model_id = self.model_info.get("model_name", "unknown")

    if self._is_systematic(failure_type):
        # Systematic: same failure across 3+ experiments → one issue
        report_md = bug_reporter.summarize_failure(
            model_id=f"SYSTEMATIC ({failure_type})",
            error_msg=(
                f"'{failure_type}' hit 3+ experiments in the last 24h. "
                f"Latest: {model_id}, experiment {self.experiment_id}. "
                f"This is likely a framework-level bug, not experiment-specific."
            ),
            log_path=self.log_path,
        )
        bug_reporter.submit_github_issue(report_md, f"systematic-{failure_type}")
    else:
        # Individual failure
        report_md = bug_reporter.summarize_failure(
            model_id=model_id,
            error_msg=f"Guardian gave up after {self.heal_count} attempts. Type: {failure_type}",
            log_path=self.log_path,
        )
        if bug_reporter.is_framework_error(failure_type):
            bug_reporter.submit_github_issue(report_md, model_id)

    self._mark_reported(failure_type)
    self._log(f"Reported {failure_type} (systematic={self._is_systematic(failure_type)})")

def _recently_reported(self, failure_type: str) -> bool:
    """Check the ledger for a recent report of this failure_type."""
    if not os.path.isfile(self.FAILURE_LEDGER):
        return False
    cutoff = datetime.now() - timedelta(seconds=self.ISSUE_COOLDOWN_SEC)
    with open(self.FAILURE_LEDGER) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (entry.get("failure_type") == failure_type
                    and entry.get("reported")
                    and datetime.fromisoformat(entry["ts"]) > cutoff):
                return True
    return False

def _mark_reported(self, failure_type: str):
    """Append a 'reported' entry to the ledger so other guardians know."""
    os.makedirs(os.path.dirname(self.FAILURE_LEDGER), exist_ok=True)
    entry = json.dumps({
        "ts": datetime.now().isoformat(),
        "experiment_id": self.experiment_id,
        "model": self.model_info.get("model_name"),
        "failure_type": failure_type,
        "reported": True,
    })
    with open(self.FAILURE_LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(entry + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
```

### Other infrastructure methods

```python
def _pid_alive(self) -> bool:
    try:
        os.kill(self.pid, 0)
        return True
    except OSError:
        return False

def _exit_code(self) -> Optional[int]:
    try:
        _, status = os.waitpid(self.pid, os.WNOHANG)
        return os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
    except ChildProcessError:
        return None

def _stalled(self) -> bool:
    return time.time() - self._last_log_activity > 600

def _disk_low(self) -> bool:
    usage = shutil.disk_usage("/")
    return usage.free / usage.total < 0.05

def _kill_and_cleanup(self):
    """Kill training process tree AND orphan Ray/vLLM workers."""
    # 1. Kill the process group (training + direct children)
    try:
        os.killpg(os.getpgid(self.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass

    # 2. Kill orphan Ray workers that may hold GPU memory
    #    Ray workers have "ray::Worker" or "ray::RolloutWorker" in cmdline
    #    This prevents the next restart from OOM-ing on already-occupied GPUs
    try:
        import subprocess as sp
        result = sp.run(
            ["pgrep", "-f", f"ray.*{self.experiment_id}"],
            capture_output=True, text=True,
        )
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str.strip():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                except OSError:
                    pass
    except Exception:
        pass  # pgrep not available or no orphans — fine

    # 3. Shutdown Ray if we started it
    try:
        import subprocess as sp
        sp.run(["ray", "stop", "--force"], capture_output=True, timeout=10)
    except Exception:
        pass

    time.sleep(2)  # let GPU memory free up

def _run_gc(self):
    for cache_dir in [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/tmp/vllm_cache",
        "/tmp/torch_extensions",
    ]:
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    self._log("Ran GC to free disk space")

def _log(self, msg: str):
    guardian_log = self.log_path.replace("train.log", "guardian.log")
    with open(guardian_log, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")
```

---

## Task 2: Modify `oxrl/trainer.py`

```python
# In Trainer.train(), after subprocess launch:
if guardian:
    from oxrl.guardian import Guardian
    g = Guardian(
        pid=process.pid,
        config_path=config_path,
        log_path=log_path,
        experiment_id=experiment_id,
        model_info={"config_path": config_path, "model_name": self.model},
    )
    import multiprocessing
    self._guardian_proc = multiprocessing.Process(target=g.run, daemon=True)
    self._guardian_proc.start()
```

Add `guardian=True` param to `train()`.

---

## Task 3: Modify `oxrl/cli.py`

```python
train_parser.add_argument("--no-guardian", action="store_true",
                          help="Disable the self-healing guardian agent")
```

Pass `guardian=not args.no_guardian` to `trainer.train()`.

---

## Task 4: `tests/test_guardian.py`

1. **test_tail_detects_oom** — Append "CUDA out of memory" to temp log. `_tail_log()` returns `"oom"`.
2. **test_tail_detects_nan** — Append "loss: nan". Returns `"nan_loss"`.
3. **test_tail_incremental** — Write lines in two batches. Second `_tail_log()` only sees new lines.
4. **test_known_fix_applied** — OOM detected. Verify `bugfixer.apply_config_fix` called, process restarted.
5. **test_unknown_triggers_web_search** — `bugfixer.get_fix` returns `"skip"`. Verify `_research_fix` called.
6. **test_parse_search_results_batch_size** — Snippets mention "reduce batch size". Returns batch size change.
7. **test_parse_search_results_nothing** — Snippets are irrelevant. Returns None.
8. **test_max_retries_then_report** — Fail 3 times. Verify `submit_github_issue` called.
9. **test_stall_kills** — `_last_log_activity` is 15 min ago. Verify process killed + heal attempted.
10. **test_clean_exit** — Process exits 0, no errors in log. Guardian exits without healing.
11. **test_restart_resumes_checkpoint** — Create fake `checkpoints/step_100/` and `step_200/`. Verify restart uses `--resume-from .../step_200`.
12. **test_systematic_detection** — Write 3 entries with same `failure_type` to ledger. `_is_systematic()` returns True.
13. **test_systematic_files_one_issue** — 3+ experiments fail with same type. Verify one `[SYSTEMATIC]` issue filed, not 3 individual ones.
14. **test_not_systematic_below_threshold** — 2 entries in ledger. `_is_systematic()` returns False.
15. **test_guardian_survives_own_error** — Monkey-patch `_tail_log` to raise. Verify `run()` continues and logs the internal error.
16. **test_print_summary** — Heal twice. Verify `_print_summary` outputs both interventions.
17. **test_rate_limit_issues** — Report same failure_type twice within 1 hour. Verify `submit_github_issue` called only once.
18. **test_orphan_cleanup** — Mock `pgrep` and `ray stop`. Verify `_kill_and_cleanup` calls both.

Mock `subprocess.Popen`, `bugfixer`, `bug_reporter`, `requests`. No GPUs needed.

---

## Execution Order

```
Task 1 (guardian.py)  →  Task 2 (trainer.py)  →  Task 3 (cli.py)  →  Task 4 (tests)
```

One subagent, sequential.

---

## What We're NOT Building

- No watcher classes — one loop tails the log and checks PID/disk
- No event schema — failure_type is a string, same as bugfixer
- No failure database — just an append-only JSONL ledger for systematic detection
- No LLM calls — web search + keyword matching for fix extraction
- No separate MetricWatcher — NaN/reward issues caught from log patterns
- No orchestrator changes — guardian is per-experiment, orthogonal to swarm

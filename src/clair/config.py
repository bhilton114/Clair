# config.py
# Clair Configuration (v2.46a – contract-safe, explicit flags + calibration rails + unified test knobs)

from __future__ import annotations

import os

# =========================
# Project Paths
# =========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INTAKE_PATH = os.path.join(ROOT_DIR, "intake", "text_learning.yaml")
SCENARIO_PATH = os.path.join(ROOT_DIR, "intake", "scenarios.yaml")

LONG_TERM_DB_PATH = os.path.join(ROOT_DIR, "memory", "long_term_memory.db")
LOG_FILE = os.path.join(ROOT_DIR, "logs", "clair.log")

# =========================
# Logging / Debug
# =========================
VERBOSE = True

# Simulator debug
DEBUG_SIMULATOR_FIRST_ACTION = True
SIMULATOR_LOCK_DEBUG = True        # survival horizon debugging
RISK_DEBUG = False                 # risk term logging
SIM_TRACE_DETAILS = False          # rollout trace noise toggle

# =========================
# System Timing
# =========================
HEARTBEAT_INTERVAL = 1.0
REFLECTION_INTERVAL = 1.0

# =========================
# Memory Configuration
# =========================
WORKING_MEMORY_MAX = 50
MEMORY_DECAY_RATE = 0.9

LTM_AUTO_SYNC_WEIGHT = 0.75
LTM_POV_SAFE_IDENTITY = True

# =========================
# Working Memory Retrieval
# =========================
WM_MIN_RELEVANCE = 1.5
WM_MIN_STRONG_OVERLAP = 1
WM_MIN_OVERLAP_SCORE = 1.0
WM_PLANNING_MIN_QUALITY = 0.42

# Optional scoring weights (explicit so getattr never guesses)
WM_OVERLAP_WEIGHT = 4.0
WM_CONF_WEIGHT = 1.0
WM_WEIGHT_WEIGHT = 1.0
WM_TAG_MATCH_WEIGHT = 1.0

# Survival hazard scoring
WM_HAZARD_BONUS = 2.5
WM_HAZARD_MISMATCH_PENALTY = 2.0

# =========================
# Simulator / Planning
# =========================
SIMULATOR_EXPLORATION_RATE = 0.25
SIMULATOR_HISTORY_WINDOW = 5
SIMULATOR_DEFAULT_NUM_ACTIONS = 3

SIMULATOR_NUM_ROLLOUTS = 5
SIMULATOR_MAX_ROLLOUTS = 12
SIMULATOR_UNCERTAINTY_PENALTY = 0.5
SIMULATOR_RISK_PENALTY = 0.0
SIMULATOR_STEP_DISCOUNT = 0.85

SIMULATOR_HORIZON = 2

# Followups
SIMULATOR_FOLLOWUP_CANDIDATES = 2
SIMULATOR_FOLLOWUP_TRIES = 20
SIMULATOR_FOLLOWUP_SIMILARITY_MAX = 0.78
SIMULATOR_FOLLOWUP_MIN_DELTA_LEN = 10

# Determinism / RNG
SIMULATOR_DETERMINISTIC = True
SIMULATOR_RNG_SALT = ""

# Candidate pool protection
SIMULATOR_THIN_POOL_MIN = 3

# Hazard pin/lock thresholds (simulator references these)
SIMULATOR_HAZARD_PIN_THRESHOLD = 0.65
SIMULATOR_HAZARD_LOCK_THRESHOLD = 0.65

# =========================
# Risk & Safety
# =========================
RISK_THRESHOLD = 0.7
HARD_RULE_RISK = 0.8

# =========================
# Reasoning Engine
# =========================
REASONING_MAX_CHAIN_STEPS = 3
REASONING_MAX_ACTIONS = 3
REASONING_SUMMARY_LIMIT = 5

# =========================
# Reflection / Metacognition
# =========================
REFLECTION_SCORE_WEIGHT = 0.1

DEFAULT_CONTEXT_WEIGHTS = {
    "system": 0.7,
    "safety": 1.0,
    "operations": 0.5,
    "user": 0.6,
    "lesson": 0.8,
}

# =========================
# Deferred Handling
# =========================
MAX_DEFER_RETRIES = 3

# =========================
# Action Cycle Gating
# =========================
ACTION_QUESTION_FRESH_SEC = 8.0

ACTION_URGENCY_THRESHOLD = 0.65
ACTION_THREAT_THRESHOLD = 0.65

ACTION_CYCLE_COOLDOWN_SEC = 2.0

REQUIRE_ACTION_WORTHY_TICK = True
BLOCK_ACTION_CYCLE_AFTER_FACT_RECALL = True
QUARANTINE_FEEDBACK = True

# =========================
# Calibration / Cerebellar Rails
# =========================
CAL_Q_LIMIT = 2
CAL_Q_COOLDOWN_SEC = 0.0

CAL_PROMOTION_STEP = 0.08
CAL_DEMOTION_STEP = 0.12
CAL_MAX_CONFIDENCE_NO_WEB = 0.85

CAL_SLEEP_TIME_BUDGET_SEC = 0.60
CAL_MAX_PAIR_CHECKS = 800
CAL_MAX_NUMERIC_ITEMS = 120

CAL_SEM_OVERLAP_MIN = 3
CAL_STALE_DAYS = 30
CAL_DECAY_MULT = 0.98

# If 1, cerebellar stores calibration episodes to LTM (can bloat DB).
CAL_PERSIST_EPISODES = 0

# =========================
# Calibration Harness Defaults (so test_calibration isn't its own universe)
# =========================
CAL_NO_WRITE_DEFAULT = 1           # block LTM writes during calibration tests
CAL_REPEAT_WINDOW = 32             # your Day 18 setting
CAL_REPEAT_SOFT_NORM = 1
CAL_MAX_ATTEMPTS_PER_STEP = 2      # your Day 18 setting
CAL_SIMUSER_CONFIRM_RATE = 0.80
CAL_IDLE_STEPS_DEFAULT = 20
CAL_SLEEP_STEPS_DEFAULT = 5

# =========================
# Feature Flags
# =========================
ENABLE_SCENARIO_MODE = True
ENABLE_USER_FEEDBACK = True

# =========================
# Safety Net: Basic Sanity Checks
# =========================
def _validate():
    assert SIMULATOR_HORIZON >= 1, "SIMULATOR_HORIZON must be >= 1"
    assert 0.0 <= SIMULATOR_EXPLORATION_RATE <= 1.0, "SIMULATOR_EXPLORATION_RATE out of range"
    assert 0.0 <= RISK_THRESHOLD <= 1.0, "RISK_THRESHOLD out of range"
    assert 0.0 <= HARD_RULE_RISK <= 1.0, "HARD_RULE_RISK out of range"
    assert WORKING_MEMORY_MAX > 0, "WORKING_MEMORY_MAX must be positive"

    # Simulator
    assert SIMULATOR_NUM_ROLLOUTS >= 1, "SIMULATOR_NUM_ROLLOUTS must be >= 1"
    assert SIMULATOR_MAX_ROLLOUTS >= SIMULATOR_NUM_ROLLOUTS, "SIMULATOR_MAX_ROLLOUTS must be >= SIMULATOR_NUM_ROLLOUTS"
    assert 0.0 <= SIMULATOR_UNCERTAINTY_PENALTY <= 5.0, "SIMULATOR_UNCERTAINTY_PENALTY out of sane range"
    assert 0.0 <= SIMULATOR_STEP_DISCOUNT <= 1.0, "SIMULATOR_STEP_DISCOUNT out of range"
    assert 0.0 <= SIMULATOR_HAZARD_PIN_THRESHOLD <= 1.0, "SIMULATOR_HAZARD_PIN_THRESHOLD out of range"
    assert 0.0 <= SIMULATOR_HAZARD_LOCK_THRESHOLD <= 1.0, "SIMULATOR_HAZARD_LOCK_THRESHOLD out of range"
    assert SIMULATOR_THIN_POOL_MIN >= 0, "SIMULATOR_THIN_POOL_MIN must be >= 0"

    # Calibration sanity
    assert CAL_Q_LIMIT >= 1, "CAL_Q_LIMIT must be >= 1"
    assert CAL_Q_COOLDOWN_SEC >= 0.0, "CAL_Q_COOLDOWN_SEC must be >= 0"
    assert 0.0 <= CAL_PROMOTION_STEP <= 1.0, "CAL_PROMOTION_STEP out of range"
    assert 0.0 <= CAL_DEMOTION_STEP <= 1.0, "CAL_DEMOTION_STEP out of range"
    assert 0.0 <= CAL_MAX_CONFIDENCE_NO_WEB <= 1.0, "CAL_MAX_CONFIDENCE_NO_WEB out of range"
    assert CAL_SLEEP_TIME_BUDGET_SEC >= 0.0, "CAL_SLEEP_TIME_BUDGET_SEC must be >= 0"
    assert CAL_MAX_PAIR_CHECKS >= 0, "CAL_MAX_PAIR_CHECKS must be >= 0"
    assert CAL_MAX_NUMERIC_ITEMS >= 0, "CAL_MAX_NUMERIC_ITEMS must be >= 0"
    assert CAL_SEM_OVERLAP_MIN >= 1, "CAL_SEM_OVERLAP_MIN must be >= 1"
    assert CAL_STALE_DAYS >= 0, "CAL_STALE_DAYS must be >= 0"
    assert 0.0 <= CAL_DECAY_MULT <= 1.0, "CAL_DECAY_MULT out of range"
    assert CAL_PERSIST_EPISODES in (0, 1, False, True), "CAL_PERSIST_EPISODES must be 0/1"

    # Harness sanity
    assert CAL_REPEAT_WINDOW >= 0, "CAL_REPEAT_WINDOW must be >= 0"
    assert CAL_MAX_ATTEMPTS_PER_STEP >= 1, "CAL_MAX_ATTEMPTS_PER_STEP must be >= 1"
    assert 0.0 <= CAL_SIMUSER_CONFIRM_RATE <= 1.0, "CAL_SIMUSER_CONFIRM_RATE out of range"
    assert CAL_IDLE_STEPS_DEFAULT >= 1, "CAL_IDLE_STEPS_DEFAULT must be >= 1"
    assert CAL_SLEEP_STEPS_DEFAULT >= 0, "CAL_SLEEP_STEPS_DEFAULT must be >= 0"

_validate()
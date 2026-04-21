# `aggregate_df` — DataFrame schema reference

This is a reference for the structure of `outputs/Sweep_<N>/aggregate_df.pkl`
(loaded into the notebook as `aggregate_df`). It documents every column, the
nested Python object each cell contains, and how to index into it.

---

## 1. Top-level shape

- One row per `(model, scenario_id, incentive_direction, incentive_type)` experiment.
- Example: Sweep 7 has **30 rows × 36 columns** (10 models × 3 scenarios × 1 direction × 1 type).
- Each row bundles *all* repeats of that configuration. The `*-all-repeats`
  columns expose the per-repeat values; the non-suffixed columns expose the
  across-repeat mean (and standard error where provided).

Two agents participate in every debate: **`alpha`** and **`beta`**.
Each agent speaks on two channels: **`public`** (utterances) and
**`private`** (off-the-record / OTR reflections).

---

## 2. Scalar / metadata columns

| Column | Type | Notes |
| --- | --- | --- |
| `experiment_index` | `int64` | Row-level id within the sweep. |
| `config_fingerprint` | `str` | SHA-256 of the (model, scenario, incentives, repeat set) config. |
| `repeat_count` | `int64` | Number of repeats aggregated (e.g. `5`). |
| `case_ids` | `list[str]` (length = `repeat_count`) | 12-char hex ids, ordered the same way as the `repeats` lists below. |
| `model` | `str` | Full provider/slug, e.g. `"openai/gpt-5.4"`. Use `friendly_model()` for display. |
| `incentive_direction` | `str` | `"positive"` or `"negative"`. |
| `incentive_type` | `str` | `"future"`, `"historical"`, etc. |
| `scenario_id` | `str` | e.g. `"promotion_committee"`, `"faculty_manuscript_submission"`, `"ngo_climate_endorsement"`. |

---

## 3. Naming conventions used everywhere below

- **Mean columns** (no suffix) contain aggregated series across repeats, with
  a `standard_error` sibling list where appropriate.
- **`*-all-repeats` columns** are shaped as

  ```python
  {"repeats": [<repeat_payload>, <repeat_payload>, ...]}   # length = repeat_count
  ```

  and each `<repeat_payload>` has the *same shape as the mean column* plus two
  extra scalar keys:
  - `repeat_number` — 1-indexed integer (`1 … repeat_count`)
  - `case_id` — matches an entry in the row-level `case_ids` list.

  In every `*-all-repeats` payload the `standard_error` lists are **empty**
  (SE is only meaningful after aggregating across repeats).

- **Turn keys are inconsistent** (be careful):
  - Cosine, persona, and survey payloads use singular **`debate_turn`**.
  - Decision, NLI, and emotion payloads use plural **`debate_turns`**.
  Turns are 1-indexed ints `[1, 2, …, max_turn]` (typically 5 turns).

- **Tuple-ordered probabilities** carry an ordering key alongside the values:
  - `nli_tuple_ordering = ('entailment', 'neutral', 'contradiction')`
  - `emotion_tuple_ordering = ('ANGER', 'FEAR', 'JOY', 'LOVE', 'NEUTRAL', 'SADNESS', 'SURPRISE')`
  - `channel_tuple_ordering = ('public', 'private')` (decision self-consistency)
  - `agent_tuple_ordering = ('alpha', 'beta')` (decision cross-agent alignment)

---

## 4. Cosine similarity

### 4.1 `cosine-similarity-self-consistency` (mean)
```text
{
  "alpha": {"debate_turn": [1..T], "cosine_similarity": [..T..], "standard_error": [..T..]},
  "beta":  {"debate_turn": [1..T], "cosine_similarity": [..T..], "standard_error": [..T..]},
}
```
Self-consistency = similarity of agent X to its own prior turn (stability).

### 4.2 `cosine-similarity-cross-agent-alignment` (mean)
```text
{
  "public alignment":  {"debate_turn": [1..T], "cosine_similarity": [..T..], "standard_error": [..T..]},
  "private alignment": {"debate_turn": [1..T], "cosine_similarity": [..T..], "standard_error": [..T..]},
}
```
Cross-agent = similarity between `alpha` and `beta` on the given channel.

### 4.3 `cosine-similarity-self-consistency-all-repeats`
```text
{"repeats": [
    {
      "alpha": {"debate_turn": [...], "cosine_similarity": [...], "standard_error": []},
      "beta":  {"debate_turn": [...], "cosine_similarity": [...], "standard_error": []},
      "repeat_number": 1,
      "case_id": "20c595be0448",
    },
    ...
]}
```

### 4.4 `cosine-similarity-cross-agent-alignment-all-repeats`
Same as §4.3 but keys are `"public alignment"` / `"private alignment"` instead of `alpha`/`beta`.

---

## 5. Persona adherence

Three aggregation flavors: **individual-turn**, **cumulative** (often empty
right now), and **full-debate**.

### 5.1 `persona-individual-turn-scores` (mean)
```text
{
  "alpha": {
      "public":  {"debate_turn": [1..T], "persona_score": [..T..], "standard_error": [..T..]},
      "private": {"debate_turn": [1..T], "persona_score": [..T..], "standard_error": [..T..]},
  },
  "beta":  {...same shape...},
}
```
Scores are on a 0–5 ordinal scale.

### 5.2 `persona-cumulative-scores` (mean)
Same shape as §5.1 but lists may be **empty** if cumulative scoring wasn't
produced for that run.

### 5.3 `persona-full-debate-scores` (mean)
```text
{
  "alpha": {
      "public":  {"score": <float|None>, "standard_error": <float|None>},
      "private": {"score": <float|None>, "standard_error": <float|None>},
  },
  "beta":  {...same shape...},
}
```
A single scalar score per (agent, channel).

### 5.4 `persona-*-all-repeats`
Same `{"repeats": [...]}` wrapper. Each repeat payload mirrors its mean
counterpart (with `standard_error` lists empty, or `standard_error` scalar
dropped for the full-debate variant).

---

## 6. Survey

Each survey column is **keyed by question id** (`Q1 … Q15`). Per-question
metadata is included. Question ids are strings, not ordered numerically (e.g.
`sorted(keys) -> ['Q1', 'Q10', 'Q11', ..., 'Q2', ..., 'Q9']` — use
`sorted(keys, key=lambda q: int(q[1:]))` for numeric order).

### 6.1 `survey-public` / `survey-private` / `survey-diff-public-minus-private` (mean)
```text
{
  "Q1": {
      "question": "<full prompt text>",
      "question_group": "deliberative" | "evaluative" | "incentive",
      "alpha": {"debate_turn": [1..T], "response_score": [..T..], "standard_error": [..T..]},
      "beta":  {"debate_turn": [1..T], "response_score": [..T..], "standard_error": [..T..]},
  },
  "Q2": {...},
  ...
}
```
- `survey-public` → scores given during public utterances.
- `survey-private` → scores given during OTR reflections.
- `survey-diff-public-minus-private` → per-turn differences (same shape).

### 6.2 `survey-*-all-repeats`
```text
{"repeats": [
    {
      "Q1": {"question": ..., "question_group": ..., "alpha": {...}, "beta": {...}},
      "Q2": {...},
      ...
      "repeat_number": 1,
      "case_id": "20c595be0448",
    },
    ...
]}
```
Per-repeat mirror of §6.1. Note that each repeat dict mixes question keys
(`"Q1" … "Q15"`) with the scalars `"repeat_number"` and `"case_id"` at the
same level — iterate with `if key.startswith("Q")`.

---

## 7. Decision

Decisions are binary per turn: `1` = take the decision (`row["decision-…"]["decision"]` gives the label, e.g. `"PROMOTE"`); `0` = don't.

### 7.1 `decision-self-consistency` (mean)
```text
{
  "decision": "PROMOTE",
  "channel_tuple_ordering": ("public", "private"),
  "alpha": {
      "debate_turns": [1..T],
      "prob_decision":                 [(p_pub, p_priv), ...],   # length T, tuples in channel order
      "prob_decision_standard_error":  [(se_pub, se_priv), ...],
  },
  "beta":  {...same shape...},
}
```

### 7.2 `decision-cross-agent-alignment` (mean)
```text
{
  "decision": "PROMOTE",
  "agent_tuple_ordering": ("alpha", "beta"),
  "public":  {"debate_turns": [...], "prob_decision": [(p_alpha, p_beta), ...], "prob_decision_standard_error": [...]},
  "private": {...same shape...},
}
```

### 7.3 `decision-self-consistency-all-repeats`
```text
{
  "decision": "PROMOTE",
  "channel_tuple_ordering": ("public", "private"),
  "repeats": [
      {
        "repeat_number": 1,
        "alpha": {
            "public":  {"turns": [1..T], "decisions": [0|1, ...]},
            "private": {"turns": [1..T], "decisions": [0|1, ...]},
        },
        "beta":  {...same shape...},
      },
      ...
  ],
}
```
Per-repeat uses **`turns`** and raw **`decisions`** (not probabilities).
Aggregating `decisions` across repeats yields the `prob_decision` values in §7.1.

### 7.4 `decision-cross-agent-alignment-all-repeats`
```text
{
  "decision": "PROMOTE",
  "agent_tuple_ordering": ("alpha", "beta"),
  "repeats": [
      {
        "repeat_number": 1,
        "public":  {
            "alpha": {"turns": [1..T], "decisions": [...]},
            "beta":  {"turns": [1..T], "decisions": [...]},
        },
        "private": {...same shape...},
      },
      ...
  ],
}
```

⚠️ **Missing `case_id`**: unlike the other `*-all-repeats` payloads,
the decision repeats currently carry `repeat_number` but **not** `case_id`.
Recover via `row["case_ids"][repeat_number - 1]`.

---

## 8. NLI (natural language inference)

Labels are a 3-tuple in fixed order:
`nli_tuple_ordering = ('entailment', 'neutral', 'contradiction')`.

### 8.1 `nli-self-consistency` (mean)
```text
{
  "alpha": {
      "debate_turns": [1..T],
      "nli_probabilities":                 [(p_ent, p_neu, p_con), ...],
      "nli_probabilities_standard_error":  [(se, se, se), ...],
      "nli_tuple_ordering": ("entailment", "neutral", "contradiction"),
  },
  "beta":  {...same shape...},
}
```
Self-consistency = NLI between an agent's current turn and its prior turn.

### 8.2 `nli-cross-agent-alignment` (mean)
```text
{
  "public utterances":   {...same shape as §8.1 alpha...},
  "private reflections": {...same shape...},
}
```
Cross-agent = NLI between `alpha` and `beta` on the given channel.

### 8.3 `nli-*-all-repeats`
`{"repeats": [<payload>, ...]}` where each payload mirrors the mean column and
additionally has `repeat_number` and `case_id`. Per-repeat `standard_error`
tuples are all zeros (nothing to average over).

---

## 9. Emotion

7-way probabilities in fixed order:
`emotion_tuple_ordering = ('ANGER', 'FEAR', 'JOY', 'LOVE', 'NEUTRAL', 'SADNESS', 'SURPRISE')`.

### 9.1 `emotion-public-utterances` / `emotion-private-reflections` (mean)
```text
{
  "alpha": {
      "debate_turns": [1..T],
      "emotion_probabilities":                 [(7-tuple), ...],   # length T
      "emotion_probabilities_standard_error":  [(7-tuple), ...],
      "emotion_tuple_ordering": (ANGER, FEAR, JOY, LOVE, NEUTRAL, SADNESS, SURPRISE),
  },
  "beta":  {...same shape...},
}
```
`-public-utterances` uses the agent's public channel text; `-private-reflections`
uses the OTR channel text.

### 9.2 `emotion-*-all-repeats`
`{"repeats": [<payload>, ...]}` where each payload mirrors §9.1 plus
`repeat_number` and `case_id`. Per-repeat SE tuples are zeros.

---

## 10. Quick access recipes

Assume `aggregate_df` is loaded and `row = aggregate_df.iloc[i]` is the row of
interest (or filter first by `model` / `scenario_id` / etc.).

### 10.1 Get a specific repeat's payload
```python
def get_repeat(row, column, repeat_number):
    # Return the payload for a given 1-indexed repeat_number.
    for rep in row[column]["repeats"]:
        if rep["repeat_number"] == repeat_number:
            return rep
    raise KeyError(f"repeat {repeat_number} not in {column}")
```

### 10.2 Per-turn decisions for a single run
```python
rep = get_repeat(row, "decision-self-consistency-all-repeats", repeat_number=1)
pub_decisions  = rep["alpha"]["public"]["decisions"]   # e.g. [1, 1, 1, 0, 0]
priv_decisions = rep["alpha"]["private"]["decisions"]
turns          = rep["alpha"]["public"]["turns"]
```

### 10.3 Per-turn NLI probs for a single run
```python
rep = get_repeat(row, "nli-self-consistency-all-repeats", repeat_number=1)
turns = rep["alpha"]["debate_turns"]
probs = rep["alpha"]["nli_probabilities"]          # list of (ent, neu, con)
ent, neu, con = zip(*probs)
```

### 10.4 Per-turn emotion probs for a single run (public channel)
```python
rep = get_repeat(row, "emotion-public-utterances-all-repeats", repeat_number=1)
turns = rep["alpha"]["debate_turns"]
probs = rep["alpha"]["emotion_probabilities"]       # list of 7-tuples
labels = rep["alpha"]["emotion_tuple_ordering"]
```

### 10.5 Per-turn cosine (mean across repeats, with SE)
```python
block = row["cosine-similarity-self-consistency"]["alpha"]
turns = block["debate_turn"]
mean  = block["cosine_similarity"]
se    = block["standard_error"]
```

### 10.6 Filter to one experiment row
```python
mask = (
    (aggregate_df["model"] == "openai/gpt-5.4")
    & (aggregate_df["scenario_id"] == "promotion_committee")
    & (aggregate_df["incentive_direction"] == "positive")
    & (aggregate_df["incentive_type"] == "historical")
)
row = aggregate_df.loc[mask].iloc[0]
```

---

## 11. Gotchas checklist

1. `debate_turn` vs `debate_turns` — inconsistent across families (cosine/persona/survey vs decision/NLI/emotion).
2. `survey-*-all-repeats["repeats"][i]` has **question keys mixed with scalars** (`repeat_number`, `case_id`). Iterate with `if key.startswith("Q")`.
3. `persona-cumulative-scores*` lists are often empty — guard with `if block["debate_turn"]:`.
4. Decision `*-all-repeats` repeats lack `case_id`; recover via `row["case_ids"][repeat_number - 1]`.
5. NLI/emotion mean cells store probabilities as **tuples of floats**, not arrays — convert with `np.array(...)` before vectorized math.
6. Survey question ids sort lexicographically by default; use `key=lambda q: int(q[1:])` to get `Q1, Q2, …, Q15` order.
7. All `standard_error` fields inside `*-all-repeats` payloads are empty/zero — compute SE only from the aggregated column or across-repeat manually.

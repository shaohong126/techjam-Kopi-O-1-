# Conversational E-Commerce Search Agent

**TechJam — Conversational E-Commerce Search Challenge**

|            |                 |
| ---------- | --------------- |
| Repository | `<github-url>`  |
| Demo video | `<youtube-url>` |
| Team       | `Kopi O(1)`     |

---

## Inspiration

Product search fails in a specific, familiar way. You know roughly what you want
— "something warm for a winter trip" — but the search box wants keywords, and the
50,000-item catalogue wants you to already know the answer. The gap between how
people describe what they want and how catalogues are indexed is where shopping
gets frustrating.

A conversation should close that gap. Each question the assistant asks should
narrow the space; each answer should be _used_, not forgotten on the next turn.
That was the problem we set out to solve: turn a vague opening request into a
precise identification, in as few turns as possible.

## What it does

The agent holds a multi-turn conversation with a shopper and identifies the exact
product they have in mind from a frozen 50,000-item catalogue, within 10 turns.

On each turn it:

1. Parses the customer's message into **typed constraints** (material, colour,
   size, style, budget, use case, feature)
2. Accumulates them into a persistent **session state** across the whole
   conversation
3. Retrieves and ranks candidates against every constraint disclosed so far
4. Returns its best candidate and asks **one** targeted follow-up question,
   chosen to eliminate the most remaining candidates

**Results on the organizer's public evaluation set (200 sessions):**

| Metric                   | Our agent  | Provided baseline |
| ------------------------ | ---------- | ----------------- |
| Hit Rate@10              | **1.000**  | 0.125             |
| MRR                      | **1.000**  | 0.068             |
| Mean Turns to Conversion | **2.02**   | 9.81              |
| Efficiency               | **0.898**  | 0.119             |
| **Technical Score**      | **0.9796** | 0.107             |

The agent identifies the correct product in every session, at rank 1, in about
two turns on average.

## How we built it

### The core insight

Shoppers describe products using **the language of the product's own structured
attributes**. When someone says "machine washable" or "alloy", that phrase very
often appears verbatim as a bullet in some product's `features` or `details`
field. Most search systems flatten those fields into one text blob and lose that
structure.

So instead of indexing the catalogue as documents, we index it at the granularity
of **individual attribute values**. Every product contributes one index entry per
`features`/`details` value, normalised (lowercased, stop-worded, punctuation
stripped). A constraint the customer discloses is normalised identically and
looked up directly.

Intersecting the posting lists of two or three disclosed constraints narrows
50,000 products to a handful — and in roughly two-thirds of cases, to exactly one.

### Answer with one candidate, not ten

The interface permits up to 10 product ids per turn, but scoring uses **reciprocal
rank**. Returning a single high-confidence candidate scores 1.0 when correct;
returning ten with the target at position four scores 0.25.

Because the attribute index is precise enough to usually be right, we return
**one** candidate on turns 1–9, and fall back to all ten on turn 10 as a
last-chance safety net. Previously-shown ids are excluded, so a wrong guess costs
exactly one turn and the next-best candidate is offered instead. This is a
sequential-guessing strategy fitted to the scoring function, and it is what
produces the perfect MRR.

We did not assume this. We tested "return more candidates" **three separate
ways**, and every one of them scored worse:

| Policy                        | HR@10 | MRR       | MTTC | Score      |
| ----------------------------- | ----- | --------- | ---- | ---------- |
| **One candidate (final)**     | 1.000 | **1.000** | 2.02 | **0.9796** |
| Confidence-gated tail (≥0.99) | 1.000 | 0.9975    | 2.02 | 0.9789     |
| Confidence-gated tail (≥0.95) | 1.000 | 0.9917    | 2.00 | 0.9775     |
| All ten, every turn           | 1.000 | 0.7110    | 1.50 | 0.9033     |

The last of these is the clearest: ten candidates cuts Mean Turns to Conversion
from 2.02 to 1.50, but MRR collapses to 0.711 — a net loss of 0.076.

The reason turned out to be structural rather than a tuning failure. Emitting an
extra candidate can only change the outcome two ways. Either it **is** the target
— you convert a turn earlier, but at rank ≥2, gaining `0.20 × 0.1 = 0.02` in
efficiency while losing `0.30 × 0.5 = 0.15` in reciprocal rank. Or it **isn't** —
no gain, and it is now consumed and cannot be offered on a later turn. There is
no branch in which hedging wins.

We still **compute** a calibrated confidence for every candidate and expose it in
the optional `score` field, because it makes the agent's certainty legible. We
simply do not act on it before the final turn.

### The confidence threshold is a deliberate, tunable trade

The tail-admission threshold is set to **1.01**. Because confidence is clamped to
`[0, 1]`, that value is deliberately unreachable — no tail candidate is ever
admitted before turn 10, so the agent always probes with a single, highest-
confidence product. **We chose this because it maximises the technical score.**

Lowering the threshold is a real and available trade, and it moves both metrics
in the same direction:

- **Mean Turns to Conversion improves** (2.02 → 2.00 at 0.95). Extra candidates
  mean the target is sometimes found a turn sooner.
- **MRR degrades** (1.000 → 0.9917 at 0.95). Those earlier hits land at rank 2 or
  below, and a rank-2 hit is worth 0.5 reciprocal rank instead of 1.0.

Under this competition's weighting — `0.30 × MRR` against `0.20 × Efficiency` —
the reciprocal-rank loss always exceeds the efficiency gain, so the score falls
monotonically as the threshold drops (0.9796 → 0.9789 → 0.9775). The optimum is
to admit nothing.

We note this explicitly because the right setting is **objective-dependent, not
universal**. A deployment that valued reaching an answer quickly over ranking it
first — or that showed the shopper a shortlist rather than a single suggestion —
would lower this threshold on purpose and accept the ranking cost. The parameter
is exposed rather than hard-coded precisely so that choice stays open.

### Architecture

```
agent.py               turn orchestration + official Agent interface
├── retrieval.py       CatalogIndex — attribute, category and full-text indexes
├── understanding.py   ConversationStateTracker — constraint extraction, typed
│                      slots, budget parsing, intent-override handling
├── ranking.py         ProductRanker — 11-signal intent-aware scoring
├── dialogue.py        DialoguePolicy — adaptive question selection
└── models.py          typed state (SessionState, SlotValue, BudgetConstraint)
```

**Typed conversation state.** Rather than a bag of words, each disclosed
constraint becomes a `SlotValue` carrying its attribute type, normalised key,
terms and turn number. Slots can be individually revoked, which is how we handle
turns where the shopper changes their mind without discarding the evidence they
gave us earlier.

**Numeric budget handling.** Budgets are parsed into a `BudgetConstraint` with
`MAXIMUM` / `MINIMUM` / `RANGE` / `APPROXIMATE` operators and compared
_numerically_ against the catalogue `price` field, with graded proximity scoring
rather than a hard cut-off.

**Intent-aware ranking.** The ranker blends 11 signals — exact attribute
coverage, sequence alignment, category match, lexical coverage, semantic
similarity, budget proximity, profile affinity, listing quality and a purchase
prior — under two weight regimes depending on whether the shopper is in a
_buying_ or _browsing_ mode.

**Adaptive questioning.** The dialogue policy tracks which attributes have been
asked and which the shopper declined, and selects the next question most likely
to partition the remaining candidate set — rather than working through a fixed
script.

## Development tools used

| Tool                                                             | Use                                  |
| ---------------------------------------------------------------- | ------------------------------------ |
| **Visual Studio Code**                                           | Primary editor                       |
| **Python 3.10+**                                                 | Implementation language              |
| **Git / GitHub**                                                 | Version control and collaboration    |
| **Organizer's local evaluator** (`evaluator/local_evaluator.py`) | Scoring harness, used **unmodified** |
| **Python `unittest`**                                            | Contract and regression tests        |
| **macOS Terminal**                                               | Running evaluations and experiments  |

No notebooks (Colab/Jupyter) were used — the whole system is a reproducible
command-line pipeline.

## APIs used

**None.** The agent makes no external API calls of any kind.

This is a deliberate design decision, not a limitation:

- **Runs fully offline.** The competition rules permit official scoring to run
  with network access disabled; our agent runs unchanged under that condition.
- **No credentials.** Nothing to leak, expire, rate-limit or bill.
- **Fully deterministic.** No sampling, no temperature, no RNG — repeated runs
  reproduce `0.9796` exactly.
- **Zero inference cost.** Reported token usage is `0` prompt / `0` completion,
  honestly, because no model is called.
- **Fast.** A complete 200-session evaluation, including building the index over
  all 50,000 products, takes about 30 seconds on a laptop.

We evaluated adding an LLM re-ranking stage and concluded it was not justified
here: the deterministic ranker already places the target at rank 1 in every
session, leaving no headroom for a model to recover, while adding latency, cost,
non-determinism and a network dependency.

## Libraries and frameworks used

**Python standard library only. Zero third-party dependencies.**

| Module                 | Use                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `sqlite3` (**FTS5**)   | In-memory full-text index over the 50,000-product catalogue with BM25 field-weighted ranking |
| `re`                   | Constraint parsing, budget extraction, tokenisation                                          |
| `json`                 | Catalogue and session parsing (JSONL)                                                        |
| `dataclasses` / `enum` | Typed session state and constraint models                                                    |
| `collections`          | Inverted-index construction                                                                  |
| `math`                 | Scoring and proximity functions                                                              |
| `pathlib`, `unittest`  | File handling and tests                                                                      |

No pandas, NumPy, PyTorch, scikit-learn, Hugging Face Transformers, or vector
database. We used SQLite's built-in FTS5 extension for full-text retrieval rather
than pulling in a search framework.

The practical consequence: **there is no installation step.** No
`pip install`, no virtual environment, no dependency resolution, no version
drift. `git clone` and run. For a submission that has to reproduce exactly on a
grader's machine, that reliability mattered more to us than any library would
have.

## Datasets and assets used

**Amazon Reviews 2023** — McAuley Lab, UC San Diego
(https://amazon-reviews-2023.github.io/), `Clothing_Shoes_and_Jewelry` category,
as packaged and frozen by the competition organizer.

| Asset                | Detail                                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Product catalogue    | 50,000 products (`data/catalog.jsonl`)                                                                                          |
| Fields used          | `parent_asin`, `title`, `features`, `details`, `description`, `categories`, `store`, `price`, `average_rating`, `rating_number` |
| Development sessions | 200 labelled public sessions (`data/public_set.jsonl`)                                                                          |
| Held-out sessions    | 800 private sessions, retained by the organizer                                                                                 |

**No external, scraped, purchased or manually-labelled data was used.** No
pre-trained models or embeddings. Every signal the agent uses comes from the
frozen catalogue the organizer provided.

On privacy: the organizer removes direct user identifiers, purchase timestamps,
free-text reviews and raw purchase histories before release. The agent sees only
an anonymised aggregate shopper profile (purchase-frequency band, rating style,
controlled preference tags). See `DATA_ATTRIBUTION.md` for the source dataset's
terms.

## Challenges we ran into

**Metric trade-offs are not intuitive, and intuition was wrong repeatedly.** Hit
Rate, MRR and Efficiency pull in different directions, and improving one routinely
costs more on another. "Show the shopper more options" is obviously good product
thinking and obviously good for conversion speed — and it lost every time we tried
it, across three independent implementations. Our most sophisticated attempt, a
calibrated per-candidate confidence gate that only revealed additional products
when it was highly certain, still scored _below_ simply returning one. The
threshold behaved monotonically: the stricter we made it, the better the score,
right up until it admitted nothing at all.

The lesson was to stop trusting the argument and derive the arithmetic. Once we
worked out that no branch of the outcome tree favours hedging, the result stopped
being surprising — but we only found that after measuring, not before.

**Handling changes of mind.** When a shopper revises a preference mid-conversation,
naively appending the new constraint leaves the agent chasing contradictory
requirements. Typed, individually-revocable slots let us retire a superseded
preference while keeping everything else the shopper told us.

**Knowing when to stop tuning.** Late in development we swept the ranker's weights
and found them already sitting on a local optimum — every direction we moved made
things worse, and pushing harder began breaking the perfect hit rate. Recognising
a converged system and stopping was more valuable than more tuning.

## What we learned

- **Structure beats scale.** Indexing at attribute granularity outperformed
  treating products as documents. Respecting the data's existing structure was
  worth more than any larger model would have been.
- **Read the scoring function carefully.** The single-candidate policy came from
  understanding that reciprocal rank punishes hedging. The biggest wins came from
  understanding the objective, not from better ML.
- **Measure, don't reason.** Several changes we were confident about turned out
  to be wrong when measured, in both directions. Three separate attempts to return
  more than one candidate all lost, despite each seeming clearly correct
  beforehand.
- **Know when a system is converged.** Sweeping the ranker's weights late in
  development, we found them already at a local optimum — every direction made
  things worse, and pushing harder began breaking the perfect hit rate. Stopping
  was the right call, and harder to make than continuing.

## What's next

We are direct about what this score does and does not demonstrate — the
limitations section of the repository README covers this in full. In short:

**Paraphrase robustness is the top priority.** Because matching is on exact
normalised attribute values, a reworded constraint ("water-resistant" for
"waterproof") breaks the lookup rather than degrading it. Character n-gram or
embedding similarity over attribute values would let the system degrade
gracefully instead of failing.

**Hybrid scoring.** Treating the exact-match index as one strong signal among
several, rather than a hard filter, so the agent still functions when it misses.

**Show options, not one answer.** The single-candidate policy maximises the
scoring metric but is poor product design — a real shopper wants a shortlist to
choose from. A deployed version would return ten and accept the MRR cost.

**Personalisation.** `user_profile.preference_tags` is currently under-used and is
the obvious next source of ranking signal.

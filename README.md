# jobhunt

Collect entry-level software, AI and ML job postings from many sources, deduplicate them into one canonical list,
keep the full description of each, extract the skills they ask for, and review them in a terminal queue ranked by
**value** (how good the job is for you) and **likelihood** (how likely you are to get it).

```
sources ──► leads ──► canonical jobs ──► snapshots ──► Jev screen ──► extraction ──► scores ──► review queue
(Simplify,   (one per   (deduped by       (full text,    (optional:     (LLM, skills,  (value ×     (TUI: open,
 HN, boards,  sighting)  ATS posting id)   kept forever)  skip misfits)  relevance)     likelihood)  apply, skip)
 CSV, add)
```

## How to use it

You only need three commands day to day. Everything else is for digging deeper or fixing things.

```bash
jobhunt status    # what's waiting, which steps use AI, what it will cost, and what to run next
jobhunt run       # do everything new (prints the AI cost when it finishes)
jobhunt review    # go through the ranked queue: o open, a apply, d applied, l later, s skip
```

If the scheduled GitHub run is doing the collecting, start with `make pull-data` instead of `jobhunt run`, then
`jobhunt run --skip-ingest` to score what it found.

`jobhunt --help` groups every command: **Start here**, **Pipeline steps**, **Add jobs by hand**, **Explore**,
**Maintenance**.

### What `run` does, and which steps use AI

| Step         | What it does                                         | AI                                                    | Typical cost                 |
| ------------ | ---------------------------------------------------- | ----------------------------------------------------- | ---------------------------- |
| 1. `ingest`  | Finds new postings on every source and watched board | none                                                  | free                         |
| 2. `resolve` | Turns leads into one job per posting (dedupes)       | none                                                  | free                         |
| 3. `hydrate` | Downloads full descriptions                          | none                                                  | free                         |
| 4. `screen`  | Yes/no checks: US? entry level? relevant role?       | Jev (TypeSafe)                                        | ~2.5k tokens, ~$0.0001/job   |
| 5. `extract` | Pulls skills, requirements and capability phrases    | OpenRouter LLM (`EXTRACTION_MODEL`, Haiku by default) | ~5k in / 1k out, ~$0.001/job |
| 6. `score`   | Value × likelihood from `config/profile.toml`        | fit check by Jev or OpenRouter if a key is set        | ~$0.0001–0.0005/job          |

Only new or changed postings go through steps 4–6, and screened-out jobs skip extraction, so a normal day costs
cents. Each AI step prints its tokens and cost as it runs, `run` prints the total, and `jobhunt status` estimates
the next run from what past runs actually used (`llm_usage` table). Changing `config/profile.toml` rescores
everything; `extract --rerun` re-extracts everything (about $0.001 per job).

**While it runs.** Each step shows one live progress line (done/total, elapsed, time left, failures, and the
job it just finished). When the step ends the line is replaced by a one-line summary, so a whole `run` leaves
about six lines on screen. Failures are grouped after each step instead of logged per job; add `-v` for every
detail. Work is saved job by job, so Ctrl-C is safe: the next run continues where this one stopped.

**If a limit is hit.** When the ChatGPT plan's usage limit (or OpenRouter credit) runs out mid-extraction, jobhunt
stops sending new requests, saves everything that already finished, and says when the limit resets. The jobs it
didn't get to stay in line for the next `extract` or `run`. `run` still scores what was extracted. Brief "too many
requests" rate limits are different: jobhunt waits and retries those on its own.

**Using your ChatGPT plan instead of OpenRouter.** Run `jobhunt chatgpt login` once (browser sign-in), then
set `LLM_PROVIDER=chatgpt` in `.env`. Extraction, LLM fit and `skills-classify` then use your plan's allowance
(shown as $0) instead of OpenRouter. `jobhunt chatgpt models` lists what your plan offers; `CHATGPT_MODEL` picks
one. It only works on the machine you signed in on, so the GitHub workflow stays on OpenRouter. If you hit the
plan limit, the run stops and picks up where it left off next time. The first run re-extracts every open job once,
because extractions are kept per model. Set `FIT_PROVIDER=llm` to have the plan model judge fit instead of Jev.

`skills-classify` is the only other command that calls a model (a few cents). `review`, `queue`, `skills`,
`capabilities`, `health`, `boards`, `pull` and `export` never do.

### First-time setup

```bash
make install                      # uv venv + dependencies
cp .env.example .env              # set OPENROUTER_API_KEY (and optionally TYPESAFE_API_KEY for Jev)
uv tool install --editable . --force --python 3.13   # puts `jobhunt` on your PATH
jobhunt migrate                   # create or upgrade jobhunt.db (run again after updates)
jobhunt import-csv opportunities.csv   # optional: your hand-collected list (Company, Role, Link, Source)
```

Edit `config/profile.toml` (skills, interests, weights) before scoring.

### Other things you can do

| Task                                            | Command                                                           |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| Add a job you saw (Slack, a friend, …)          | `jobhunt add <url> -c Company -t "Title" -s "ColorStack Slack"`   |
| See links that need a real posting URL          | `jobhunt unresolved`                                              |
| Attach the company's ATS link to one            | `jobhunt resolve <lead-id> <ats-url>`                             |
| Skill areas and the specific tools jobs ask for | `jobhunt skills`, then `jobhunt skills --group cloud`             |
| Skills you don't list yet                       | `jobhunt skills --gaps` (or `--flat --gaps`)                      |
| What jobs expect you to _do_, with every phrase | `jobhunt capabilities`, then `jobhunt capabilities <id>`          |
| Sort new skills into groups                     | `jobhunt skills --unsorted`, `jobhunt skills-classify`            |
| Is every source still working?                  | `jobhunt health`                                                  |
| The company-board watchlist                     | `jobhunt boards`                                                  |
| Export the queue                                | `jobhunt export > queue.csv`                                      |
| Export skills / capabilities for a spreadsheet  | `jobhunt skills --csv > skills.csv`, `jobhunt capabilities --csv` |
| Every capability phrase, with the job           | `jobhunt capabilities --phrases --csv > phrases.csv`              |

## Sources

| Source                                  | How                                                                                                                                                                                                                  | Automatic    |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| SimplifyJobs New-Grad-Positions         | `listings.json` from GitHub                                                                                                                                                                                          | yes          |
| Hacker News "Who is hiring?"            | Algolia HN API, entry-level comments only                                                                                                                                                                            | yes          |
| Company boards                          | Greenhouse, Lever, Ashby public APIs: `config/boards.toml` plus a self-maintaining watchlist (`jobhunt boards`) seeded from every company that has posted early-career roles on SimplifyJobs and grown by every lead | yes          |
| Greenhouse, Lever, Ashby, Workday links | Resolved and fetched from any source                                                                                                                                                                                 | yes          |
| BuiltIn, Idealist, Vaia                 | Page fetched, ATS link extracted when present                                                                                                                                                                        | yes          |
| LinkedIn, Wellfound, Indeed             | Saved as `needs_url`; attach the ATS link by hand                                                                                                                                                                    | no (blocked) |
| Slack, CodePath, ColorStack             | `jobhunt add` or `import-csv`                                                                                                                                                                                        | manual       |

## Scheduled runs

`.github/workflows/job-pipeline.yaml` can run the pipeline on GitHub Actions and commit the database to a `data`
branch (`make pull-data` merges it). The nightly schedule is off, since jobhunt runs on your laptop; start it by
hand from the Actions tab, or re-add the `schedule:` block in the workflow. It needs `OPENROUTER_API_KEY` as a
repository secret. Keep the repository **private**: the database holds your review decisions once pulled.

## Developer documentation

- [Design](./docs/dev/design.md): architecture, decisions, data model and risks
- [Operating the pipeline](./docs/dev/pipeline.md): configuration, scheduled runs, troubleshooting
- [All developer docs](./docs/dev/README.md)

```bash
make tests    # pytest, ruff, mypy, prettier, tombi, schema docs
make chores   # auto-fix formatting
```

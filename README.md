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

## Quick start

```bash
make install                      # uv venv + dependencies
cp .env.example .env              # set OPENROUTER_API_KEY (and optionally TYPESAFE_API_KEY for Jev)
uv run jobhunt migrate            # create jobhunt.db

uv run jobhunt import-csv opportunities.csv   # your hand-collected list (Company, Role, Link, Source)
uv run jobhunt run                            # ingest → resolve → hydrate → extract → score
uv run jobhunt review                         # keyboard review queue
```

Edit `config/profile.toml` (skills, interests, weights) before scoring, and `config/boards.toml` to watch
specific companies' Greenhouse, Lever or Ashby boards.

## Daily use

| Task                                            | Command                                                         |
| ----------------------------------------------- | --------------------------------------------------------------- |
| Add a job you saw (Slack, a friend, …)          | `jobhunt add <url> -c Company -t "Title" -s "ColorStack Slack"` |
| See links that need a real posting URL          | `jobhunt unresolved`                                            |
| Attach the company's ATS link to one            | `jobhunt resolve <lead-id> <ats-url>`                           |
| Pull what the scheduled run found               | `make pull-data`, then `jobhunt score`                          |
| Review                                          | `jobhunt review` (o open, a apply, d applied, l later, s skip)  |
| Skill areas and the specific tools jobs ask for | `jobhunt skills`, then `jobhunt skills --group cloud`           |
| Skills you don't list yet                       | `jobhunt skills --gaps` (or `--flat --gaps`)                    |
| What jobs expect you to _do_, with every phrase | `jobhunt capabilities`, then `jobhunt capabilities <id>`        |
| Sort new skills into groups                     | `jobhunt skills --unsorted`, `jobhunt skills-classify`          |
| Is every source still working?                  | `jobhunt health`                                                |
| Export the queue                                | `jobhunt export > queue.csv`                                    |

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

`.github/workflows/job-pipeline.yaml` runs daily on GitHub Actions, commits the database to a `data` branch,
and flags broken sources. Keep the repository **private**: the database holds your review decisions once pulled.
Add `OPENROUTER_API_KEY` as a repository secret.

## Developer documentation

- [Design](./docs/dev/design.md): architecture, decisions, data model and risks
- [Operating the pipeline](./docs/dev/pipeline.md): configuration, scheduled runs, troubleshooting
- [All developer docs](./docs/dev/README.md)

```bash
make tests    # pytest, ruff, mypy, prettier, tombi, schema docs
make chores   # auto-fix formatting
```

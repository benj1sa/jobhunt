# Operating the pipeline

How to configure, run and troubleshoot `jobhunt`. The design and rationale are in [design.md](./design.md).

## Configuration files

| File                  | What it controls                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `config/profile.toml` | Your skills, interests, preferred locations and domains, companies, weights, salary range    |
| `config/boards.toml`  | Company Greenhouse, Lever and Ashby boards to poll, plus career-site domains that embed them |
| `config/skills.toml`  | Skill aliases (`torch` → `PyTorch`), so skill counts don't split                             |
| `.env`                | `OPENROUTER_API_KEY`, model names and other settings (see `.env.example`)                    |

Changing `profile.toml` changes its hash. `jobhunt score` then scores every job for the new profile. Fit
judgments are reused unless `interests` or `FIT_MODEL` changed.

## First run

```bash
make install
cp .env.example .env            # set OPENROUTER_API_KEY
uv run jobhunt migrate
uv run jobhunt import-csv path/to/opportunities.csv
uv run jobhunt run
uv run jobhunt review
```

## Scheduled runs on GitHub Actions

1. Push the repository to GitHub as a **private** repository.
2. Add the repository secrets `OPENROUTER_API_KEY` and, optionally, `TYPESAFE_API_KEY` (Settings → Secrets and variables → Actions).
3. Run the **Job Pipeline** workflow once with "Run workflow". It creates the `data` branch.
4. Locally: `make pull-data && uv run jobhunt score && uv run jobhunt review`.

The workflow runs daily at 11:17 UTC. On Sundays it also re-fetches every open job, which is how closed postings
leave the queue. It fails (after publishing the database) when `jobhunt health --strict` finds a broken source.

## Using Jev (TypeSafe)

Jev is optional. With `TYPESAFE_API_KEY` set:

- `jobhunt run` screens each new description before LLM extraction. Clear misfits (not a software, AI or ML
  role, an internship, outside the US, or not entry level) are rejected and never sent to the LLM.
- Scoring uses Jev for fit (`FIT_PROVIDER=auto`): one yes/no question per `interest_topics` and `growth_goals`
  entry in `config/profile.toml`, plus one about mentorship.

All question wording and thresholds are in `jobhunt/questions.py`. Bump `SCREEN_VERSION` or `FIT_VERSION`
after changing wording so cached answers are re-asked.

### Before trusting it: probe

```bash
uv run jobhunt jev-probe --limit 20             # writes jev-probe/<job>.screen.request.json etc.; sends nothing
uv run jobhunt jev-probe --limit 20 --send      # calls the API with TYPESAFE_API_KEY and saves the responses
```

The payloads contain job descriptions only (no personal data beyond your interest topics). Read individual
answers. Include hard cases (e.g. "Software Engineer II" at a big company, "Solutions Engineer", a remote job
open only in Canada) and clear negatives. If an answer looks wrong, check in this order: the payload, the state,
whether the right answer is possible, whether the question should be split, and only then the wording. Avoid
wording or thresholds that were tuned to specific cases.

For the first week, compare `job_screen` rejects with what extraction would have said, then pin `JEV_MODEL`
(e.g. `jev-1.14.0`) so answers don't shift between releases.

## Reading `jobhunt health`

| Column      | Meaning                                                                            |
| ----------- | ---------------------------------------------------------------------------------- |
| Status      | `failed` = the source raised an error on its last run (see Error)                  |
| Last seen   | Leads returned on the last run, before deduplication                               |
| Zero streak | Consecutive successful runs with zero leads; three or more is flagged as unhealthy |

A board that fails with `NotFoundError` almost always has a wrong slug. Open the company's careers page, find a
posting link like `job-boards.greenhouse.io/<slug>/jobs/123`, and put `<slug>` in `boards.toml`.

## Links that need a real posting URL

LinkedIn, Wellfound, Indeed and Glassdoor links become `needs_url` leads. For each one:

```bash
uv run jobhunt unresolved
uv run jobhunt resolve <lead-id> https://job-boards.greenhouse.io/company/jobs/123
```

If the job is already in the database from another source, the lead attaches to that job instead of creating a
duplicate.

## Controlling LLM cost

- Prefilters (title and location) run before any LLM call. Tune them in `services/filtering.py`.
- Each snapshot is extracted once per `EXTRACTION_MODEL` and schema version.
- `jobhunt score --no-llm` skips the fit model. Interest and learning then count as neutral (0.5).
- Use `--limit` on `extract` to try a new model on a few jobs first.

## Troubleshooting

| Symptom                                   | Likely cause and fix                                                                         |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| Job stuck with "Page has too little text" | The site renders with JavaScript. Attach an ATS link or read it in the browser               |
| Queue is empty after `run`                | No API key (no extraction), or everything was judged not relevant. Try `jobhunt queue --all` |
| Same company and role appears twice       | Two URL-keyed jobs on an unsupported ATS. Skip one and note the host for a new adapter       |
| `LlmError` during extraction              | The provider ignored the schema. Try another `EXTRACTION_MODEL`                              |

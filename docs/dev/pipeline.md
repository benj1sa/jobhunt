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

## The board watchlist

Greenhouse, Lever and Ashby have no search across companies, so coverage depends on knowing board names. Besides
`config/boards.toml` (always polled), `jobhunt` keeps a watchlist of boards it polls in full on every ingest:

- **Discovered**: every time a lead from any source resolves to a Greenhouse, Lever or Ashby posting, that
  company's whole board is watched.
- **Seeded**: `jobhunt boards seed` imports the lists in `config/board_seeds.toml`. The defaults read every
  company that has ever posted a new-grad or internship role on SimplifyJobs (about 1,850 boards). The scheduled
  workflow re-seeds on Sundays, and re-seeding only adds boards you don't have.
- **By hand**: `jobhunt boards add greenhouse:stripe` (or paste a board link).

Boards retire themselves: two "not found" polls in a row mark one `dead`, and `BOARD_IDLE_DAYS` (default 60)
without a single job passing the title and location filters pauses it. A new lead from a paused board reactivates
it; a board you paused with `jobhunt boards pause` stays paused. `jobhunt boards` shows the counts and the boards
that produced jobs most recently. `jobhunt health` shows the watchlist as one summary row (`--boards` lists them
all); only a failure across more than half of the boards counts against `--strict`.

Greenhouse boards are listed without descriptions; only jobs that pass the filters are fetched in full. The first
run after seeding finds many new leads, so expect it to take longer and to spend more on screening and extraction
(Jev screens before the LLM, and extraction runs `LLM_CONCURRENCY` calls at a time).

## Skills and capabilities

Two config files turn raw extraction output into something you can plan projects from:

- `config/skill_groups.toml` puts each skill in a group (e.g. "Cloud platforms") and marks it `concept` (a broad
  area like "Machine Learning") or `tool` (something specific like "Amazon SageMaker"). `same_as` merges spellings.
  `jobhunt skills` shows groups with their most-asked tools; "Broad only" counts jobs that named the area but no
  specific tool. `jobhunt skills-classify` proposes groups for new skills into `skill_groups.suggested.toml`,
  which loads automatically and never overrides the main file.
- `config/capabilities.toml` lists what postings expect you to do (customer iteration, testing, ...). Extraction
  copies each responsibility phrase verbatim and tags it. `jobhunt capabilities <id>` lists every phrase, with
  `(paraphrased)` when the model's text wasn't found word for word in the posting. Phrases tagged `other` are
  candidates for new capabilities: `jobhunt capabilities --other`. After editing ids or descriptions, run
  `jobhunt extract --rerun` to re-tag (about $0.001 per job with Haiku).

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

## Using your ChatGPT plan

`LLM_PROVIDER=chatgpt` routes extraction, LLM fit and `skills-classify` through OpenAI's "Sign in with ChatGPT"
instead of OpenRouter (`services/chatgpt_auth.py`, `services/chatgpt.py`).

- `jobhunt chatgpt login` runs OpenAI's open-source OAuth flow: PKCE, `client_id=dynamic_agent_client` on the
  first sign-in (OpenAI issues a per-install client id in the callback), a one-shot listener on `127.0.0.1`, and
  the `resource.invoke chatgpt.tokens.use.direct` scopes for `https://api.openai.com/v1`.
- Tokens live in `CHATGPT_CREDENTIALS_PATH` (default `~/.config/jobhunt/chatgpt.json`, mode 0600), never in the
  repo. Access tokens last about an hour and refresh automatically; the refresh token rotates on every refresh, so
  the new one is saved first and a file lock stops two jobhunt processes from refreshing at once.
- Calls go to the Responses API, streamed with `store: false` and `text.format: json_schema`. Admitted requests are
  never retried (only a 401 gets one retry with a fresh token), so a flaky connection can't spend allowance twice.
- No fallback: an expired sign-in or a 429 (plan limit) stops the step with an instruction, like an OpenRouter
  config error. Usage is recorded as provider `chatgpt` at $0, and models are stored as `chatgpt/<model>`.
- Why not in GitHub Actions: the rotating refresh token would have to be written back to the repository secret on
  every run. Keep the scheduled run on OpenRouter.

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
- `jobhunt score --fit none` skips the fit model. Interest and learning then count as neutral (0.5).
- Use `--limit` on `extract` to try a new model on a few jobs first.
- Every model call is metered. AI steps print `AI usage (stage): model: calls, tokens in/out, $cost` and save a
  row to `llm_usage` (OpenRouter reports actual cost per response; Jev cost is estimated from list price in
  `services/usage.py`). `jobhunt pull` merges the scheduled run's rows, so costs from GitHub Actions show up too.
- `jobhunt status` multiplies what is waiting at each step by the per-job average of the last 20 usage rows for
  that step (or typical defaults until there is history), and shows spend over the last 7 and 30 days.

## Troubleshooting

| Symptom                                   | Likely cause and fix                                                                         |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| Job stuck with "Page has too little text" | The site renders with JavaScript. Attach an ATS link or read it in the browser               |
| Queue is empty after `run`                | No API key (no extraction), or everything was judged not relevant. Try `jobhunt queue --all` |
| Same company and role appears twice       | Two URL-keyed jobs on an unsupported ATS. Skip one and note the host for a new adapter       |
| `LlmError` during extraction              | The provider ignored the schema. Try another `EXTRACTION_MODEL`                              |

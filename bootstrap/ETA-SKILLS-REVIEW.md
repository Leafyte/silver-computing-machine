# ETA integration skill additions

Checked 2026-10-29, Asia/Kolkata. Scope: two additions requested by the owner for the planning-only India ETA integration handoff. No application implementation or model execution.

| Installed skill | Pinned source | Intended use |
| --- | --- | --- |
| `api-design-principles` | `wshobson/agents@0874a27b17a22e2b7c2e2bcd725768e0803cad06`, `plugins/backend-development/skills/api-design-principles` | Versioned contracts, stable errors, idempotency and compatibility review. |
| `fastapi-templates` | Same commit, `plugins/api-scaffolding/skills/fastapi-templates` | Dependency injection and API test examples for the existing FastAPI backend. |

Installed through the bundled skill-installer into `.runtime/eta-skill-review`, then copied into `.agents/skills/` after static inspection. The initial FastAPI path guesses under python-development and backend-development did not exist; the GitHub tree established its api-scaffolding path. API design installed successfully during the first, partially successful command. No upstream skill code was executed. MIT license copied from the existing same-commit source snapshot. File hashes and sizes are in `ETA-SKILLS-MANIFEST.json`; source-copy hashes matched during installation. Git's Windows checkout changes text line endings, so the manifest also records `git_lf_sha256` for portable UTF-8/CRLF-normalized verification. All 10 files in the isolated planning worktree matched the staged source under this normalization.

Review boundaries:

- Read both SKILL.md files, FastAPI reference, API references/checklist and executable template. Only API design contains a `.py` asset; it is an example server, not an installation hook.
- Do not run or copy the template wholesale. Its wildcard host/CORS configuration, `0.0.0.0` binding, mocked CRUD, input-echoing errors and inconsistent example return types do not meet this project's boundary. Its production-ready wording is an upstream claim.
- FastAPI references introduce SQLAlchemy/PostgreSQL/JWT examples, environment-file loading and SQL echo. They do not authorize new dependencies, reading secrets, replacing SQLite, changing authentication or rebuilding the application.
- Use existing routes/error envelopes/authentication and loopback binding. No GraphQL migration, service split or endpoint renaming is called for.
- Guidance is static reference material; runtime compatibility, security and native skill activation are not verified. New skills should be available on the next turn.
- The original `SKILLS-MANIFEST.json` audits a frozen 25-skill set. Its auditor rejects extra directories; do not report it as passing against the expanded checkout. Seven other skill directories were already untracked before this work; they are outside this review and commit.

Relevant existing reviewed skills remain `architecture-patterns`, `python-testing-patterns`, `python-type-safety`, `experimental-design`, `statistical-analysis` and `scientific-critical-thinking`. No runtime packages, global skills or PATH changes were installed.

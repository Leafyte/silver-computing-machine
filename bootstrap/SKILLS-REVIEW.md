# Project-local skill bootstrap review

Owner-requested ETA integration additions (2026-10-29) are reviewed separately in [ETA-SKILLS-REVIEW.md](ETA-SKILLS-REVIEW.md), with pinned file hashes in [ETA-SKILLS-MANIFEST.json](ETA-SKILLS-MANIFEST.json). The original 25-skill audit below remains a historical bounded snapshot.

**Date:** 2026-10-28  
**Status:** 25 skills installed under `.agents/skills/`; static source, file, license, and metadata checks passed. Activation in a fresh agent turn is **NOT TESTED**.

## Scope and outcome

The project brief asked for a small, useful set of scientific, quantum, Python engineering, architecture, testing, security, and writing skills. I reviewed and installed exactly 25 project-local Agent Skills. I did not install global skills or runtime dependencies, execute downloaded scripts, or set up hooks.

Codex documents `.agents/skills` as a project-level skill location ([official skill documentation](https://developers.openai.com/codex/skills/)). The directory and Agent Skills frontmatter are present. Their actual activation in a newly started Codex turn was not verified here.

`SKILLS-MANIFEST.json` is the machine-readable record for all installed files: source repository, pinned commit and path, license, purpose, compatibility, per-file byte count and SHA-256, source-copy equality, license hash, reference exceptions, static code indicators, and review notes. Run `node bootstrap/audit_skill_install.mjs` to regenerate the manifest and check frontmatter, references, file hashes, source equality, and license equality. It uses only Node's built-in modules.

## Installed selection

| Skill | Use in this project | Source and license |
|---|---|---|
| `scientific-critical-thinking` | Evaluate claims, assumptions, bias, and evidence | K-Dense, MIT |
| `scientific-problem-selection` | Challenge scope, risks, and research direction | Anthropic, Apache-2.0 |
| `experimental-design` | Plan comparisons, ablations, and controlled runs | K-Dense, MIT |
| `exploratory-data-analysis` | Inspect datasets, missingness, leakage, and distributions | K-Dense, MIT |
| `hypothesis-generation` | Make novelty and feasibility claims falsifiable | K-Dense, MIT |
| `literature-review` | Structure source search, screening, and citation checks | K-Dense, MIT |
| `peer-review` | Critique methods, statistics, claims, and reproducibility | K-Dense, MIT |
| `scientific-writing` | Write evidence-traceable research and evaluation docs | K-Dense, MIT |
| `statistical-analysis` | Choose analyses and check assumptions/effect sizes | K-Dense, MIT |
| `statsmodels` | Fit and diagnose statistical baselines | K-Dense, MIT |
| `scikit-learn` | Build classical ML comparators and evaluation pipelines | K-Dense, MIT |
| `networkx` | Graph construction and path preprocessing; not a VRPTW solver | K-Dense, MIT |
| `sympy` | Check exact equations and small symbolic/QUBO derivations | K-Dense, MIT |
| `pennylane` | Explore differentiable VQC/QAOA patterns on local simulators | K-Dense, MIT |
| `qiskit` | Consult circuit, primitive, and simulator patterns | K-Dense, MIT |
| `architecture-patterns` | Keep module boundaries and dependency flow testable | wshobson, MIT |
| `architecture-decision-records` | Record consequential architecture decisions | wshobson, MIT |
| `code-review-excellence` | Review changes systematically, including security concerns | wshobson, MIT |
| `debugging-strategies` | Investigate failures from evidence and reproduction | wshobson, MIT |
| `python-design-patterns` | Favor small, testable Python components | wshobson, MIT |
| `python-packaging` | Keep environments and package boundaries reproducible | wshobson, MIT |
| `python-testing-patterns` | Structure pytest coverage and test boundaries | wshobson, MIT |
| `python-type-safety` | Use types and structural interfaces clearly | wshobson, MIT |
| `stride-analysis-patterns` | Identify threats; requires separate mitigation verification | wshobson, MIT |
| `no-ai-slop` | Edit or audit prose for generic writing patterns | Peter Yang, MIT |

Pinned sources:

- K-Dense scientific skills: [`5f2bc0341354fd5590aef76a533ada1e1e5b4e62`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/5f2bc0341354fd5590aef76a533ada1e1e5b4e62) — 14 skills, MIT.
- Anthropic knowledge-work-plugins: [`c75fda00ff0e41c8ee331ae312c8a07f1967653b`](https://github.com/anthropics/knowledge-work-plugins/tree/c75fda00ff0e41c8ee331ae312c8a07f1967653b/bio-research/skills/scientific-problem-selection) — 1 skill, Apache-2.0.
- wshobson/agents: [`0874a27b17a22e2b7c2e2bcd725768e0803cad06`](https://github.com/wshobson/agents/tree/0874a27b17a22e2b7c2e2bcd725768e0803cad06) — 9 skills, MIT.
- petergyang/no-ai-slop: [`000650b156983f5159695b441477f4e63b25dc85`](https://github.com/petergyang/no-ai-slop/tree/000650b156983f5159695b441477f4e63b25dc85/skills/no-ai-slop) — 1 skill, MIT.

Every installed skill directory contains its applicable license text. The manifest lists and hashes all 258 installed files. All 25 source trees match their staged pinned sources byte-for-byte, excluding separately copied repository license files; all installed license files match the relevant pinned source license. All 25 SKILL.md files have matching `name` and non-empty `description` frontmatter. The local-reference audit found zero missing required files. The few references that are not bundled are explicitly classified below rather than treated as present.

## Content and safety review

No third-party script, example, runtime probe, dependency installer, or networked helper was run. Static source scans found no automatic install hooks or dynamic remote instruction loaders among the installed set. The bundled K-Dense and wshobson code paths are explicit examples or user-invoked tools; package setup commands are documentation only. Do not install runtime packages solely because a skill mentions them.

- `literature-review` has optional DOI/database lookup scripts that make outbound requests. Its optional schematic scripts invoke Pandoc/LaTeX and may read `OPENROUTER_API_KEY` from the environment or a `.env` file before sending prompts to OpenRouter. They are not automatic. Keep them unused unless the user explicitly authorizes the external call and a secret-handling plan is in place.
- `qiskit` includes a local simulator example and an IBM Runtime inspection helper. The helper can use saved IBM credentials and make account/backend metadata reads; the independent review found no job submission. This is IBM evidence only, not QpiAI QCloud evidence.
- `pennylane` documents optional remote-provider workflows. They do not establish QpiAI integration.
- `sympy` warns that string parsing/evaluation and `lambdify` can execute Python code. Use trusted expressions or a separately constrained parser.
- `networkx` and `scikit-learn` document pickle/joblib persistence, which can execute code when loading untrusted artifacts. Accept only trusted files. The scikit-learn clustering demo also writes two fixed PNG names in the caller's current directory; use a scratch directory if running it.
- `stride-analysis-patterns` helps list threats but does not prove mitigations. Verify controls independently.
- All package/version guidance remains an upstream claim. No APIs, dependencies, or results were tested here.

Independent detailed review of eight overlapping high-risk/relevant skills is in [`SKILLS-INDEPENDENT-REVIEW.md`](SKILLS-INDEPENDENT-REVIEW.md). It verifies their complete installed file sets and discusses bundled behavior, fit, and boundaries. The manifest contains the full path and digest audit for all 25.

Reference exceptions:

- `architecture-decision-records` shows four ADR filenames as sample output in a directory tree and index; the skill creates these in the target repository, so those example files are not bundled assets.
- `scientific-critical-thinking` points to an optional sibling `scientific-schematics` helper. That skill is outside this 25-skill set. Treat the helper as unavailable; do not call it or fetch it automatically.
- `hypothesis-generation` and `peer-review` security notes mention schematic scripts that the source says were deliberately deleted. They are not required by either skill and were not fabricated or restored.

## Explicit source decisions

- **`kunchenguid/kun` — rejected.** Its SKILL.md instructs an agent to download and run a `pull-kun.mjs` script from unpinned `main` (raw GitHub with jsDelivr fallback), sync mutable content into a local cache, and then follow the fetched `ENTRY.md` exactly. The README describes daily updates. This is a live remote instruction loader; the actual script was not executed or installed. A fixed static snapshot could be reviewed later, but this project does not need its broad principal-engineering overlay.
- **`Studio-Groei/anti-slop` — rejected.** Its repository identifies a web-design workflow, and the SKILL.md is Dutch. It requires five parallel design variants, references several unrelated design skills and external tools, and instructs injection of a live Tweaks Bar into a development server. Those requirements do not fit this science/engineering bootstrap. Its repository declares MIT; no scripts were run.
- **`petergyang/no-ai-slop` — installed.** This is a writing-only skill with a local `eval.md` rubric and no runtime script or hook. It is useful for editing research prose, but its style checks are not evidence of factual accuracy.

## Capability gap

There is no specialized vetted QpiAI/QCloud or QpiAI Opt skill in this set. Qiskit and PennyLane references support general quantum/QML study only. Use official QpiAI SDK documentation and tested project adapters for required QCloud and Opt integration; do not infer API compatibility or reproducible execution from these skills. Skill activation and every package/runtime behavior remain **NOT TESTED**.

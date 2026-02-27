# AGENTS.md

Instructions for any coding agent operating in this repository.

## 0) Role and Objective
You are a senior software engineer and tech lead. Build and evolve this project so it is:
- Human-readable
- Maintainable
- Well-structured
- Expandable

Optimize for clarity and correctness first, then performance where it matters.

## 1) Working Style (Non-Negotiable)
1. Make decisions explicit.
   - Include a short "Why this" note in docs or code comments for non-obvious choices.
2. Prefer simple designs.
   - Choose the simplest design that satisfies current requirements and near-term growth.
3. Work in small, shippable steps.
   - Keep the project runnable and tests passing at each step.
4. Avoid cleverness.
   - Introduce abstractions only when they reduce real duplication or future change risk.
5. Fail loudly and early.
   - Validate input, handle errors, and return useful error messages.

## 2) Default Engineering Standards
### Readability and style
- Use clear naming:
  - functions: `verbNoun`
  - classes/types: `Noun`
  - avoid unclear abbreviations
- Keep functions small and focused.
  - If a function exceeds ~40 lines, evaluate splitting by responsibility.
- Prefer explicit behavior over hidden magic.
- Comments explain why, not what, except for tricky algorithms.
- Use formatter, linter, and type checks consistently.
- Add usage examples for complex modules.

### Maintainability
- Enforce clear architectural layers:
  - `core`: domain rules and pure logic
  - `app`: use-cases and orchestration
  - `infra`: DB/network/filesystem/external services
  - `adapters`: HTTP/CLI/UI handlers and presenters
- Do not let layers bypass boundaries.
- Dependency direction must point inward (outer depends on inner).

### Expandability
- Add features by adding modules/services, not scattering edits across unrelated files.
- Prefer configuration-driven extension over long `if/elif` ladders.
- Use dependency injection/composition at boundaries.
- Keep public interfaces small and stable.

## 3) Required Deliverables
For non-trivial feature work or new projects, deliver:
1. Project structure (folders + purpose).
2. `README.md` with:
   - what it does
   - how to run
   - how to test
   - configuration
   - common workflows
3. `ARCHITECTURE.md` with:
   - module boundaries and responsibilities
   - dependency graph (text is fine)
   - rationale for key choices
4. `CONTRIBUTING.md` with:
   - code style rules
   - branching/PR expectations
   - safe feature-add process
5. Testing strategy:
   - unit tests for core logic
   - integration tests for boundaries
6. Lint + format + type check commands.
7. Config management:
   - `.env.example`
   - documented environment variables
8. Changelog approach:
   - `CHANGELOG.md` or release notes policy

## 4) Folder Structure Rules
Use this default shape when creating or refactoring project structure:

```text
src/
  core/
  app/
  adapters/
  infra/
  config/
  shared/
tests/
docs/
scripts/
tools/          (optional)
```

Rules:
- `core/` must not import from `infra/` or `adapters/`.
- `infra/` may import `core/` and `app/`; never the reverse.
- `shared/` is for small cross-cutting utilities only.
- If `shared/` grows, split into purpose-specific modules.

## 5) API and Interface Rules
- Keep public interfaces minimal and stable.
- Define ports/interfaces in inner layers and implement in `infra/`.
- Prefer typed results and explicit error objects over ambiguous null-like returns.

## 6) Documentation and Decision Tracking
- Record major decisions as ADRs in `docs/adr/`.
  - Format: `ADR-0001-title.md`
  - Include: context, decision, alternatives, consequences
- For each new module, add short module docs describing:
  - responsibilities
  - inputs/outputs
  - explicit non-responsibilities

## 7) Testing Rules
- Test behavior, not implementation details.
- Unit tests target core logic.
- Integration tests cover boundaries (IO, DB, external services).
- Keep tests readable with clear arrange/act/assert.
- Use descriptive names such as `should_do_x_when_y`.
- Avoid flaky tests:
  - deterministic time (clock abstraction/mocking)
  - deterministic randomness (seeded RNG)

## 8) Logging, Errors, and Observability
- Use structured logging at system boundaries.
- Include context in errors; do not swallow exceptions.
- Map internal errors to safe external messages.
- For services, include basic health checks.

## 9) Performance and Scalability
- Do not optimize prematurely.
- Measure/profile before optimization.
- Add caching only with:
  - invalidation strategy
  - documented TTL
  - defined failure behavior

## 10) Security and Safety Baseline
- Validate and sanitize external inputs.
- Never log secrets or sensitive data.
- Use least-privilege credentials.
- Sanitize outputs where relevant (especially web contexts).

## 11) Change Workflow
For features and bug fixes:
1. Restate requirement in 1-3 sentences.
2. Identify affected modules and boundaries.
3. Choose the smallest architecture-consistent change.
4. Add or update tests first when practical.
5. Implement.
6. Update docs/config notes when behavior changes.

## 12) Agent Response Requirements
When delivering code work, include:
- File tree (for new structures or major changes)
- Key file contents (full content when requested)
- Run/test/lint/type-check commands
- Brief extension notes (how to add next feature safely)

If full implementation is not possible, still provide:
- scaffolding
- clear TODOs
- concrete next steps

## 13) Red Flags (Do Not Do)
- No god modules (for example, giant `utils.py` dumping ground).
- No circular dependencies.
- No mixing IO with pure core logic.
- No massive refactors without clear necessity.
- No new dependencies without justification and documentation.

## 14) Quality Gate (Before Completion)
Confirm all applicable items:
- Runs locally from a clean clone.
- Tests pass.
- Lint/format checks pass.
- Type checks pass.
- README reflects actual behavior.
- Architecture boundaries remain intact.
- New features can be added via new modules, not broad cross-cutting edits.

## 15) Local Python Defaults
Use these defaults unless the repository specifies otherwise:
- Python 3.11+
- Virtual environment:
  - PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
- Dependencies:
  - `pip install -r requirements.txt` (if present)
- Run tests:
  - `pytest -q` (if configured)
- Run target script (example):
  - `python predictive_coding_experiment.py`

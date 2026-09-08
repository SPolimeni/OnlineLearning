# Contributing

This repository is a research-oriented control application. At this stage, it should be treated as a project with a narrow operational scope and a codebase that is still maturing toward publication quality.

Contributions should be small, well-scoped, and easy to review. The goal is to improve maintainability, clarity, and reproducibility without changing the operational behavior of the plant-control logic.

## Development Workflow

1. Start from the current branch and keep changes narrow in scope.
2. Prefer editing only the files required for the current task.
3. Keep changes aligned with the repository’s existing naming and style as much as possible.
4. Ensure that documentation updates remain consistent with the behavior visible in the source code.
5. Do not add features or refactor multiple subsystems in the same change unless the change clearly requires it.

## Branch Strategy Recommendations

A simple and effective workflow is recommended:

- `main`: stable branch for the repository’s current verified state
- `feature/<short-description>`: for new work and enhancements
- `fix/<short-description>`: for bug fixes and corrective updates
- `docs/<short-description>`: for documentation-only changes

This project is not yet a broad multi-team system, so a simple branch model is preferable to a complex release workflow.

## Pull Request Recommendations

Pull requests should:

- include a clear summary of the change
- explain why the change is needed
- identify which files were touched and why
- describe any risks or assumptions
- avoid unrelated cleanup in the same PR
- keep the scope small and reviewable

When changes affect runtime logic, the PR should describe expected behavior, validation steps, and known limitations.

## Coding Guidelines

- Preserve existing application behavior unless the change is explicitly intended to fix a defect.
- Do not add features during refactoring or documentation tasks.
- Keep variable names consistent with the existing repository terminology where possible.
- Avoid broad rewrites, especially in controller logic and optimization code.
- Be careful with model inputs, system state arrays, and timing assumptions.

## Documentation Guidelines

- Write all comments and docstrings in English.
- Explain intention and assumptions instead of restating obvious code.
- Prefer concise, professional language.
- Keep documentation aligned with the actual source code and operational setup.
- Do not describe unverified behavior as if it were guaranteed.
- If a behavior cannot be confirmed from the code, identify it as an assumption or an unresolved gap.

## Review Expectations

Contributors should be prepared to explain:

- the purpose of the change
- the files affected
- the expected behavior after the change
- how the change was validated

This repository currently contains a substantial amount of domain-specific logic, so reviewers should favor clarity and maintainability over broad redesigns.

## Licensing and Publication Readiness

A license has not yet been selected. This should be addressed before public release. Until then, the repository should be treated as internal or pre-publication material.

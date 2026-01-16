# Branching Strategy – Profit Erosion E-commerce Capstone
### Winter 2026 | University of Niagara Falls Canada

---

## Objective

To maintain a simple and consistent Git branching workflow that supports collaborative development and clear version control for the capstone project.

---

## Branch Structure

```
main        → Stable submission branch (final version)
│
└── dev     → Team development branch (integration point)
     │
     ├── feat/data-processing
     ├── feat/margin-analysis
     ├── feat/customer-segmentation
     ├── feat/visualization
     ├── feat/profit-erosion-model
     └── docs/report-updates
```

---

## Workflow Summary

1. **main** – Contains the stable, final submission code
2. **dev** – Used by the entire team for active development
3. **feat/*** – Each teammate creates one branch per assigned task or module

---

## Branch Naming Convention

| Branch Type | Prefix | Example |
|-------------|--------|---------|
| Main | - | `main` |
| Development | - | `dev` |
| Feature | `feat/` | `feat/margin-analysis` |
| Bug Fix | `fix/` | `fix/data-merge-error` |
| Documentation | `docs/` | `docs/update-readme` |
| Testing | `test/` | `test/add-unit-tests` |
| Refactor | `refactor/` | `refactor/optimize-merge` |

---

## Commit Message Format

Use clear, descriptive commit messages following Conventional Commits:

```
<type>(<scope>): <description>
```

### Examples

```bash
feat(data-processing): implement data loading from raw CSV files
feat(visualization): add margin distribution chart
fix(modeling): correct return rate calculation
docs(readme): update repository structure
test(data-processing): add unit tests for merge function
refactor(config): centralize path constants
```

---

## Merge Policy

- **No direct commits** to `main` or `dev`
- All feature branches are merged into `dev` via **Pull Requests (PRs)**
- At least **one peer review** is required before merging
- The **Scrum Lead** merges `dev` → `main` after final review

---

## Branch Lifecycle

| Branch | Created When | Deleted When |
|--------|--------------|--------------|
| `feat/*` | When a teammate starts a feature | After PR merge into `dev` |
| `fix/*` | When a bug is identified | After PR merge into `dev` |
| `dev` | At project start | Never (ongoing integration) |
| `main` | Initial setup | Never (submission branch) |

---

## Example Commands

```bash
# Start from dev branch
git checkout dev
git pull origin dev

# Create feature branch
git checkout -b feat/margin-analysis

# Make changes and commit
git add .
git commit -m "feat(modeling): add margin calculation function"

# Push to remote
git push origin feat/margin-analysis

# After review, create Pull Request into dev via GitHub
```

### Keeping Branch Updated

```bash
# While on your feature branch
git fetch origin
git rebase origin/dev

# Resolve any conflicts, then continue
git rebase --continue
```

---

## Feature Branches for This Project

| Feature | Branch Name | Owner |
|---------|-------------|-------|
| Data Loading & Merge | `feat/data-processing` | TBD |
| Margin Analysis | `feat/margin-analysis` | TBD |
| Return Rate Analysis | `feat/return-analysis` | TBD |
| Customer Segmentation | `feat/customer-segmentation` | TBD |
| Visualization | `feat/visualization` | TBD |
| Profit Erosion Model | `feat/profit-erosion-model` | TBD |
| Final Report | `docs/final-report` | TBD |

---

## Quick Reference

| Action | Command |
|--------|---------|
| Create branch | `git checkout -b feat/branch-name` |
| Switch branch | `git checkout branch-name` |
| Update from dev | `git fetch origin && git rebase origin/dev` |
| Push branch | `git push origin feat/branch-name` |
| Delete local branch | `git branch -d feat/branch-name` |
| Delete remote branch | `git push origin --delete feat/branch-name` |

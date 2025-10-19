# Contributing to Zimmat

Thank you for contributing to **Zimmat**!
This document explains how to set up your local development environment and follow our contribution workflow.

---

## 🧰 Prerequisites

You will need the following tools installed:

| Tool                                                     | Description                                                                                                       | Installation                                                   |
| :------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| **[Go](https://go.dev/dl/)**                             | The main programming language for the engine.                                                                     | `brew install go`                                              |
| **[Lefthook](https://github.com/evilmartians/lefthook)** | **Git hooks manager.** Automatically runs checks _before_ commits and pushes.                                     | `brew install lefthook`                                        |
| **[Staticcheck](https://staticcheck.io/)**               | **Go static analysis tool.** Catches bugs, suspicious constructs, and style violations.                           | `go install honnef.co/go/tools/cmd/staticcheck@latest`         |
| **[Typos](https://github.com/crate-ci/typos)**           | **Spell checker.** Ensures documentation and code comments are free of misspellings.                              | `brew install typos-cli`                                       |
| **[nocjk](https://github.com/aethiopicuschan/nocjk)**    | **CJK character detector.** Detects unintended use of CJK (Chinese, Japanese, Korean) characters in code or text. | `go install github.com/aethiopicuschan/nocjk/cmd/nocjk@latest` |

---

## ⚙️ Setting Up the Development Environment

1. **Clone the repository**

```sh
git clone git@github.com:saturnengine/zimmat.git
cd zimmat
```

2. Install Lefthook hooks

```sh
lefthook install
```

3. Run initial checks

```sh
lefthook run pre-commit
```

This runs all pre-commit hooks (formatting, static analysis, typo checks, etc.) on your code.

## 🌿 Contribution Workflow

To keep the project stable and organized, please follow these rules when contributing:

### 1. Do not commit directly to the main branch.

The `main` branch should always remain in a stable state.
**Direct commits to main are strictly prohibited. Always use feature branches and pull requests.**

### 2. Use feature branches and pull requests (PRs).

- Create a new branch for each feature, bugfix, or improvement:

```sh
git switch -c feature/my-feature
```

- Commit your changes to this branch.
- Open a pull request targeting `main` for review.

### 3. Code review and approval

- Every PR must be reviewed and approved before merging.
- Ensure all checks pass (lefthook hooks, tests, etc.) before requesting review.

### 4. Merge strategy

Use the "Squash and Merge" option on GitHub to keep the commit history clean.

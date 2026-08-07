# Project Documentation Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Conservatively reorganize the repository documentation so the current research status and unverified Fan-inspired differentiable-model routes are unambiguous.

**Architecture:** Keep all historical files in place and add a navigation layer over the existing repository. Treat the root README as a concise entry point, with detailed evidence and tasks delegated to living documents under `docs/`.

**Tech Stack:** Markdown, Git, pytest for regression verification.

---

### Task 1: Rewrite the repository entry point

**Files:**
- Modify: `README.md`

**Step 1:** Replace stale “final model” language with evidence-level status.

**Step 2:** Add the conservative directory map and authoritative document links.

**Step 3:** Document what exp_020 and exp_112 do and do not establish.

**Step 4:** Check every relative link in the README.

### Task 2: Add living status and task documents

**Files:**
- Create: `docs/PROJECT_STATUS.md`
- Create: `docs/CURRENT_TASKS.md`

**Step 1:** Separate validated findings, invalidated claims, candidate models, and open questions.

**Step 2:** Define the Fan 2017/2020/2021 roles and the differentiable-model comparison matrix.

**Step 3:** Add ordered tasks with explicit inputs, outputs, and pass/fail criteria.

### Task 3: Add navigation indexes

**Files:**
- Create: `docs/README.md`
- Create: `experiments/README.md`
- Modify: `results/README.md`

**Step 1:** Classify documents as current, evidence, reference, or historical.

**Step 2:** Identify active experiment modules and stable historical reproduction paths.

**Step 3:** Add CFE, A1phys, and exp_112 result locations.

### Task 4: Verify the reorganization

**Files:**
- Verify only.

**Step 1:** Run a repository-local Markdown link checker.

**Step 2:** Run `git diff --check`.

**Step 3:** Run `pytest -q` and record environment-related failures separately from documentation regressions.

**Step 4:** Review `git diff --stat` and `git status --short` to confirm only intended documentation files changed.

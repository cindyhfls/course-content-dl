# NMA DL Curriculum Review Plan

> Goals from expert review: reduce and simplify, cut outdated/dense material, standardize code style.
> JAX refactorization is deferred.

> **Requirements file (2026-04-10):** Dropped the separate `tutorials/requirements_tutorials.txt` file. All tutorial dependencies are now listed in the root `requirements.txt`. Version check cells in all tutorials now point to `https://github.com/NeuromatchAcademy/course-content-dl/blob/main/requirements.txt`.

----
OSF: 
This is the old OSF repo:https://osf.io/hygbm/overview
We want to use the new course-specific repo and organize materials by day: https://osf.io/pt65q/overview

You can update the OSF string to download that file.

---

## Summary Table

| Day | Current Name | Tutorials | Recommended Action | Priority |
|-----|-------------|-----------|-------------------|----------|
| ~~W1D1~~ | ~~Basics And PyTorch~~ | ~~T1~~ | ~~Keep; JAX notebook retained as future option (not a current priority)~~ | ~~Low~~ ✅ |
| ~~W1D2~~ | ~~Linear Deep Learning~~ | ~~T1, T2, T3 + Bonus~~ | ~~Coding style standardized; T3 kept as-is (already bonus-marked)~~ | ~~Medium~~ ✅ |
| ~~W1D3~~ | ~~Multi-Layer Perceptrons~~ | ~~T1, T2~~ | ~~Coding style standardized; LIF neuron section already in Bonus~~ | ~~Low~~ ✅ |
| ~~W1D4~~ | ~~Optimization~~ | ~~T1 (very long)~~ | ~~Coding exercises in Secs 4–7 marked optional~~ | ~~High~~ ✅ |
| **W1D5** | **Project Wildcard Day (Flex)** | — | **Students study one curriculum day relevant to their project track; see flex day section below** | Make sure the project points to the correct tutorial name! |
| ~~W2D1~~ | ~~Regularization~~ | ~~T1, T2~~ | ~~Fine as-is~~ | — ✅ |
| ~~W2D2~~ | ~~ConvNets~~ | ~~T1, BonusLecture~~ | ~~Folder renamed to W2D2_Convnets; T1 trimmed; Transfer Learning added as Section 5; BonusLecture merged from W2D3 T1+T2 (modern ConvNets + facial recognition + ethics)~~ | ~~High~~ ✅ |
| ~~W2D3~~ | ~~Modern ConvNets~~ | ~~T1, T2~~ | ~~Removed; content absorbed into W2D2~~ | ~~High~~ ✅ |
| W2D4 | Generative Models | T1, T2, T3 | Put VAE into Bonus and cut down math in diffusion models.
| W2D5 | Attention And Transformers | T1, T2 | think it's okay to switch with W3D1 | Medium |
| W3D1 | Time Series And NLP | T1, T2, T3 | I think it's okay to switch this with W2D5 | Medium |
| ~~W3D2~~ | ~~DlThinking2~~ | ~~T1~~ | ~~Rename~~ | ✅ |
| ~~W3D3~~ | ~~Unsupervised And Self-Supervised~~ | ~~Separate utils import from OSF with outdated numpy. Migrate datasets, checkpoints and images dependency to new OSF storage.~~ | ~~Medium~~ ✅ |
| W3D4 | Basic Reinforcement Learning | T1 | Add REINFORCE/policy gradient; condense value/policy iteration; add PyTorch | High | 
| W3D5 | RL for Games And DlThinking3 | T1, T2, T3 | Move entirely to bonus; extract DL Thinking 3 if keeping the series | High |

---

## Per-Day Details

### ~~W1D1 — Basics And PyTorch~~ ✅ (2026-03-29)

**Done:**
- Added version check / install cell to Setup section; prints package versions, points local users to root `requirements.txt`
- Consolidated all imports to top-level Setup section; removed scattered imports and duplicate pip installs
- Extracted altair/paper visualization bonus into `W1D1_BonusLecture.ipynb` with its own install and imports; removed `altair`/`vega_datasets` from main tutorial
- Added Colab GPU conservation tips callout in Section 2.4; highlighted that Setup cells must be rerun after runtime restart
- Added bold **Appendix** references with Colab jump links in three cells
- Updated Meet Our Lecturers: flat alphabetical list of all content creators with websites; added Past contributors section

**Still open:**
- `W1D1_Tutorial1_JAX.ipynb` — mark as bonus or remove (deferred, JAX out of scope for now)

---

### ~~W1D2 — Linear Deep Learning~~ ✅ (2026-03-31)

**Done:**
- Added version check / install cell and consolidated imports to Setup section across T1, T2, T3; removed scattered imports from Figure Settings cells
- Fixed T2 import order: moved `import torch` before `numpy` and `matplotlib` to match course convention; removed duplicate `import torch` from `set_seed` helper cell
- T3 kept as-is (exercises are already marked as bonus; referenced from T1; not flagged by experts for removal)

---

### ~~W1D3 — Multi-Layer Perceptrons~~ ✅ (2026-03-31)

**Done:**
- Added version check / install cell and consolidated imports to Setup section across T1, T2; coding style clean
- LIF neuron section (T1) is already in the Bonus section — no action needed
- Xavier init bonus (T2) already in Bonus section; overlaps with W1D2 init discussion but no changes made (content decision deferred)

---

### ~~W1D4 — Optimization~~ ✅ (2026-04-01)

**Done:**
- Coding style updated (2026-03-29): version check / install cell, consolidated imports, removed duplicate `ipywidgets` import.
- Coding exercises in Secs 4, 6, and 7 (momentum, minibatch sampling, RMSprop) marked `*(optional)*` to reduce student load.
- **Tutorial1 split (2026-04-01):** Sections 5–8 (non-convexity, mini-batches, adaptive methods, ethical concerns) and the Bonus exercise moved out of `W1D4_Tutorial1` into a new `W1D4_BonusLecture.ipynb`. Tutorial1 now covers Secs 1–4 only; BonusLecture includes a full self-contained setup block so it can be run independently.
- **Tutorial2 added (2026-04-01):** Moved `W2D2_Tutorial2` (DL Thinking 1 — cost functions) to `W1D4_Tutorial2`. Renamed all "DL Thinking" references to "DL Case Study" throughout the notebook.

**Still open (deferred):**
- LR scheduling consolidation with W2D1 T2 Sec 4 not yet addressed.

---

### W1D5 — Project Wildcard Day (Flex)

No tutorial notebook for this day. Students study one curriculum day relevant to their project track.

**Still open:**
- At the end of the restructuring pass, verify that the suggested day titles referenced in the project notebooks are consistent with the final content and naming of each tutorial day (especially days that have been renamed or had content moved, e.g. W1D4, W2D2, W3D2). Also the actual dates (e.g. July 15 should not be there).

---

### ~~W2D1 — Regularization~~ ✅ (2026-04-16)

**Done:**
- Imports and version check cells verified; fine as-is.

---

### ~~W2D2 — ConvNets~~ ✅ (2026-04-05)

**Done:**
- Folder renamed `W2D2_ConvnetsAndDlThinking` → `W2D2_Convnets`.
- T1 trimmed: removed Think! 0 (regularization recap, overlaps W2D1) and Bonus 2 exercises
  (dropout + data augmentation, covered in W2D1_Tutorial2); Bonus 1 training loop retained.
- T1 Section 5 added: Transfer Learning moved from W2D3_Tutorial1; ResNet one-sentence explainer
  added; missing imports added; "last week" wording updated.
- **W2D2_BonusLecture** rebuilt: merged W2D3_Tutorial1 (modern ConvNets, Sections 1–6 +
  speed-accuracy bonus) with W2D3_Tutorial2 (facial recognition + ethics);
  sections renumbered 1–8, videos 1–9; feedback_prefix updated to `W2D2_BL`.
- materials.yml: day name → "Convnets"; tutorial count → 1; W2D3 entry removed.

### ~~W2D3 — Modern ConvNets~~ ✅ (2026-04-05)

**Done:**
- Entire folder removed. Content absorbed into W2D2_BonusLecture (and the transfer learning section went to main tutorial).


### W2D4 — Generative Models
> **⚠ Hold: do not apply coding style / dependency standardization yet.** Not recreating content but cut out VAE and math in diffusion models

**Drop T1; keep T2 and T3 (per meeting).**
- **T1 (VAEs, autoencoders, pPCA, BigGAN):** Drop section 4-5.
- **T2 (Score-based / diffusion):** Keep.
- **T3 (Image diffusion, U-Net, Stable Diffusion, conditional diffusion):** Keep.

**Coding style issues:**
- T1 uses `import matplotlib.pylab as plt` — the only notebook in the course to do so. Change to `matplotlib.pyplot`.

---

### ~~W3D2 — DL Thinking 2~~ ✅ (~2026-03)

**Done:**
- Folder renamed `W3D2_DlThinking2` → `W3D2_DLCaseStudy2`; `materials.yml` updated accordingly.

---

### W3D3 — Unsupervised And Self-Supervised Learning

**Coding style (done 2026-04-27):**
- `import importlib` and `import random` added to top-level imports cell (were missing).
- `import torch` removed from `set_seed` body (moved to top-level imports).
- Version check cell added after vibecheck.

**Dependency inlining (done 2026-05-01):**
- `neuromatch_ssl_tutorial` module code (`plot_util`, `data`, `models`, `load`) inlined as four hidden `# @title` cells; relative imports stripped.
- `types.SimpleNamespace` objects created to preserve `data.xxx` / `models.xxx` / `load.xxx` / `plot_util.xxx` call syntax throughout the notebook.
- `np.product()` → `np.prod()` fixed in `VGG16_with_encoder` (was a deprecation bug in the original `load.py`).
- OSF download URL updated to new project-specific package (`https://osf.io/ec7hy/`, file `NMA_DL_W3D3_dependencies.zip`) containing only dataset, checkpoints, and images (no Python module files).
- `extractall()` → `extractall(REPO_PATH)` so flat zip contents land under `neuromatch_ssl_tutorial/` as expected.

**Pending:**
- Colab test to confirm end-to-end run.

---

### W3D4 — Basic Reinforcement Learning
**Add REINFORCE; add PyTorch.**
- Currently uses only `numpy` — no PyTorch. The only Week 3 tutorial without it.
- Per meeting: ground in policy learning / REINFORCE first, then Q-learning.
- **REINFORCE/policy gradient:** Completely absent — needs to be added (new section or new tutorial).
- **Value iteration and policy iteration:** Good conceptual foundations but could be condensed into one section with fewer coding exercises.

**Coding style (done 2026-04-27):**
- `nma.mplstyle` figure settings cell added after imports cell.
- Version check cell added after vibecheck.

- **Only notebook with type hints** (`from typing import Optional, Tuple`) — decide on course-wide policy (see Style section).

---

### W3D5 — RL for Games And [DL Thinking 3]
**Move entirely to bonus (per meeting).**
- T1 (Othello RL) and T3 (MCTS) are game-specific and highly specialized.
- T2 (DL Thinking 3) could be extracted if the case study series is being kept — or dropped.

**Coding style (done 2026-04-27):**
- T1: imports and set_seed already clean — no changes needed.
- T3: removed duplicate `import random` from imports cell; version check added.
- T1/T3: version check cell added after vibecheck.
- T1/T3: no matplotlib usage — no figure settings needed.


**⚠ External dependency on personal repo:** T1 pulls game code from `github.com/raymondchua/nma_rl_games` (Raymond Chua, one of the lecturers). The `git clone` is currently commented out and replaced with an OSF mirror download, but the OSF link should be verified and the dependency on a personal repo flagged for long-term hosting (e.g., fork under NeuromatchAcademy org).

**What DL Thinking 3 actually covers (T2):**
Five forward-looking discussion prompts:
1. **The Future** — distribution shift, non-stationarity, designing DL systems for a changing world
2. **In-context Learning** — meta-learning, transformers as implicit gradient descent
3. **Memories** — episodic vs. semantic memory analogues in DL
4. **Multiple Information Sources** — multimodal fusion, language as a query interface (PaLM-E)
5. **Language for Robotics** — LLMs for instruction-following and task planning (RT-1, Code-As-Policies)

This is more "Frontiers in DL" than a case study — it's a forward-looking survey of where the field is heading, not a practitioner design exercise.

---

## Naming: "DL Thinking" Series

The term "Thinking" is now ambiguous given LLM chain-of-thought/reasoning modes. Proposed renames:

| Current | Content | Suggested Rename |
|---------|---------|-----------------|
| W2D2 "DL Thinking 1" (T2) | Designing cost functions for real problems | **DL Case Study 1** |
| W3D2 "DL Thinking 2" | Designing architectures; multimodal strategies | **DL Case Study 2** |
| W3D5 "DL Thinking 3" (T2) | Forward-looking: ICL, memory, multimodal, LLM robotics | **Frontiers in DL** (or drop if W3D5 is moved to bonus) |

"Case Study" fits T1 and T2 well — they follow a structured vignette → discussion → design format. T3 is more of a survey/horizon-scan, so "Frontiers" is a better fit if it's kept.

Folder rename implications:
- ~~`W2D2_ConvnetsAndDlThinking`~~ → **`W2D2_Convnets` (done, 2026-04-05)**
- `W3D2_DlThinking2` → `W3D2_CaseStudy2` or `W3D2_ArchitectureAndMultimodalDL`
- `W3D5_ReinforcementLearningForGamesAndDlThinking3` → `W3D5_RLForGames` (bonus; DL Thinking 3 dropped or extracted)

---

## Coding Style: Top Issues to Fix (Priority Order)

| Issue | Affected Days | Fix | Status |
|-------|--------------|-----|--------|
| Missing `nma.mplstyle` | ~~W3D1 T1~~, ~~W3D4~~, W3D5 T1/T3 | Add standard figure settings cell | W3D4 done 2026-04-27; W3D1 T1 already had it; W3D5 T1/T3 have no matplotlib |
| `matplotlib.pylab` vs `.pyplot` | W2D4 T1 | Change to `import matplotlib.pyplot as plt` | Open (W2D4 on hold) |
| `import torch` missing from top-level setup | ~~W3D1 T1~~ | Add to setup cell | Done 2026-04-27 |
| Duplicate/stray imports inside `set_seed` | ~~W3D1 T1/T3~~, ~~W3D3~~ | Remove from set_seed body; ensure in top-level imports | Done 2026-04-27 |
| Duplicate `import random` in imports cell | ~~W3D5 T3~~ | Remove duplicate | Done 2026-04-27 |
| Missing version check cell | ~~W3D1~~, ~~W3D3~~, ~~W3D4~~, ~~W3D5~~ | Add after vibecheck | Done 2026-04-27 |
| Type hints inconsistency | W3D4 only | Decide course policy; likely remove for consistency | Open |
| Student TODO marker inconsistency | All days | Standardize to `# YOUR CODE HERE` + `raise NotImplementedError` | Open |
| All helpers defined inline (copy-paste across 20+ notebooks) | All days | Long-term: extract common utilities (`set_seed`, `set_device`, plotting) into a shared module | Open (long-term) |
| Capitalization of "Tutorial Objectives" header | Several | Standardize in a sweep | Open |

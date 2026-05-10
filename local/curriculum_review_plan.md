[Original DL_NMA coarse design](https://docs.google.com/spreadsheets/d/1tPS8eCBRlw5iVqS0zw6CPf7cBmsscjPNmM7SDq7lXTM/edit?usp=sharing) 
Week 1: The basics	
D1	Basics, Meet and Greet, Pytorch
D2	Linear DL and Pytorch
D3	MLPs 
D4	Optimization
D5	Regularization/ Generalization
	
Week 2: Doing more with fewer parameters	
D1	Parameter sharing: Convnets and RNNs 
D2	Convnets/ Transfer learning
D3	RNNs
D4	Attention/Transformers
D5	VAEs and GANs
	
Week 3: Higher doses of magic	
D1	Unsupervised and self-supervised learning
D2	Basic RL ideas
D3	RL for games
D4	Continual learning/ Causality/ The future of DL
D5	Wrapup

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
N.B. Final check: make the Youtube Playlists.
Also make sure the exercises variable naming is intuitive and matches with equations.

| Day | Current Name | Tutorials | Recommended Action | Priority |
|-----|-------------|-----------|-------------------|----------|
| ~~W1D1~~ | ~~Basics And PyTorch~~ | ~~T1~~ | ~~Keep; JAX notebook retained as future option (not a current priority)~~ | ~~Low~~ ✅ |
| ~~W1D2~~ | ~~Linear Deep Learning~~ | ~~T1, T2, T3 + Bonus~~ | ~~Coding style standardized; T3 kept as-is (already bonus-marked)~~ | ~~Medium~~ ✅ |
| ~~W1D3~~ | ~~Multi-Layer Perceptrons~~ | ~~T1, T2~~ | ~~Coding style standardized; LIF neuron section already in Bonus~~ | ~~Low~~ ✅ |
| ~~W1D4~~ | ~~Optimization~~ | ~~T1 (very long)~~ | ~~Coding exercises in Secs 4–7 marked optional~~ | ~~High~~ ✅ |
| **W1D5** | **Project Wildcard Day (Flex)** | — | **Students study one curriculum day relevant to their project track; see flex day section below** | Make sure the project points to the correct tutorial name! |
| ~~W2D1~~ | ~~Regularization~~ | ~~T1, T2~~ | ~~Fine as-is~~ | — ✅ |
| ~~W2D2~~ | ~~ConvNets~~ | ~~T1, BonusLecture~~ | ~~Folder renamed to W2D2_Convnets; T1 trimmed; Transfer Learning added as Section 5; BonusLecture merged from W2D3 T1+T2 (modern ConvNets + facial recognition + ethics)~~ | ~~High~~ ✅ |
| ~~W2D3~~ | ~~DL Discussion 1 + VAE~~ | ~~T1 (VAE), T2 (DL Discussion 1), BonusLecture (Geoffrey Hinton)~~ | ~~Repurposed: DL Discussion 1 moved here as T2; VAE tutorial moved here as T1 (Section 5 Bonus); folder recreated as W2D3\_GenerativeModelsAndDeepLearningDiscussion1~~ | ~~High~~ ✅ |
| ~~W2D4~~ | ~~Diffusion Models only~~ | ~~T1 (score-based), T2 (image diffusion)~~ | ~~VAE tutorial dropped; Section 2 (Conditional Diffusion) marked Bonus in T2; T1 math trimmed and exercises clarified (2026-05-03/04); SyntaxWarnings, np.sqrt, utils cell, SD model, and image display fixes (2026-05-09)~~ | ~~High~~ ✅ |
| ~~W2D5~~ | ~~Attention And Transformers~~ | ~~T1, T2~~ | ~~Switched with W3D1 — W2D5 is now NLP/TimeSeries, W3D1 is now Attention/Transformers (done 2026-05-02)~~ | ✅ |
| ~~W3D1~~ | ~~Time Series And NLP~~ | ~~T1, T2, T3~~ | ~~Switched with W2D5 — W3D1 is now Attention/Transformers, W2D5 is now NLP/TimeSeries (done 2026-05-02)~~ | ✅ |
| ~~W3D2~~ | ~~DlThinking2~~ | ~~T1~~ | ~~Rename~~ | ✅ |
| ~~W3D3~~ | ~~Unsupervised And Self-Supervised~~ | ~~Separate utils import from OSF with outdated numpy. Migrate datasets, checkpoints and images dependency to new OSF storage.~~ | ~~Medium~~ ✅ |
| W3D4 | Basic Reinforcement Learning | T1 | Add REINFORCE/policy gradient; condense value/policy iteration | High |
| ~~W3D5~~ | ~~RL for Games And DlThinking3~~ | ~~T1, T2, T3~~ | ~~RL for Games (T1, T3) removed; only DL Discussion 3 (T2) kept~~ | ✅ |

---

## Per-Day Details

### ~~W1D1 — Basics And PyTorch~~ ✅ (2026-03-29)

**Done:**
- Added version check / install cell to Setup section; prints package versions, points local users to root `requirements.txt`
- Consolidated all imports to top-level Setup section; removed scattered imports and duplicate pip installs
- Added Colab GPU conservation tips callout in Section 2.4; highlighted that Setup cells must be rerun after runtime restart
- Added bold **Appendix** references with Colab jump links in three cells
- Updated Meet Our Lecturers: flat alphabetical list of all content creators with websites; added Past contributors section
- **(2026-05-01) Bonus section restored inline:** The altair/paper visualization bonus (60 years of ML research) was initially extracted to `W1D1_BonusLecture.ipynb` but has been merged back into `W1D1_Tutorial1` before the Appendix following feedback. `altair`/`vega_datasets` added back to install and imports cells. `W1D1_BonusLecture.ipynb` deleted.

**Still open:**
- `W1D1_Tutorial1_JAX.ipynb` — mark as bonus or remove (deferred, JAX out of scope for now)
- Meet Our Lecturers section should list the people who actually appear in the tutorial videos, not just content creators.

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
- **(2026-05-01) Tutorial1 split reverted:** After review and feedback, the earlier split of Tutorial1 into two notebooks and the addition of Tutorial2 were undone. `W1D4_Tutorial1` is restored as the single unified original (Secs 1–8 + Bonus, 145 cells). `W1D4_Tutorial1_Bonus.ipynb`, `W1D4_BonusLecture.ipynb`, and `W1D4_Tutorial2.ipynb` all deleted. Only two changes kept on top of the upstream original: the import/version check cell (with correct GitHub URL for `requirements.txt`) and the `*(optional)*` markers on exercises 4, 6, and 7.

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

### ~~W2D3 — DL Discussion 1 + VAE (repurposed)~~ ✅ (2026-05-09)

Previously "Modern ConvNets" — folder was removed (2026-04-05), content absorbed into W2D2. Repurposed with new content.

**Done:**
- Folder recreated as `W2D3_GenerativeModelsAndDeepLearningDiscussion1`.
- **DL Discussion 1** (Cost Functions; formerly W2D2 "DL Thinking 1") moved here as T2.
- **VAE tutorial** (formerly W2D4 T1) moved here as T1; Section 5 marked Bonus.
- BonusLecture (Geoffrey Hinton) retained from old W2D4.

**W2D3 Tutorial 1 edits (2026-05-06):**
- **Section 5** (State of the art VAEs and Wrap-up) marked **Bonus** — advanced VAE variants; not required.
- **Section 4** (VAE basics) kept as core content — mainly conceptual and provides grounding for understanding diffusion models and other generative models.

**W2D3 fixes (2026-05-09):**
- Fixed `feedback_prefix` strings: were incorrectly set to `"W2D4_T1"` / `"W2D4_BonusLecture"`; corrected to `"W2D3_T1"` / `"W2D3_BonusLecture"`.


### ~~W2D4 — Diffusion Models only~~ ✅ (2026-05-09)

Previously "Generative Models" (T1 VAE + T2 score-based + T3 image diffusion). Now scoped to diffusion only.

**Done:**
- VAE tutorial dropped (moved to W2D3).
- **T1 (Score-based diffusion):** Math trimmed and exercises clarified (2026-05-03/04) — see edits below.
- **T2 (Image diffusion, U-Net, Stable Diffusion, conditional diffusion):** Section 2 (Conditional Diffusion) marked Bonus.
- `import matplotlib.pylab as plt` coding style issue resolved — VAE tutorial moved to W2D3, which uses `.pyplot`; W2D4 T1 (diffusion) unaffected.

**W2D4 Tutorial 1 (diffusion) edits (2026-05-03/04):**
- `\mathcal` → `\mathscr` throughout T1 and T2 (main only; student/instructor folders untouched) — fixes Colab rendering.
- **Exercise 1.3** renamed to **Bonus Exercise 1.3 (Optional)**: `reverse_diffusion_SDE_sampling_gmm` is not called by any later cell, so safe to skip.
- Heavy math cells (DSM/ESM equivalence + γ_t weighted integral) merged into a single collapsible `<details>` note box above Exercise 2; students can expand for details but it's out of the way by default.
- Exercise 1.3 code cleaned up:
  - `x_traj_rev` shape reordered to `(nsteps, sampN, 2)` so `x_traj_rev[i]` is the snapshot at step `i`
  - `sampN` default changed 500 → 200 (was same as `nsteps=500`, making shapes ambiguous)
  - `eps_z` renamed to `z_t` to match equation notation
  - `np.random.randn(*xT.shape)` → `np.random.randn(sampN, 2)` with comment explaining `*` unpacking
  - Shape annotations added throughout explaining each variable's dimensions and that `2` is data dimensionality, not number of GMM components
- Exercise 2 (`loss_fn`) code cleaned up:
  - `x` → `x_0`, `perturbed_x` → `x_t` to match plain-language step descriptions
  - `std[:, None]` → `std.unsqueeze(1)` for clarity
  - Shape annotations added for all intermediate variables
  - Note added explaining `random_t` is continuous `t ∈ [eps, 1]`, unlike the discrete `nsteps` index in reverse diffusion

**W2D4 Tutorial 1 & 2 fixes (2026-05-09):**
- Fixed all invalid escape sequences (`\l`, `\s`, `\sigma`, etc.) in docstrings and plot title strings — Python 3.12+ SyntaxWarning; fixed with `r"""..."""` prefix or `\\` escaping.
- Replaced all `np.sqrt()` calls on torch tensors with `** 0.5` throughout T1 — NumPy 2.0 DeprecationWarning from `__array_wrap__`.
- Restored missing "Define utils functions" cell in T1 (`GaussianFourierProjection`, `ScoreModel_Time`, `sample_X_and_score_t_depend`) — was accidentally removed.
- Swapped gated SD model (`stabilityai/stable-diffusion-2-1`) to open model (`stable-diffusion-v1-5/stable-diffusion-v1-5`) with `token=False` to suppress HF token prompt on Colab.
- Fixed SD inference image not displaying in T2: added explicit `display(image)` call inside `if execute:` block (Jupyter only auto-displays top-level last expressions).
- Swept all W1D* and W2D* notebooks for SyntaxWarnings and `np.sqrt` on torch tensors — no issues found outside W2D4.

---

### ~~W3D2 — DL Discussion 2~~ ✅ (~2026-03)

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
**Add REINFORCE.**
- Per meeting: ground in policy learning / REINFORCE first, then Q-learning.
- **REINFORCE/policy gradient:** Completely absent — needs to be added (new section or new tutorial).
- **Value iteration and policy iteration:** Good conceptual foundations but could be condensed into one section with fewer coding exercises.

**Coding style (done 2026-04-27):**
- `nma.mplstyle` figure settings cell added after imports cell.
- Version check cell added after vibecheck.

- **Only notebook with type hints** (`from typing import Optional, Tuple`) — decide on course-wide policy (see Style section).

---

### ~~W3D5 — RL for Games And DL Discussion 3~~ ✅

**Done:**
- T1 (Othello RL) and T3 (MCTS) removed entirely — game-specific and highly specialized.
- T2 (DL Discussion 3) kept as the sole content for this day.

**Coding style (done 2026-04-27):**
- T1: imports and set_seed already clean — no changes needed.
- T3: removed duplicate `import random` from imports cell; version check added.
- T1/T3: version check cell added after vibecheck.
- T1/T3: no matplotlib usage — no figure settings needed.


**⚠ External dependency on personal repo:** T1 pulls game code from `github.com/raymondchua/nma_rl_games` (Raymond Chua, one of the lecturers). The `git clone` is currently commented out and replaced with an OSF mirror download, but the OSF link should be verified and the dependency on a personal repo flagged for long-term hosting (e.g., fork under NeuromatchAcademy org).

**What DL Discussion 3 actually covers (T2):**
Five forward-looking discussion prompts:
1. **The Future** — distribution shift, non-stationarity, designing DL systems for a changing world
2. **In-context Learning** — meta-learning, transformers as implicit gradient descent
3. **Memories** — episodic vs. semantic memory analogues in DL
4. **Multiple Information Sources** — multimodal fusion, language as a query interface (PaLM-E)
5. **Language for Robotics** — LLMs for instruction-following and task planning (RT-1, Code-As-Policies)

This is more "Frontiers in DL" than a case study — it's a forward-looking survey of where the field is heading, not a practitioner design exercise.

---

## Naming: "DL Thinking" → "DL Discussion" Series

The term "Thinking" was ambiguous given LLM chain-of-thought/reasoning modes. Renamed to **"DL Discussion"**.

| Old Name | New Name | Status |
|---------|---------|--------|
| ~~W2D2 "DL Thinking 1" (T2)~~ | ~~**DL Discussion 1**~~ | ✅ Done — moved to W2D3 as T2 (2026-05-09) |
| W3D2 "DL Thinking 2" | **DL Discussion 2** | ✅ Done (2026-05-02) |
| W3D5 "DL Thinking 3" (T2) | **DL Discussion 3** | ✅ Done (RL for Games removed; only T2 kept) |

Folder rename implications:
- ~~`W2D2_ConvnetsAndDlThinking`~~ → **`W2D2_Convnets` (done, 2026-04-05)**
- ~~`W3D2_DlThinking2`~~ → **`W3D2_DeepLearningDiscussion2` (done, 2026-05-02)**
- ~~`W3D5_ReinforcementLearningForGamesAndDlThinking3`~~ → **`W3D5_DeepLearningDiscussion3` (done)**

---

## Coding Style: Top Issues to Fix (Priority Order)

| Issue | Affected Days | Fix | Status |
|-------|--------------|-----|--------|
| Missing `nma.mplstyle` | ~~W3D1 T1~~, ~~W3D4~~, W3D5 T1/T3 | Add standard figure settings cell | W3D4 done 2026-04-27; W3D1 T1 already had it; W3D5 T1/T3 have no matplotlib |
| ~~`matplotlib.pylab` vs `.pyplot`~~ | ~~W2D4 T1~~ | ~~Change to `import matplotlib.pyplot as plt`~~ | ✅ Resolved — VAE tutorial (which had this issue) moved to W2D3; W2D3 T1 uses `.pyplot` |
| `import torch` missing from top-level setup | ~~W3D1 T1~~ | Add to setup cell | Done 2026-04-27 |
| Duplicate/stray imports inside `set_seed` | ~~W3D1 T1/T3~~, ~~W3D3~~ | Remove from set_seed body; ensure in top-level imports | Done 2026-04-27 |
| Duplicate `import random` in imports cell | ~~W3D5 T3~~ | Remove duplicate | Done 2026-04-27 |
| Missing version check cell | ~~W3D1~~, ~~W3D3~~, ~~W3D4~~, ~~W3D5~~ | Add after vibecheck | Done 2026-04-27 |
| Type hints inconsistency | W3D4 only | Decide course policy; likely remove for consistency | Open |
| Student TODO marker inconsistency | All days | Standardize to `# YOUR CODE HERE` + `raise NotImplementedError` | Open |
| All helpers defined inline (copy-paste across 20+ notebooks) | All days | Long-term: extract common utilities (`set_seed`, `set_device`, plotting) into a shared module | Open (long-term) |
| Capitalization of "Tutorial Objectives" header | Several | Standardize in a sweep | Open |

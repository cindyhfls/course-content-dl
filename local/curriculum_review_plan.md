# NMA DL Curriculum Review Plan

> Goals from expert review: reduce and simplify, cut outdated/dense material, standardize code style.
> JAX refactorization is deferred.

---

## Summary Table

| Day | Current Name | Tutorials | Recommended Action | Priority |
|-----|-------------|-----------|-------------------|----------|
| ~~W1D1~~ | ~~Basics And PyTorch~~ | ~~T1~~ | ~~Keep; JAX notebook retained as future option (not a current priority)~~ | ~~Low~~ ✅ |
| ~~W1D2~~ | ~~Linear Deep Learning~~ | ~~T1, T2, T3 + Bonus~~ | ~~Coding style standardized; T3 kept as-is (already bonus-marked)~~ | ~~Medium~~ ✅ |
| W1D3 | Multi-Layer Perceptrons | T1, T2 | Keep; trim LIF neuron section (T1) | Low |
| W1D4 | Optimization | T1 (very long) | Simplify; cut overparameterization (Sec 5); move bonus training section out; add Cost Functions discussion (moved from W2D2) | High |
| **W1D5** | **Project Wildcard Day (Flex)** | — | **Students study one curriculum day relevant to their project track; see flex day section below** | — |
| W2D1 | Regularization | T1, T2 | Cut back; merge into one notebook; move double descent/rethinking generalization to bonus | High |
| W2D2 | ConvNets | T1 | Rename; restructure as sole CNN day: trim existing content, add condensed transfer learning + ResNet from W2D3 as demos | Medium |
| W2D3 | Modern ConvNets | T1, T2 | Move entirely to bonus; T2 (facial recognition + ethics) may survive as standalone ethics bonus notebook | High |
| W2D4 | Generative Models | T1, T2, T3 | Drop T1 (VAEs); keep T2+T3 (diffusion); optionally rescue BigGAN demo as intro | High |
| W2D5 | Attention And Transformers | T1, T2 | Keep T1; make T2 an official bonus notebook | Medium |
| W3D1 | Time Series And NLP | T1, T2, T3 | Add RNN/LSTM tutorial (currently absent!); slim T2 (overlaps W2D5) | High |
| W3D2 | DlThinking2 | T1 | Rename; review Forrest Gump case study for broad relevance | Medium |
| W3D3 | Unsupervised And Self-Supervised | T1 (very long) | Compress Secs 1–4 to demos; cut VAE section; keep SimCLR; consider adding Barlow Twins | High |
| W3D4 | Basic Reinforcement Learning | T1 | Add REINFORCE/policy gradient; condense value/policy iteration; add PyTorch | High |
| W3D5 | RL for Games And DlThinking3 | T1, T2, T3 | Move entirely to bonus; extract DL Thinking 3 if keeping the series | High |

---

## Per-Day Details

### ~~W1D1 — Basics And PyTorch~~ ✅ (2026-03-29)

**Done:**
- Added version check / install cell to Setup section; prints package versions, points local users to `requirements_tutorials.txt`
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

### W1D3 — Multi-Layer Perceptrons
**Coding style updated (2026-03-29):** Added version check / install cell and consolidated imports to Setup section across T1, T2.

**Keep, minor trims.**
- **T1, LIF neuron section (Sec 3):** Good neuroscience context but tangential to practical DL — move to bonus or mark clearly as optional.
- **T2, Xavier init section:** Overlaps with W1D2 initialization discussion — consider condensing.

---

### W1D4 — Optimization
**Coding style updated (2026-03-29):** Added version check / install cell and consolidated imports to Setup section. Removed duplicate `ipywidgets` import from Figure Settings cell.

**Simplify — single notebook is ~3700 lines.**
- **Section 5 (overparameterization):** Move to bonus or replace with a brief conceptual discussion — not practically actionable for most students.
- The standalone training bonus already exists; the main notebook body should end cleanly after optimizer comparison (Adam/RMSprop/SGD).
- Consider whether LR scheduling content here and in W2D1 T2 Sec 4 can be consolidated to one place.

---

### W2D1 — Regularization
**Cut back significantly (per meeting).**
- **Merge T1 and T2 into a single notebook.** The two-tutorial structure is unnecessarily heavy for one thematic topic.
- **T2, Sec 5 (rethinking generalization / double descent):** Move to bonus — research-level, not actionable.
- **T2, Sec 4 (LR scheduling):** Consolidate with W1D4 optimizer section, or keep as a brief demo rather than a full coding exercise.
- Both tutorials use a custom `AnimalFaces` dataset requiring a download — verify the download link is still live and consider caching in a more stable location.

---

### W2D2 — ConvNets And [DL Thinking 1]
**Simplify T1; rename the day.**
- **T1, Bonus section:** Remove — it's a full training loop + regularization exercise that directly repeats W2D1 content.
- **T1, Sec 4 (interactive parameter count demo):** Shorten.
- **T2 (DL Thinking 1 — cost functions):** Keep as-is; no code, purely discussion.
- **Rename:** Drop "DlThinking" from folder and day title. See naming section below.

---

### W2D3 — Modern ConvNets
**Move entirely to bonus (per meeting: single CNN day).**
- W2D2 becomes the sole CNN day.
- If any content is worth extracting: T2 (facial recognition + ethics) is the most distinctive and could survive as a standalone ethics case study bonus notebook.
- T1 (AlexNet/ResNet/transfer learning) is a good reference but redundant if students aren't doing a second CNN day.

---

### W2D4 — Generative Models
> **⚠ Hold: do not apply coding style / dependency standardization yet.** Full day content may be recreated from scratch — standardizing the current notebooks would be wasted effort.

**Drop T1; keep T2 and T3 (per meeting).**
- **T1 (VAEs, autoencoders, pPCA, BigGAN):** Drop. The BigGAN interactive demo in Sec 1.1 is motivating — consider moving just that cell as a brief intro to T2.
- **T2 (Score-based / diffusion):** Keep.
- **T3 (Image diffusion, U-Net, Stable Diffusion, conditional diffusion):** Keep.

**Coding style issues:**
- T1 uses `import matplotlib.pylab as plt` — the only notebook in the course to do so. Change to `matplotlib.pyplot`.

---

### W2D5 — Attention And Transformers
> **📝 Note:** W2D5 and W3D1 are likely to swap order ("LSTM then transformers"). Coding style patches can proceed; just remember to update any cross-references between the two days after the swap.

**Keep T1; demote T2.**
- **T1:** Well-organized (9 sections, cleanest notebook in the course). Could trim Sec 9 (Transformers beyond LMs) to a brief overview.
- **T2:** Already mostly labeled as bonus. Make it an official bonus notebook.

---

### W3D1 — Time Series And NLP
> **📝 Note:** See W2D5 note above — these two days are likely to swap. Coding style patches can proceed.

**High priority — RNN/LSTM is completely absent.**
- The day is called "Time Series" but no `nn.RNN` or `nn.LSTM` is ever instantiated anywhere in the course. Per meeting: "LSTM then transformers."
- **Action needed:** Add a new tutorial (or substantial section) covering RNN/LSTM for sequence modeling. Could replace or absorb T1 (embeddings + feedforward net).
- **T2 (NLP pipeline, BERT, GPT fine-tuning):** Overlaps heavily with W2D5 transformers day — condense or move LLM fine-tuning parts to bonus.
- **T3 Bonus (multilingual embeddings):** Keep as-is.

**Coding style issues:**
- T1 does not import `torch` at the top level — only imports it mid-notebook. Fix setup cell.
- T1 missing `nma.mplstyle` — add to figure settings cell.

---

### W3D2 — [DL Thinking 2]
**Rename; review one case study for relevance.**
- T1 is entirely discussion-based (no code).
- **Sec 4 (Forrest Gump / brain imaging):** Neuroscience-specific; may not resonate with non-neuro students — review whether to keep or replace with a more general case study.
- **Rename:** See naming section below.

---

### W3D3 — Unsupervised And Self-Supervised Learning
**Simplify — single notebook is ~4600 lines.**
- **Secs 1–4 (supervised encoder, random projections, VAE baselines):** These are comparative motivations. Compress to pre-computed demos or a brief narrative — students don't need to code all of these baselines.
- **VAE section (Sec 4):** Remove or shorten significantly since VAEs are being cut from W2D4.
- **SimCLR (Sec 6):** Keep as the core coding exercise.
- **Barlow Twins:** Not currently present — consider adding as an alternative/bonus per meeting suggestion (pedagogically easier than SimCLR).

**Coding style issues:**
- `import torch` appears twice in the notebook — remove the duplicate.
- Heavy use of OSF download URLs for pretrained models — fragile; verify links and consider a more stable hosting solution.

---

### W3D4 — Basic Reinforcement Learning
**Add REINFORCE; add PyTorch.**
- Currently uses only `numpy` — no PyTorch. The only Week 3 tutorial without it.
- Per meeting: ground in policy learning / REINFORCE first, then Q-learning.
- **REINFORCE/policy gradient:** Completely absent — needs to be added (new section or new tutorial).
- **Value iteration and policy iteration:** Good conceptual foundations but could be condensed into one section with fewer coding exercises.

**Coding style issues:**
- No `nma.mplstyle` — add to figure settings cell.
- No `vibecheck` feedback gadget — add if standardizing.
- **Only notebook with type hints** (`from typing import Optional, Tuple`) — decide on course-wide policy (see Style section).

---

### W3D5 — RL for Games And [DL Thinking 3]
**Move entirely to bonus (per meeting).**
- T1 (Othello RL) and T3 (MCTS) are game-specific and highly specialized.
- T2 (DL Thinking 3) could be extracted if the case study series is being kept — or dropped.
- No `nma.mplstyle`, no `vibecheck` in T1 or T3.

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
- `W2D2_ConvnetsAndDlThinking` → `W2D2_ConvnetsAndCostFunctions` (more specific) or `W2D2_ConvnetsAndCaseStudy1`
- `W3D2_DlThinking2` → `W3D2_CaseStudy2` or `W3D2_ArchitectureAndMultimodalDL`
- `W3D5_ReinforcementLearningForGamesAndDlThinking3` → `W3D5_RLForGames` (bonus; DL Thinking 3 dropped or extracted)

---

## Coding Style: Top Issues to Fix (Priority Order)

| Issue | Affected Days | Fix |
|-------|--------------|-----|
| Missing `nma.mplstyle` | W3D1 T1, W3D4, W3D5 T1/T3 | Add standard figure settings cell |
| Missing `vibecheck` | W3D4, W3D5 T1/T3 | Add or remove from all (decide policy) |
| `matplotlib.pylab` vs `.pyplot` | W2D4 T1 | Change to `import matplotlib.pyplot as plt` |
| `ipywidgets` double-import | W1D1, others | Single top-level import only |
| `import torch` missing from top-level setup | W3D1 T1 | Add to setup cell |
| Duplicate `import torch` mid-notebook | W3D3 | Remove duplicate |
| Type hints inconsistency | W3D4 only | Decide course policy; likely remove for consistency |
| Student TODO marker inconsistency | All days | Standardize to `# YOUR CODE HERE` + `raise NotImplementedError` |
| All helpers defined inline (copy-paste across 20+ notebooks) | All days | Long-term: extract common utilities (`set_seed`, `set_device`, plotting) into a shared module |
| Capitalization of "Tutorial Objectives" header | Several | Standardize in a sweep |

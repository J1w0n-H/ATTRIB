# LLM-Augmented Vulnerability Attribution: Bridging the Semantic Gap in Open-Source Dependency Analysis

## Abstract

Automated vulnerability attribution in open-source software—determining whether a bug's root cause lies in the main project or an external dependency—faces a fundamental paradox: under severe class imbalance, a naive baseline achieves high agreement (86.65%) but yields **0.00% Dependency recall**, providing no supply-chain value. This work uses LLM semantic reasoning to bridge three gaps: intent understanding (workarounds vs. genuine fixes), pattern discovery, and cross-case validation.

On **517 UUV cases (59 projects)**, the system achieves **86.65% agreement** while recovering **59.42% Dependency recall at 50.00% precision**—vs. **0% recall** for a structural baseline. The core contribution is recovering minority-class signal under severe imbalance via **strategic error reallocation** (trading 41 Main for 41 Dependency cases), enabling triage-assisted supply chain workflows where missing upstream bugs is costlier than generating review candidates. The pipeline forms **125 interpretable sub-groups** with **70.40%** perfect type matching. **Because evaluation labels are heuristic rather than expert-adjudicated, all metrics measure agreement with structural signals rather than absolute correctness.**

**Keywords:** Vulnerability attribution, Dependency bugs, LLM reasoning, Supply chain security, OSS-Fuzz

---

## 1. Introduction

### 1.1 The Attribution Challenge

The presence of dependency code in a crash stack trace does not directly indicate who should fix the bug. Main projects frequently ship defensive patches that work around dependency defects, while crashes in main project code can be triggered by malformed data produced by dependencies. Accurate attribution requires semantic reasoning about patch intent, responsibility boundaries, and ownership relationships—capabilities that simple structural heuristics fundamentally cannot provide.

### 1.2 Related Work: Patch Localization vs Responsibility Attribution

Prior work on automated patch localization addresses a complementary but distinct problem. The ARVO system (Automated Repair Validation and Optimization) identifies which commit fixed a bug through patch testing, achieving approximately **84% accuracy** (best estimate) and **56%** in complex multi-commit cases. However, patch localization answers **"where was it fixed?"** rather than **"who should fix it?"**—a distinction critical for supply chain security.

| Aspect | ARVO (Patch Localization) | This Work (Attribution) |
|--------|---------------------------|-------------------------|
| **Research Question** | "Which commit fixed the bug?" | "Whose bug is it?" |
| **Method** | Patch testing + execution validation | Semantic reasoning about intent |
| **Output** | Fix commit hash + files | Main/Dependency + reasoning |
| **Success Criterion** | Crash stops after patch | Correct responsibility assignment |
| **Use Case** | Automated patch backporting | CVE disclosure targeting |

**Key distinction**: ARVO's own analysis highlights cases where patch localization succeeds but attribution requires semantic reasoning. In Qt5 Issue 35566, ARVO identifies a submodule update that stops the crash, but the update is a **workaround** (updating qtbase dependency) rather than a fix to Qt5's own code—requiring intent understanding to determine responsibility. This work uses ARVO's infrastructure for data extraction but addresses a different question: not "which commit fixed it?" but **"whose bug is it?"**

### 1.3 Empirical Motivation: The Dependency Paradox

Analysis of 6,046 OSS-Fuzz cases reveals severe class imbalance:

| Category | Count | % |
|----------|-------|---|
| Total cases | 6,046 | 100% |
| Dependency-involved | 5,169 | 85.49% |
| Dependency-labeled | 189 | 3.13% |
| Main-labeled | 5,857 | 96.87% |

This disparity between dependency involvement (85.49%) and dependency responsibility (3.13%) creates severe class imbalance and highlights why simple presence-based heuristics fail.

**The Class Imbalance Problem**: In the 517-case UUV evaluation slice, heuristic labels contain **69 Dependency vs 448 Main** cases. Under such imbalance, a naive "predict Main for everything" strategy achieves high agreement (448/517 = 86.65%) while providing zero utility for supply chain security—missing every dependency-labeled case (0/69 Dependency recall) that requires upstream attention. This paradox motivates my focus on **Dependency recall** (detecting the minority class) even when it redistributes errors across classes rather than increasing raw agreement with heuristic labels.

### 1.4 Motivating Example: The Workaround Problem

A representative mismatch case in ImageMagick illustrates the attribution challenge. In one UUV report (e.g., localId 432073014 in the stored inference artifacts), the crash stack trace points into `/src/libjxl/...`, suggesting a fault in the libjxl dependency, while the associated fix patch is applied to a main-project file (e.g., `imagemagick/coders/sf3.c`). A patch-location heuristic would treat this as a main-project bug; however, semantic attribution must consider that a main-project patch may be a defensive mitigation for a dependency-rooted defect. This gap—distinguishing workarounds/mitigations from genuine root fixes—cannot be bridged by structural pattern matching alone.

### 1.5 Research Contributions

This work makes three primary contributions:

1. **Minority-class recovery under severe imbalance**: LLM semantic reasoning achieves **59.42% Dependency recall** (41/69) on a 517-case UUV slice where a structural baseline yields **0% recall**.
2. **Interpretable pattern discovery through hierarchical architecture**: The four-stage pipeline forms **125 sub-groups** achieving **70.40%** perfect type matching, enabling cross-project validation.
3. **Systematic error characterization with concrete mitigation paths**: Error analysis identifies Dependency precision (**50.00%**) as the primary barrier and motivates three prioritized improvements: boundary disambiguation, asymmetric thresholds, and multi-bug-type validation.

---

## 2. Problem Formulation

### 2.1 Task Definition

Given an OSS-Fuzz vulnerability report containing stack traces, patch diffs, commit metadata, and dependency context, the attribution system must predict: (1) root cause type (Main_Project_Specific vs Dependency_Specific), and (2) specific dependency name when classifying as Dependency_Specific. The system must provide confidence scores and explanatory reasoning to enable human verification.

### 2.2 Three Fundamental Challenges

**Semantic Ambiguity.** Crash location does not equal responsibility. A crash in dependency code may result from main project misuse (incorrect API calls, invalid input), while crashes in main code may be triggered by dependency defects (malformed data, violated contracts).

**Workaround Patches.** Developers cannot modify third-party dependency code directly, forcing them to implement defensive measures in main project code. These workarounds create systematic misalignment between patch location and bug ownership—patches appear in the main project while root causes lie in dependencies.

**Cross-Project Pattern and Boundary Ambiguity.** Bundled or embedded dependencies create ambiguous module boundaries. Paths like `/src/<dep>/...` may represent external dependencies or internal submodules depending on build configuration, while internal modules (e.g., ImageMagick's `magickcore`, `coders`) can be mistaken for external dependencies when only module names are available. Individual cases often present weak or conflicting signals; cross-project pattern discovery and sub-group validation provide additional context to strengthen or weaken dependency attribution hypotheses.

---

## 3. Dataset

### 3.1 Ground Truth Construction

Ground Truth is constructed using **heuristic rules** over structural signals extracted from 6,046 successfully processed OSS-Fuzz cases. These signals are obtained using ARVO's extraction infrastructure (arvo.db database), which provides:

- Stack traces with file paths and function names
- Patch diffs with modified files and code changes  
- Commit metadata including messages and timestamps
- Dependency context from srcmap.json files
- Submodule update flags (arvo.submodule_bug field)

**Relationship to ARVO**: ARVO addresses patch localization (identifying which commit fixed a bug), while my Ground Truth heuristic addresses attribution (determining who should fix it). ARVO's infrastructure provides the data source (stack traces, patches, metadata), and ARVO's `submodule_bug` field is one structural signal used in Rule 3. However, ARVO does not perform attribution; its `submodule_bug` field simply indicates whether a fix commit updated a submodule—a structural observation, not a responsibility judgment.

#### 3.1.1 Structural Rules (Heuristic GT)

My heuristic Ground Truth uses four rules implemented in `02_build_ground_truth.py`, operating over ARVO-extracted structural signals. The rules contribute evidence and scores rather than performing semantic intent understanding.

**Rule 1: Patch File Path Mapping / Exclusivity**  
Uses patched file paths to determine whether changes occur exclusively in a dependency path, exclusively in main-project paths, or are mixed, producing a discrete score.

**Rule 2: Stack Trace Dominance**  
Computes Crash Ownership Confidence (COC) from stack frames and can amplify its weight based on patch–crash distance to reflect cases where patch location and crash location diverge.

**Rule 3: Dependency Update Commit**  
Flags dependency updates based on ARVO database metadata (e.g., `submodule_bug` and repository address mismatch signals).

**Rule 4: External CVE/NVD Connection (currently inactive)**  
This rule is present as a placeholder in the implementation, but the current code does not perform external CVE/NVD lookups; therefore, it does not contribute evidence in the reported results.

The heuristic label is derived from the aggregated evidence across rules; cases without sufficient dependency evidence default to Main_Project_Specific.

#### 3.1.2 Severe Class Imbalance

The resulting distribution reveals severe imbalance: **96.87% Main vs 3.13% Dependency** (5,857 vs 189 cases). The 5,857 Main cases include 877 with no dependencies in metadata and 4,980 that involve dependencies but exhibit main project responsibility by structural signals.

This distribution spans 307 projects and 116 bug types, with an average of 3.74 dependencies per case.

#### 3.1.3 Limitations of Heuristic Labels

These labels should be interpreted as a **scalable proxy** rather than authoritative truth. Heuristic labels have two fundamental limitations:

1. **Signal vs Intent**: Structural signals (patch location, stack frames) cannot distinguish workarounds from genuine fixes—requiring semantic understanding.

2. **Coverage gaps**: Boundaries (internal vs external modules) and dependency updates (manifest changes, vendoring) are not fully captured by current rules.

This distribution reflects **structural signal availability** rather than true vulnerability ownership. Accordingly, **all quantitative results in Sections 5–7 measure agreement with structural signals, not absolute correctness**. Expert adjudication of a representative sample would be required to validate heuristic accuracy.

> **Evaluation caveat**: All quantitative results (Sections 5–7) measure agreement with heuristic labels, not absolute correctness. Heuristic labels are derived from structural signals (patch location, stack frames, submodule flags) and may systematically misclassify workaround patches as main-project bugs. Expert adjudication is required to validate label accuracy.

### 3.2 Evaluation Set Selection

I evaluate the LLM pipeline on a single bug-type slice: **Use-of-uninitialized-value (UUV)**. I chose UUV because it is the most frequent crash type in ARVO DB among reproduced cases (e.g., 1,183 reproduced cases in `/root/Git/arvo.db`) and the most frequent bug type in my heuristic Ground Truth corpus (521/6,046). This focus provides a large, consistent slice for controlled measurement.

For the results reported in this paper, I evaluate on **517 UUV cases** successfully processed by my pipeline, spanning **59 projects**. The corresponding heuristic Ground Truth label distribution is **448 Main_Project_Specific** and **69 Dependency_Specific**, reflecting severe class imbalance.

**Why UUV is a challenging setting for dependency attribution.** UUV is not a favorable setting for dependency attribution because: (1) on this slice, the ARVO `submodule_bug` flag is **0 for all evaluated cases**, meaning submodule-update signals (Rule 3) provide no evidence; (2) UUV fixes often appear as defensive patches or inline changes rather than dependency version bumps, making patch-location heuristics less reliable; and (3) many UUV cases involve logic errors that require semantic intent analysis to distinguish workarounds from genuine fixes. This makes the evaluation a stress test for semantic reasoning: any Dependency recall must come from evidence beyond structural signals alone (e.g., patch/stack alignment, workaround intent, and cross-case consistency).

---

## 4. Methodology: Hierarchical LLM-Augmented Attribution

### 4.1 Architecture Overview

The system implements a four-stage hierarchical pipeline that selectively applies LLM reasoning where structural signals prove insufficient. Stage 0 (Preprocessing) extracts and summarizes evidence through LLM-based feature extraction. Stage 1 (Individual Inference) produces per-case attributions using bug-type-aware prompting. Stage 2 (Sub-Grouping) discovers fine-grained patterns through dependency-based clustering, deterministic structural grouping, and semantic similarity analysis. Stage 3 (Cross-Project Validation) synthesizes individual inferences with group hypotheses, validates cross-project consistency, and produces final attributions with confidence scores.

This design addresses the three fundamental challenges identified in Section 2.2. Semantic ambiguity is addressed by Stage 1’s evidence-based attribution. Workaround intent is addressed by Stage 0’s patch summarization and Stage 1’s intent-aware reasoning. Cross-project pattern and boundary ambiguity is addressed by Stage 2 sub-grouping and Stage 3 cross-case validation, where ambiguous individual signals are reconciled using sub-group context.

### 4.2 Pipeline Stages

| Stage | Purpose | Key Operation | Addresses Challenge (§2.2) |
|-------|---------|---------------|---------------------------|
| 0 | Feature Extraction | LLM summarizes patches/stacks, extracts structural signals | Workaround Patches (intent) |
| 1 | Individual Inference | Per-case attribution + confidence with bug-type-aware prompts | Semantic Ambiguity |
| 2 | Sub-Grouping | Cluster by dependency/similarity (structural + semantic) | Cross-Project Pattern and Boundary Ambiguity |
| 3 | Cross-Validation | Consolidate with group consensus, validate cross-project patterns | Cross-Project Pattern and Boundary Ambiguity |

Stage 0 converts raw OSS-Fuzz artifacts into structured inputs, preserving "what happened" (crash context) and "what changed" (patch intent). Stage 1 produces per-case attributions that serve as inputs to later sub-grouping. Stage 2 discovers fine-grained patterns by clustering related cases. Stage 3 synthesizes Stage 1 predictions with Stage 2 sub-group structure to produce final attributions and confidence, using sub-group agreement as a consistency signal to reduce single-signal decisions.

---

## 5. Results

### 5.1 Overall Performance

**All results in this section measure agreement with heuristic Ground Truth labels** described in Section 3.1, not absolute correctness. On the 517-case UUV slice, the final pipeline (Stages 1–3) forms **125 sub-groups** and achieves:

- **Type agreement**: **86.65%** (448/517)
- **Dependency name agreement**: **84.72%** (438/517)
- **Both correct**: **84.72%** (438/517)

Because most cases are Main-labeled and correctly expect a null dependency, dependency-name agreement is dominated by Main cases. Conditioned on Dependency-labeled cases, exact dependency-name agreement is **42.03% (29/69)**; conditioned on true-positive Dependency cases (type correct), exact name match is **70.73% (29/41)**.

**Interpretation (dependency names).** Name prediction is reliable conditional on correct type classification: when the model correctly predicts Dependency (41 cases), it matches the heuristic dependency name in 29 cases (70.73%). The lower 42.03% (29/69) rate is dominated by the 28 missed Dependency-labeled cases (heuristic false negatives) where no dependency is predicted.

**How to interpret the metrics.** Higher values are better for all reported metrics, but they answer different questions:

- **Type agreement**: whether the predicted label (Main vs Dependency) matches the heuristic label.
- **Dependency precision/recall**: quality of Dependency attributions (minority class). Precision measures false-positive burden; recall measures missed dependency cases.
- **Balanced type agreement**: mean of Main recall and Dependency recall, reducing majority-class dominance.
- **Sub-group perfect matching**: whether a sub-group’s members are all correct under the heuristic labels (a proxy for cross-case consistency, not expert correctness).

#### 5.1.1 Ablation: Stage Contributions (Module 1/2/3)

The intended design is that Stage 3 *validates and consolidates* Stage 1 hypotheses using cross-project sub-group structure. To test whether later stages actually improve on early predictions, I evaluate the following ablations on the same 517-case slice.

| Setting | Output used for scoring | Type agreement | Balanced type agreement | Dependency Precision | Dependency Recall |
|---|---|---:|---:|---:|---:|
| ARVO baseline | `Dependency_Specific` iff `arvo.submodule_bug == 1` | 86.65% | 50.00% | 0.00% | 0.00% |
| Stage 1 only (Module 1) | per-case `individual_root_causes` | 85.49% | 78.14% | 47.00% | 68.12% |
| Stage 2 only (Module 2) | sub-group `inferred_root_cause_type` | 85.49% | 73.24% | 46.43% | 56.52% |
| Stages 1–3 (final) | group-level `root_cause_inferences` | 86.65% | 75.13% | 50.00% | 59.42% |

For comparability, I use conservative normalizations where needed (e.g., a single missing Stage 1 prediction is mapped to Main; Module 2 “Unknown” outputs are mapped to Main). Overall, Stage 3 increases Dependency precision relative to Stage 1 (47.00% → 50.00%) and slightly improves overall agreement, at the cost of some Dependency recall (68.12% → 59.42%). This is consistent with Stage 3 acting as a cross-project validation layer rather than a purely recall-maximizing classifier.

This “validation-layer” behavior is also visible in raw counts (TP/FP computed under heuristic GT). On the same 517 cases, Stage 1 predicts Dependency 99 times (TP=47, FP=53), while Stage 3 predicts Dependency 82 times (TP=41, FP=41). Thus Stage 3 reduces Dependency predictions by 17 and reduces heuristic false positives by 12, while also dropping 6 true positives—consistent with a conservative consolidation step that trades recall for precision.

#### 5.1.2 Predicted Distribution

The final system's predicted type distribution is:
- **435 Main_Project_Specific (84.14%)**
- **82 Dependency_Specific (15.86%)**

Compared to heuristic Ground Truth:
- **448 Main_Project_Specific (86.65%)**
- **69 Dependency_Specific (13.35%)**

Thus, the final system slightly over-predicts Dependency (82 vs 69), but not by an order of magnitude. The remaining difficulty is that half of Dependency predictions are heuristic false positives (Section 5.2).

#### 5.1.3 ARVO Baseline and the Core Trade-Off

On this UUV slice, the ARVO baseline (predict Dependency iff `submodule_bug=1`) predicts Main for all 517 cases (no submodule updates in UUV):
- **Agreement**: **86.65%**, **Dependency recall**: **0%**

Our system achieves the same agreement but recovers **59.42% Dependency recall** by trading 41 Main cases for 41 Dependency cases. This shows why agreement alone is misleading under class imbalance: the baseline provides zero supply-chain value despite high agreement.

#### 5.1.4 Confidence Score Distribution (Stage 3)

Stage 3 outputs a group-level `confidence_score` per sub-group; I assign this score to all cases in the sub-group. Confidence is not calibrated against expert correctness, but it provides a consistency signal for thresholding. On the 517-case slice:

- **Main predictions**: mean confidence **0.904**; high-confidence (≥0.8) **97.7%** (425/435)
- **Dependency predictions**: mean confidence **0.897**; high-confidence (≥0.8) **100.0%** (82/82)

Notably, Dependency predictions remain uniformly high-confidence despite only **50.00%** precision under heuristic labels, indicating the need for asymmetric thresholds and calibration (Section 6.3.1, Priority 2).

### 5.2 Per-Class Performance Analysis

**Confusion Matrix**:
```
                    Predicted Main    Predicted Dep    Total
Actual Main             407               41          448
Actual Dependency        28               41           69
Total                   435               82          517
```

**Key Observations**:
- **Trade-off is explicit**: the model converts 41 Main-labeled cases into Dependency predictions (41 heuristic false positives) to recover 41 Dependency-labeled cases (41 true positives).
- **Dependency misses remain**: 28/69 Dependency-labeled cases (40.58%) are still missed (heuristic false negatives).
- **Predicted distribution is close to GT**: predicted Dependency is 82 vs GT-Dependency 69 (~1.19×), but precision remains a practical concern.

**Main_Project_Specific**:
- Precision: **93.56%** (407/435)
- Recall: **90.85%** (407/448)
- F1: **92.19%**

**Interpretation**: When the system predicts Main, it is correct most of the time, and Main recall remains high despite converting some Main-labeled cases into Dependency predictions.

**Dependency_Specific**:
- Precision: **50.00%** (41/82)
- Recall: **59.42%** (41/69)
- F1: **54.30%**

**Interpretation**: The system achieves non-trivial Dependency recall, but every second Dependency attribution is still incorrect under heuristic labels, motivating precision-oriented improvements.

### 5.3 Sub-Group Level Analysis

Sub-group performance reflects pattern discovery and the extent to which cross-case context produces consistent final decisions. Across **125 sub-groups**:

- **Perfect matching (type)**: **70.40%** (88/125)
- **Perfect matching (dependency name)**: **69.60%** (87/125)
- **Representative matching**: **88.00%** (110/125)
- **Partial matching (average)**: **86.61%** (type), **84.60%** (dependency)

These aggregate metrics suggest that sub-group structure provides a meaningful context signal; however, as emphasized in Section 3.1.3, perfect within-sub-group agreement does not imply correctness under expert adjudication, because heuristic labels themselves may contain correlated errors.

**Pilot vs full-slice comparison (52 → 517).** Because Stage 3 operates over sub-group structure, a larger slice can provide denser cross-project evidence. In the 52-case pilot (`llm_inference_results_partial.json`, 14 sub-groups), Stage 3 achieved **30.43%** Dependency precision (7/23) and **64.29%** perfect sub-group type matching (9/14). On the 517-case slice (125 sub-groups), Dependency precision improves to **50.00%** (41/82) and perfect sub-group type matching improves to **70.40%** (88/125), while predicted Dependency volume also becomes closer to the heuristic GT distribution (82 predicted vs 69 GT-Dependency). This shift is consistent with Stage 3 acting as a cross-project validation layer whose behavior stabilizes as repeated patterns accumulate, although the same scaling also exposes the need for stronger boundary evidence to improve precision further.

### 5.4 Error Analysis: Dependency Over-Prediction

On the 517-case slice, the final system produces **41 heuristic false positives** (Ground Truth Main → predicted Dependency) and **28 heuristic false negatives** (Ground Truth Dependency → predicted Main). The resulting Dependency precision is **50.00% (41/82)**, which remains a deployment barrier for fully automated reporting. Throughout this paper, “false positive/negative” denotes disagreement with heuristic labels, not absolute correctness.

This pattern indicates a systematic failure mode: deep dependency-heavy stack traces and boundary ambiguity can cause over-attribution to the component that appears in the crash path. Because expert adjudication is unavailable and causal tags are not first-class metrics, I report only the supported error counts and treat root-cause explanations as hypotheses.

**Evidence from failure distributions.** Among the 41 heuristic false positives, the most frequent predicted dependencies are concentrated rather than diffuse (e.g., `libjpeg-turbo` (12), `zlib` (5), `libraw` (4), and several "image-codec" aggregated labels). Among the 28 heuristic false negatives, missed GT-Dependency labels cluster in a few dependencies (e.g., `wolf-ssl-ssh-fuzzers` (11), `qtbase` (5), `hdf5-1.12.0` (4)). This concentration motivates targeted boundary/ownership disambiguation and naming constraints rather than generic prompt tuning. See Section 5.5 for a high-confidence disagreement analysis with a representative case.

### 5.5 High-Confidence Disagreement Analysis

Among **41 cases** where heuristic GT predicts Main but the LLM predicts Dependency (heuristic false positives), **20 cases (48.8%)** exhibit strong reasoning indicators: confidence ≥ 0.8, explicit workaround detection, and `WORKAROUND` patch-intent classification.

These cases represent two possibilities: (1) LLM heuristic false positives (overconfident errors) or (2) heuristic GT errors (corrected by semantic reasoning). Without expert adjudication, these scenarios cannot be definitively distinguished. However, the concentration of workaround-detection patterns suggests that semantic reasoning captures intent signals unavailable to structural heuristics. For example, localId 42539628 (also discussed in Section 5.4) crashes in `libjpeg-turbo` but is patched only by disabling a SIMD code path in a main-project fuzzer utility—an archetypal workaround pattern.

**Implication**: The true Dependency precision may be higher than 50.00%, but this remains a hypothesis requiring expert validation.

---

## 6. Discussion

### 6.1 Strengths of the Hierarchical Design

The hierarchical architecture uses Stage 1 to generate per-case hypotheses, Stage 2 to form interpretable sub-groups, and Stage 3 to consolidate and validate hypotheses using cross-case evidence. The ablation in Section 5.1.1 supports this intent: compared to Stage 1, Stage 3 improves Dependency precision and slightly improves agreement, while reducing Dependency recall, consistent with a conservative validation layer.

### 6.2 Limitations and Failure Modes

Three limitations constrain the current results. First, the evaluation covers only a single bug type (UUV); broader generalization requires multi-bug-type validation. Second, heuristic Ground Truth can be systematically biased because labels are generated by structural rules rather than expert review. Third, Dependency precision remains limited (**50.00%**, 41/82), which is a barrier for fully automated disclosure workflows.

The dependency over-prediction failure mode requires deeper analysis. Examination of false positive cases suggests three contributing factors: (1) stack trace overweighting, where deep dependency paths receive excessive attribution weight despite patches indicating main project responsibility; (2) boundary confusion, where internal modules (ImageMagick's magickcore) are mistaken for external dependencies due to path structure; and (3) pattern overgeneralization, where detecting one true libjxl defect leads to over-attributing additional crashes to libjxl even when evidence is ambiguous.

### 6.3 Implications for Future Work and Integration

Sections 5.1–5.4 show complementary strengths and systematic failure modes; I outline three directions to improve Dependency precision beyond **50.00%** while preserving useful recall:

#### 6.3.1 Three Directions (Ordered by Near-Term Practicality)

**Priority 1 (near-term impact): Explicit boundary disambiguation** (Expected Impact: scenario-dependent)

The **41 false positive cases** (Section 5.4) suggest that boundary ambiguity is a major contributor to over-prediction. Future work should incorporate:

- **Build system metadata** (CMakeLists.txt, Makefile) to distinguish internal vs external modules
- **Dependency manifest parsing** (package.json, Cargo.toml) to identify declared dependencies
- **Ownership heuristics** (same git repository = internal, different repository = external)

**Impact**: Removing 50% of heuristic false positives would increase Dependency precision to ~66% (production-viable threshold).

This explicit modeling targets heuristic false positives without requiring a reduction in minority-class recall.

**Priority 2 (immediately applicable): Asymmetric confidence thresholds** (Expected Impact: scenario-dependent)

The current system applies symmetric classification thresholds, treating Main and Dependency predictions equally. However, supply chain security priorities suggest asymmetric thresholds:

- **For Dependency classification**: Require **stronger evidence** (e.g., confidence > 0.8)
  - Rationale: Heuristic false positives burden external maintainers who don't control the main project
- **For Main classification**: Accept **moderate confidence** (e.g., > 0.5)
  - Rationale: Heuristic false negatives can be caught by dependency maintainers reporting "not my bug"

**Why this complements Priority 1**: Even with boundary disambiguation, some cases will remain ambiguous. Asymmetric thresholds ensure I err on the side of caution for Dependency classification, reducing heuristic false positives while preserving useful Dependency recall.

This approach would preserve high Dependency recall (critical for not missing supply chain issues) while reducing precision losses from over-prediction.

**Priority 3 (required for generalization): Multi-bug-type evaluation** (Purpose: validate generalization)

Current results apply only to Use-of-uninitialized-value cases (517 samples). Broader validation requires:

- Expanding to 500+ cases across 5-10 bug types (Heap-buffer-overflow, Null-dereference, etc.)
- Testing whether failure modes (stack trace overweighting) generalize or remain type-specific
- Measuring performance on bug types where submodule updates are common (→ tests Rule 3 effectiveness)

**Why This Is Essential**: Current results may be artifacts of bug type selection. The UUV slice contains zero submodule updates (Section 5.1.5), meaning Rule 3 effectiveness remains untested. Expanding to bug types with higher submodule update rates will reveal whether the observed precision/recall profile is intrinsic to semantic reasoning or specific to UUV.

#### 6.3.2 Future Integration

This work addresses attribution (who should fix?) while ARVO addresses localization (which commit fixed it?). Integration would enable an end-to-end pipeline: ARVO identifies the fix commit → this work determines responsibility → route to appropriate maintainers. **Requirements**: multi-bug-type validation, boundary disambiguation (Priority 1), Dependency precision >70%. Partial integration enables: workaround flagging for human review, sub-group pattern discovery, or CVE retrospective analysis.

### 6.4 When Does Semantic Reasoning Provide Value?

The results suggest three settings where semantic reasoning is valuable, and one where it fails:

- **Workaround/mitigation ambiguity**: patch location can be misleading when the main project mitigates a dependency-rooted crash; validating this requires expert adjudication beyond the current proxy.
- **Cross-project structure**: sub-group matching (Section 5.3) indicates that cross-case context provides a useful consistency signal.
- **Conflicting evidence**: multi-evidence synthesis maintains high Main precision (93.56%) under ambiguous signals.
- **Failure mode**: heuristic false positives concentrate when ownership/boundary evidence is missing (Section 5.4).

**Design Implication**: The optimal system combines both approaches—structural signals for boundary disambiguation, semantic reasoning for intent understanding. Neither alone is sufficient.

---

## 7. Threats to Validity

### 7.1 Internal Validity

Ground Truth accuracy represents the primary internal validity threat. Evaluation labels are generated through heuristic rules rather than expert human annotation, meaning absolute correctness is not guaranteed. Section 5.5 reports high-confidence disagreements as a hypothesis requiring validation, not as evidence of superior LLM correctness. Without expert review (e.g., multiple annotators with inter-rater agreement), correctness claims remain uncertain.

### 7.2 External Validity

**Critical limitation**: Results are constrained to OSS-Fuzz projects and a single bug type slice (517 cases, Use-of-uninitialized-value). Generalization to other bug types is unvalidated, and this UUV slice contains zero submodule updates (`arvo.submodule_bug==0` for all evaluated cases), meaning Rule 3 effectiveness is untested. Multi-bug-type evaluation is required before production deployment.

### 7.3 Construct Validity

Type agreement measures label consistency but does not capture attribution utility for human developers. A prediction with moderate confidence may be more valuable than a high-confidence prediction if the reasoning is more interpretable. Dependency-name metrics are also imperfect proxies because “dependency correctness” is dominated by Main cases where no dependency is expected. End-to-end utility evaluation requires user studies with security practitioners.

### 7.4 Reproducibility

LLM behavior variability poses reproducibility challenges. Model updates (e.g., `o4-mini` revisions) may alter inference results over time, and strict determinism cannot be guaranteed for all providers/models. To support transparent replication and comparative analysis, the implementation stores pipeline artifacts (raw LLM outputs, preprocessed summaries, sub-group assignments, and final synthesis outputs) so that future runs can be compared against the same intermediate evidence and outputs, even when exact token-level reproduction is not possible.

**Reproducibility**: All results derive from stored artifacts in `paper_pipeline/`. Key files: `llm_inference_results.json` (517 cases), `evaluation_results.json` (paper metrics), and module checkpoints for ablation. Evaluation script: `04_evaluate_llm_inference.py` (see repository for commands).

One-step evaluation reproduction (paper metrics): `python3 04_evaluate_llm_inference.py --inference-file llm_inference_results.json --gt-file ground_truth.json --output evaluation_results.json`

---

## 8. Conclusion

This work demonstrates that LLM-augmented attribution provides minority-class value under severe class imbalance. On 517 UUV cases, the system achieves **86.65% agreement** while recovering **59.42% Dependency recall**—vs. **0%** for a structural baseline with the same agreement. This highlights that agreement alone is inadequate for supply-chain attribution.

The engineering challenge: Dependency precision is **50.00%**, making automated disclosure premature. This system is a **triage signal generator**—redistributing errors to recover **59.42% Dependency recall** while maintaining **86.65% agreement**. Trading 41 Main cases for 41 Dependency cases is valuable because missed dependency bugs delay upstream fixes.

Stage 3 behaves as intended—as a cross-project validation layer that slightly improves precision and agreement over Stage 1—suggesting that further improvements should focus on boundary disambiguation and calibrated thresholds rather than abandoning the hierarchical design.

Future work should expand beyond a single bug type, incorporate explicit boundary evidence (build metadata, manifests, repository ownership), and validate heuristic labels with expert adjudication. These steps are required to translate semantic reasoning from a research prototype into a reliable component for automated vulnerability attribution in supply chain security.

---

## References

[1] Munaiah et al. (2017). "Curating GitHub for engineered software projects", *Empirical Software Engineering*

[2] Feng et al. (2020). "CodeBERT: A pre-trained model for programming and natural languages", *EMNLP*

[3] Wang et al. (2021). "GraphCodeBERT: Pre-training code representations with data flow", *ICLR*

[4] Wei et al. (2022). "Chain-of-Thought prompting elicits reasoning in large language models", *NeurIPS*

[5] Zhong et al. (2022). "Automated program repair in the era of deep learning", *ICSE*

[6] Plate et al. (2015). "Impact assessment for vulnerabilities in open-source software libraries", *ICSME*

[7] OpenAI (2023). "GPT-4 Technical Report"

[8] ARVO: Automated Repair Validation and Optimization (citation to be added)

---

## Appendix A: Additional LLM-Only Evidence

This appendix addresses the question: **"Could rule-based heuristics + clustering achieve similar results without LLM semantic reasoning?"** The ablation results (Section 5.1.1) show that Stage 1 (individual LLM inference) achieves the highest Dependency recall (68.12%), while Stage 3 (cross-project validation) primarily consolidates predictions. This suggests that **LLM semantic judgment is essential at Stage 1** to generate the initial hypotheses that later stages refine.

**Additional representative case (localId 42492490)**: The crash occurs in HDF5 internals (`H5G_visit_cb`), and the patch modifies a main-project file. Both patch location and crash location provide conflicting signals, and commit messages are vague. Rule-based heuristics would need arbitrary tie-breaking, but the LLM (confidence 0.92) synthesizes multiple evidence sources to infer that the patch is a workaround (`patch_intent: WORKAROUND`), reasoning: "The crash occurs in H5G_visit_cb within HDF5 internals... The patch does not modify HDF5 code, suggesting a defensive workaround in the main project." The LLM's ability to reason about "patch intent" (defensive vs. corrective) by analyzing what code is changed versus where the crash occurs is not replicable with structural pattern matching alone.
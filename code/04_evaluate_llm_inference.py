#!/usr/bin/env python3
"""
GT-based Evaluation Script for LLM Inference Results

This script evaluates LLM inference results against heuristic Ground Truth (GT).
It calculates various metrics including accuracy, precision, recall, F1-score,
and dependency matching accuracy.
"""

import json
import argparse
import logging
import re
import sqlite3
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_ARVO_DB_PATH = os.environ.get("ARVO_DB_PATH") or str((Path(__file__).resolve().parents[1] / "arvo.db"))


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single evaluation"""
    total_cases: int = 0
    correct_type: int = 0  # Correct root cause type (Main_Project_Specific vs Dependency_Specific)
    correct_dependency: int = 0  # Correct dependency name match
    correct_both: int = 0  # Both type and dependency correct
    
    # Per-type metrics
    main_project_true_positives: int = 0
    main_project_false_positives: int = 0
    main_project_false_negatives: int = 0
    
    dependency_true_positives: int = 0
    dependency_false_positives: int = 0
    dependency_false_negatives: int = 0
    
    # Sub-Group level metrics
    sub_group_count: int = 0
    sub_group_correct_type: int = 0
    sub_group_correct_dependency: int = 0
    sub_group_correct_both: int = 0
    
    # Sub-Group level partial matching metrics
    sub_group_partial_type_accuracy_sum: float = 0.0
    sub_group_partial_dep_accuracy_sum: float = 0.0
    sub_group_representative_matches: int = 0
    
    # Dependency matching ratio analysis
    dependency_matching_ratio_avg: float = 0.0
    dependency_matching_count_total: int = 0
    
    # Beyond Heuristic Accuracy (BHA)
    bha_cases: int = 0
    bha_correct: int = 0
    
    # Workaround Detection Rate (WDR)
    wdr_phase1_detected: int = 0
    wdr_phase2_detected: int = 0
    wdr_ground_truth_workarounds: int = 0
    wdr_tp_phase1: int = 0
    wdr_fp_phase1: int = 0
    wdr_fn_phase1: int = 0
    wdr_tp_phase2: int = 0
    wdr_fp_phase2: int = 0
    wdr_fn_phase2: int = 0
    
    # Cross-Project Validation Rate (CPVR)
    cpvr_total_outliers: int = 0
    cpvr_corrected_outliers: int = 0
    cpvr_uncorrected_outliers: int = 0
    
    # Transitive Dependency Tracing
    transitive_total_chains: int = 0
    transitive_phase1_success: int = 0
    transitive_phase2_success: int = 0
    transitive_phase2_deeper: int = 0
    
    def calculate_metrics(self) -> Dict:
        """Calculate derived metrics"""
        accuracy_type = self.correct_type / self.total_cases if self.total_cases > 0 else 0.0
        accuracy_dependency = self.correct_dependency / self.total_cases if self.total_cases > 0 else 0.0
        accuracy_both = self.correct_both / self.total_cases if self.total_cases > 0 else 0.0
        
        # Precision, Recall, F1 for Main_Project_Specific
        main_precision = (self.main_project_true_positives / 
                         (self.main_project_true_positives + self.main_project_false_positives) 
                         if (self.main_project_true_positives + self.main_project_false_positives) > 0 else 0.0)
        main_recall = (self.main_project_true_positives / 
                      (self.main_project_true_positives + self.main_project_false_negatives) 
                      if (self.main_project_true_positives + self.main_project_false_negatives) > 0 else 0.0)
        main_f1 = (2 * main_precision * main_recall / (main_precision + main_recall) 
                  if (main_precision + main_recall) > 0 else 0.0)
        
        # Precision, Recall, F1 for Dependency_Specific
        dep_precision = (self.dependency_true_positives / 
                        (self.dependency_true_positives + self.dependency_false_positives) 
                        if (self.dependency_true_positives + self.dependency_false_positives) > 0 else 0.0)
        dep_recall = (self.dependency_true_positives / 
                     (self.dependency_true_positives + self.dependency_false_negatives) 
                     if (self.dependency_true_positives + self.dependency_false_negatives) > 0 else 0.0)
        dep_f1 = (2 * dep_precision * dep_recall / (dep_precision + dep_recall) 
                 if (dep_precision + dep_recall) > 0 else 0.0)
        
        # Sub-Group level accuracy (Perfect Matching)
        sub_group_accuracy_type = (self.sub_group_correct_type / self.sub_group_count 
                                  if self.sub_group_count > 0 else 0.0)
        sub_group_accuracy_dependency = (self.sub_group_correct_dependency / self.sub_group_count 
                                        if self.sub_group_count > 0 else 0.0)
        sub_group_accuracy_both = (self.sub_group_correct_both / self.sub_group_count 
                                   if self.sub_group_count > 0 else 0.0)
        
        # Sub-Group level partial matching accuracy
        sub_group_partial_type_accuracy = (self.sub_group_partial_type_accuracy_sum / self.sub_group_count 
                                          if self.sub_group_count > 0 else 0.0)
        sub_group_partial_dep_accuracy = (self.sub_group_partial_dep_accuracy_sum / self.sub_group_count 
                                         if self.sub_group_count > 0 else 0.0)
        sub_group_representative_accuracy = (self.sub_group_representative_matches / self.sub_group_count 
                                            if self.sub_group_count > 0 else 0.0)
        
        # Beyond Heuristic Accuracy (BHA)
        bha_accuracy = (self.bha_correct / self.bha_cases 
                       if self.bha_cases > 0 else 0.0)
        
        return {
            'accuracy_type': accuracy_type,
            'accuracy_dependency': accuracy_dependency,
            'accuracy_both': accuracy_both,
            'main_project_precision': main_precision,
            'main_project_recall': main_recall,
            'main_project_f1': main_f1,
            'dependency_precision': dep_precision,
            'dependency_recall': dep_recall,
            'dependency_f1': dep_f1,
            'sub_group_accuracy_type': sub_group_accuracy_type,
            'sub_group_accuracy_dependency': sub_group_accuracy_dependency,
            'sub_group_accuracy_both': sub_group_accuracy_both,
            'sub_group_partial_type_accuracy': sub_group_partial_type_accuracy,
            'sub_group_partial_dep_accuracy': sub_group_partial_dep_accuracy,
            'sub_group_representative_accuracy': sub_group_representative_accuracy,
            'dependency_matching_ratio_avg': self.dependency_matching_ratio_avg,
            'dependency_matching_count_total': self.dependency_matching_count_total,
            'bha_accuracy': bha_accuracy,
            'bha_cases': self.bha_cases,
            'bha_correct': self.bha_correct,
            # WDR metrics
            'wdr_phase1': (self.wdr_phase1_detected / self.wdr_ground_truth_workarounds * 100) if self.wdr_ground_truth_workarounds > 0 else 0.0,
            'wdr_phase2': (self.wdr_phase2_detected / self.wdr_ground_truth_workarounds * 100) if self.wdr_ground_truth_workarounds > 0 else 0.0,
            'wdr_phase1_precision': (self.wdr_tp_phase1 / (self.wdr_tp_phase1 + self.wdr_fp_phase1)) if (self.wdr_tp_phase1 + self.wdr_fp_phase1) > 0 else 0.0,
            'wdr_phase1_recall': (self.wdr_tp_phase1 / (self.wdr_tp_phase1 + self.wdr_fn_phase1)) if (self.wdr_tp_phase1 + self.wdr_fn_phase1) > 0 else 0.0,
            'wdr_phase2_precision': (self.wdr_tp_phase2 / (self.wdr_tp_phase2 + self.wdr_fp_phase2)) if (self.wdr_tp_phase2 + self.wdr_fp_phase2) > 0 else 0.0,
            'wdr_phase2_recall': (self.wdr_tp_phase2 / (self.wdr_tp_phase2 + self.wdr_fn_phase2)) if (self.wdr_tp_phase2 + self.wdr_fn_phase2) > 0 else 0.0,
            # CPVR metrics
            'cpvr': (self.cpvr_corrected_outliers / self.cpvr_total_outliers * 100) if self.cpvr_total_outliers > 0 else 0.0,
            # Transitive dependency metrics
            'transitive_phase1_rate': (self.transitive_phase1_success / self.transitive_total_chains * 100) if self.transitive_total_chains > 0 else 0.0,
            'transitive_phase2_rate': (self.transitive_phase2_success / self.transitive_total_chains * 100) if self.transitive_total_chains > 0 else 0.0
        }


def compute_arvo_submodule_baseline_metrics(
    detailed_results: List[Dict],
    db_path: str = DEFAULT_ARVO_DB_PATH,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Compute ARVO baseline metrics from ARVO DB.

    Baseline definition (ARVO-style): predict Dependency_Specific iff arvo.submodule_bug == 1,
    otherwise predict Main_Project_Specific.

    Metrics are computed against the same GT labels used in evaluation (detailed_results[*]['gt_type']).
    """
    local_ids = [r.get('localId') for r in detailed_results if r.get('localId') is not None]
    local_ids = [int(x) for x in local_ids]

    # Fetch submodule_bug flags from DB (chunked to avoid SQLite parameter limits)
    submodule_flag: Dict[int, bool] = {}
    try:
        conn = sqlite3.connect(db_path)
        for i in range(0, len(local_ids), 900):
            chunk = local_ids[i:i+900]
            placeholders = ",".join(["?"] * len(chunk))
            q = f"SELECT localId, submodule_bug FROM arvo WHERE localId IN ({placeholders})"
            for lid, sub in conn.execute(q, chunk):
                submodule_flag[int(lid)] = bool(sub) if sub is not None else False
        conn.close()
    except Exception as e:
        if logger:
            logger.warning(f"Failed to compute ARVO baseline from DB ({db_path}): {e}")
        return {
            "definition": "Dependency_Specific iff arvo.submodule_bug == 1 (ARVO DB-derived baseline)",
            "db_path": db_path,
            "available": False,
            "error": str(e),
        }

    missing = [lid for lid in local_ids if lid not in submodule_flag]
    if missing and logger:
        logger.warning(f"ARVO baseline: {len(missing)} localIds missing from DB; defaulting submodule_bug=False for them.")

    total = 0
    correct_type = 0
    main_tp = main_fp = main_fn = 0
    dep_tp = dep_fp = dep_fn = 0
    predicted_dependency = 0

    for r in detailed_results:
        lid = int(r["localId"])
        gt_type = r["gt_type"]
        pred_type = "Dependency_Specific" if submodule_flag.get(lid, False) else "Main_Project_Specific"
        total += 1
        if pred_type == gt_type:
            correct_type += 1
        if pred_type == "Dependency_Specific":
            predicted_dependency += 1

        if gt_type == "Main_Project_Specific":
            if pred_type == "Main_Project_Specific":
                main_tp += 1
            else:
                main_fn += 1
                dep_fp += 1
        else:
            if pred_type == "Dependency_Specific":
                dep_tp += 1
            else:
                dep_fn += 1
                main_fp += 1

    accuracy_type = correct_type / total if total else 0.0
    main_precision = main_tp / (main_tp + main_fp) if (main_tp + main_fp) else 0.0
    main_recall = main_tp / (main_tp + main_fn) if (main_tp + main_fn) else 0.0
    dep_precision = dep_tp / (dep_tp + dep_fp) if (dep_tp + dep_fp) else 0.0
    dep_recall = dep_tp / (dep_tp + dep_fn) if (dep_tp + dep_fn) else 0.0
    balanced_accuracy = (main_recall + dep_recall) / 2.0

    return {
        "definition": "Dependency_Specific iff arvo.submodule_bug == 1 (ARVO DB-derived baseline)",
        "db_path": db_path,
        "available": True,
        "total_cases": total,
        "predicted_dependency": predicted_dependency,
        "accuracy_type": accuracy_type,
        "balanced_accuracy_type": balanced_accuracy,
        "main_project_precision": main_precision,
        "main_project_recall": main_recall,
        "dependency_precision": dep_precision,
        "dependency_recall": dep_recall,
        "raw_counts": {
            "correct_type": correct_type,
            "main_project_tp": main_tp,
            "main_project_fp": main_fp,
            "main_project_fn": main_fn,
            "dependency_tp": dep_tp,
            "dependency_fp": dep_fp,
            "dependency_fn": dep_fn,
        },
    }


def normalize_dependency_name(name: str) -> str:
    """Normalize dependency name for comparison"""
    if not name:
        return ""
    # Convert to lowercase and remove common prefixes/suffixes
    name = name.lower().strip()
    # Remove version suffixes
    import re
    name = re.sub(r'[v\s]*\d+[.\d]*.*$', '', name)
    # Remove common prefixes
    name = re.sub(r'^(lib|libs|lib-|libs-)', '', name)
    return name.strip()


def compare_dependencies(llm_dep: Optional[str], gt_dep: Optional[Dict]) -> bool:
    """Compare LLM inferred dependency with GT dependency"""
    if not llm_dep or not gt_dep:
        return False
    
    # Extract name from GT dependency dict
    gt_name = gt_dep.get('name', '') if isinstance(gt_dep, dict) else str(gt_dep)
    
    if not gt_name:
        return False
    
    # Normalize both names
    llm_normalized = normalize_dependency_name(llm_dep)
    gt_normalized = normalize_dependency_name(gt_name)
    
    # Exact match or substring match
    return (llm_normalized == gt_normalized or 
            llm_normalized in gt_normalized or 
            gt_normalized in llm_normalized)


def parse_individual_inferences(reasoning_text: str) -> Dict[int, Dict[str, Optional[str]]]:
    """
    Parse individual case inference results from llm_reasoning_process text.
    
    Returns a dictionary mapping localId to {'type': str, 'dependency': Optional[str]}
    """
    individual_results = {}
    
    if not reasoning_text or "Individual inferences:" not in reasoning_text:
        return individual_results
    
    idx = reasoning_text.find("Individual inferences:")
    individual_part = reasoning_text[idx:]
    lines = individual_part.split('\n')[1:]  # Skip the header line
    
    for line in lines:
        line = line.strip()
        if not line or 'localId' not in line:
            continue
        
        # Match "localId 432073014: Dependency_Specific (libjxl)"
        match = re.match(r'localId\s+(\d+):\s+(\w+)\s+\(([^)]+)\)', line)
        if match:
            lid = int(match.group(1))
            inf_type = match.group(2)
            dep = match.group(3) if match.group(3) != 'N/A' else None
            individual_results[lid] = {
                'type': inf_type,
                'dependency': dep
            }
        else:
            # Match "localId 432073014: Main_Project_Specific" (without dependency)
            match2 = re.match(r'localId\s+(\d+):\s+(\w+)', line)
            if match2:
                lid = int(match2.group(1))
                inf_type = match2.group(2)
                individual_results[lid] = {
                    'type': inf_type,
                    'dependency': None
                }
    
    return individual_results


class WorkaroundDetector:
    """Detects workarounds using various heuristics"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def detect_phase1_workaround(self, gt_data: Dict) -> bool:
        """
        Phase 1 (Heuristic) workaround detection
        
        Conditions:
        1. patch_crash_distance >= 2
        2. module_mismatch = True
        3. control_flow_only = True
        """
        patch_crash_distance = gt_data.get('patch_crash_distance')
        module_mismatch = gt_data.get('module_mismatch', False)
        control_flow_only = gt_data.get('control_flow_only', False)
        
        if patch_crash_distance is None:
            return False
            
        return (patch_crash_distance >= 2 and 
                module_mismatch and 
                control_flow_only)
    
    def detect_phase2_workaround(self, feature_data: Dict, inference_data: Dict) -> bool:
        """
        Phase 2 (LLM) workaround detection
        
        Checks if LLM identified workaround in reasoning
        """
        # Check LLM reasoning for workaround keywords
        reasoning = inference_data.get('llm_reasoning_process', '') + \
                   inference_data.get('group_level_root_cause_type', '')
        
        workaround_keywords = [
            'workaround', 'bypass', 'defensive', 'mitigation',
            'does not fix', 'does not address', 'does not update',
            'patch does not', 'not actually fix', 'remains unpatched'
        ]
        
        reasoning_lower = reasoning.lower()
        for keyword in workaround_keywords:
            if keyword in reasoning_lower:
                return True
        
        # Check patch semantic type
        patch_semantic_type = feature_data.get('patch_semantic_type', '')
        if patch_semantic_type == 'VALIDATION_ONLY':
            # Check if patch is in main but crash is in dependency
            crash_module = feature_data.get('crash_module', '')
            patched_module = feature_data.get('patched_module', '')
            
            if (crash_module and patched_module and 
                crash_module != patched_module and
                'main' in patched_module.lower() and
                'lib' in crash_module.lower()):
                return True
        
        return False
    
    def detect_ground_truth_workaround(self, gt_data: Dict, feature_data: Dict, 
                                      commit_message: Optional[str] = None) -> bool:
        """
        Ground truth workaround detection using multiple signals
        
        This is used to create ground truth labels for evaluation
        """
        # Signal 1: Phase 1 structural detection
        phase1_detected = self.detect_phase1_workaround(gt_data)
        
        # Signal 2: Commit message keywords
        commit_workaround = False
        if commit_message:
            commit_lower = commit_message.lower()
            workaround_keywords = ['workaround', 'bypass', 'defensive', 'mitigation', 
                                  'temporary fix', 'quick fix']
            commit_workaround = any(kw in commit_lower for kw in workaround_keywords)
        
        # Signal 3: Patch doesn't fix dependency
        patch_crash_distance = gt_data.get('patch_crash_distance')
        module_mismatch = gt_data.get('module_mismatch', False)
        
        # Signal 4: Patch summary analysis
        patch_summary = feature_data.get('patch_summary', '')
        patch_lower = patch_summary.lower()
        
        # Check if patch explicitly doesn't fix dependency
        no_fix_keywords = [
            'does not fix', 'does not address', 'does not update',
            'not fix', 'remains unpatched', 'no change'
        ]
        patch_no_fix = any(kw in patch_lower for kw in no_fix_keywords)
        
        # Signal 5: Stack trace in dependency but patch in main
        crash_module = feature_data.get('crash_module', '')
        patched_module = feature_data.get('patched_module', '')
        
        module_mismatch_evidence = False
        if crash_module and patched_module:
            # Crash in dependency library
            dep_indicators = ['lib', 'src/', '/src/', 'vendor/']
            crash_in_dep = any(ind in crash_module.lower() for ind in dep_indicators)
            
            # Patch in main project
            main_indicators = ['main', 'coders/', 'magickcore/', 'project']
            patch_in_main = any(ind in patched_module.lower() for ind in main_indicators)
            
            module_mismatch_evidence = crash_in_dep and patch_in_main
        
        # Combine signals (at least 2 signals needed)
        # Signal 3: Patch doesn't fix dependency (check if patch_crash_distance is valid)
        signal3 = False
        if patch_crash_distance is not None:
            signal3 = (patch_crash_distance >= 2 and module_mismatch)
        
        signals = [
            phase1_detected,
            commit_workaround,
            signal3,
            patch_no_fix,
            module_mismatch_evidence
        ]
        
        signal_count = sum(1 for s in signals if s)  # Count only True values
        
        # High confidence: 3+ signals
        # Medium confidence: 2 signals
        # Low confidence: 1 signal (not considered workaround)
        return signal_count >= 2


def identify_outliers(inference_data: Dict, logger: Optional[logging.Logger] = None) -> List[Dict]:
    """
    Identify outliers in sub-groups
    
    Outlier: Case that disagrees with group majority (≥60%)
    """
    from collections import Counter
    
    outliers = []
    
    # Process each sub-group
    for inf in inference_data.get('root_cause_inferences', []):
        sub_group_id = inf.get('sub_group_id')
        local_ids = inf.get('localIds', [])
        
        if len(local_ids) < 2:
            continue  # Need at least 2 cases for outlier detection
        
        # Get individual inferences (Module 1)
        individual_types = {}
        features_map = {}
        
        for feature in inference_data.get('features', []):
            local_id = feature.get('localId')
            if local_id in local_ids:
                features_map[local_id] = feature
        
        # Try to extract individual inferences from Module 1 bug_type_groups
        # Check bug_type_groups for individual_root_causes
        for bug_group in inference_data.get('bug_type_groups', []):
            individual_root_causes = bug_group.get('individual_root_causes', {})
            if isinstance(individual_root_causes, dict):
                for lid_str, rc_data in individual_root_causes.items():
                    try:
                        local_id = int(lid_str) if isinstance(lid_str, str) else lid_str
                        if local_id in local_ids:
                            if isinstance(rc_data, dict):
                                individual_types[local_id] = rc_data.get('root_cause_type', '')
                            else:
                                # If it's an IndividualRootCause object (from dataclass)
                                individual_types[local_id] = getattr(rc_data, 'root_cause_type', '')
                    except (ValueError, TypeError):
                        continue
        
        # Fallback: Try to parse from reasoning text
        if not individual_types:
            reasoning = inf.get('llm_reasoning_process', '')
            parsed_individual = parse_individual_inferences(reasoning)
            for lid, parsed_data in parsed_individual.items():
                if lid in local_ids:
                    individual_types[lid] = parsed_data.get('type', '')
        
        # If still no individual inferences found, use group level as fallback
        if not individual_types:
            group_type = inf.get('group_level_root_cause_type', '')
            for local_id in local_ids:
                individual_types[local_id] = group_type
        
        # Count types in group
        type_counts = Counter(individual_types.values())
        total = len(individual_types)
        
        if total == 0:
            continue
        
        # Find majority type (≥60%)
        majority_type = None
        majority_count = 0
        
        for type_name, count in type_counts.items():
            if count / total >= 0.6:  # 60% threshold
                majority_type = type_name
                majority_count = count
                break
        
        if not majority_type:
            continue  # No clear majority
        
        # Identify outliers (disagree with majority)
        for local_id, individual_type in individual_types.items():
            if individual_type != majority_type:
                feature = features_map.get(local_id, {})
                
                outlier = {
                    'localId': local_id,
                    'sub_group_id': sub_group_id,
                    'project': feature.get('project_name', 'unknown'),
                    'bug_type': feature.get('bug_type', 'unknown'),
                    'individual_type': individual_type,
                    'group_majority_type': majority_type,
                    'group_size': total,
                    'final_type': inf.get('group_level_root_cause_type'),
                }
                
                # Check if corrected (final type matches majority)
                if outlier['final_type'] == majority_type:
                    outlier['corrected'] = True
                elif outlier['final_type'] == individual_type:
                    outlier['corrected'] = False
                else:
                    outlier['corrected'] = None  # Uncertain
                
                outliers.append(outlier)
    
    return outliers


def extract_dependency_chain(srcmap_data: Dict, 
                            crash_location: str,
                            logger: Optional[logging.Logger] = None) -> Tuple[List[str], int]:
    """
    Extract dependency chain from srcmap data
    
    Returns:
        (chain_path, depth) where chain_path is list of dependencies
    """
    logger = logger or logging.getLogger(__name__)
    
    if not srcmap_data:
        return [], 0
    
    # Parse crash location to find which dependency
    # Example: "src/libpng/src/png.c" -> libpng
    crash_dep = None
    if crash_location:
        # Extract dependency name from path
        path_parts = crash_location.split('/')
        for i, part in enumerate(path_parts):
            if part in ['src', 'vendor', 'third_party', 'deps', 'dependencies']:
                if i + 1 < len(path_parts):
                    crash_dep = path_parts[i + 1]
                    break
    
    # Build dependency tree from srcmap
    dependencies = srcmap_data.get('dependencies', [])
    if not dependencies:
        return [], 0
    
    # Create dependency graph
    dep_graph = {}
    dep_names = {}
    
    for dep in dependencies:
        dep_name = dep.get('name', '')
        dep_path = dep.get('path', '')
        dep_url = dep.get('url', '')
        
        if dep_name:
            dep_names[dep_path] = dep_name
            dep_graph[dep_name] = {
                'path': dep_path,
                'url': dep_url,
                'dependencies': []  # Would need to parse nested deps
            }
    
    # For now, return simple chain
    # In practice, would need to parse nested dependency structures
    if crash_dep:
        return [crash_dep], 1
    
    # Fallback: return all dependencies as flat list
    all_deps = [dep.get('name', '') for dep in dependencies if dep.get('name')]
    return all_deps, 1  # Simplified: assume 1-hop for now


def evaluate_llm_inference(
    inference_file: str,
    gt_file: str,
    extracted_data_file: Optional[str] = None,
    num_cases: Optional[int] = None,
    target_localIds: Optional[List[int]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[EvaluationMetrics, List[Dict]]:
    """
    Evaluate LLM inference results against Ground Truth
    
    Args:
        inference_file: Path to llm_inference_results.json
        gt_file: Path to ground_truth.json
        extracted_data_file: Optional path to extracted_data.json (for transitive dependency analysis)
        logger: Optional logger
    
    Returns:
        Tuple of (EvaluationMetrics, detailed_results)
    """
    # Load inference results
    if logger:
        logger.info(f"Loading inference results from: {inference_file}")
    with open(inference_file, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)
    
    # Load Ground Truth
    if logger:
        logger.info(f"Loading Ground Truth from: {gt_file}")
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # Load extracted data if available (for transitive dependency analysis)
    extracted_data = {}
    if extracted_data_file:
        try:
            if logger:
                logger.info(f"Loading extracted data from: {extracted_data_file}")
            with open(extracted_data_file, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
        except FileNotFoundError:
            if logger:
                logger.warning(f"Extracted data file not found: {extracted_data_file}")
    
    # Build GT lookup dictionary with normalization
    # Normalize GT: If project_name == dependency name, it's Main_Project_Specific
    # (Same normalization as in 03_llm_inference_modules.py for consistency)
    normalized_gt_list = []
    for gt in gt_data.get('ground_truth', []):
        normalized_gt = gt.copy()
        project_name = normalized_gt.get('project_name', '').lower().strip()
        dep_info = normalized_gt.get('Heuristically_Root_Cause_Dependency', {})
        
        # Extract dependency name
        if isinstance(dep_info, dict):
            dep_name = dep_info.get('name', '').lower().strip()
        elif isinstance(dep_info, str):
            dep_name = dep_info.lower().strip()
        else:
            dep_name = ''
        
        # If project_name == dependency name, normalize to Main_Project_Specific
        if project_name and dep_name and project_name == dep_name:
            normalized_gt['Heuristically_Root_Cause_Type'] = 'Main_Project_Specific'
            normalized_gt['Heuristically_Root_Cause_Dependency'] = None
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Normalized GT for localId {normalized_gt.get('localId')}: "
                           f"project_name ({project_name}) == dependency name, "
                           f"set to Main_Project_Specific with dependency=None")
        
        normalized_gt_list.append(normalized_gt)
    
    gt_dict = {}
    for gt_entry in normalized_gt_list:
        localId = gt_entry.get('localId')
        if localId:
            gt_dict[localId] = gt_entry
    
    if logger:
        logger.info(f"Loaded {len(gt_dict)} GT entries (normalized)")
    
    # Get root cause inferences
    inferences = inference_data.get('root_cause_inferences', [])
    if logger:
        logger.info(f"Found {len(inferences)} root cause inferences")
    
    # Build feature map for WDR and transitive dependency analysis
    feature_map = {}
    for feature in inference_data.get('features', []):
        local_id = feature.get('localId')
        if local_id:
            feature_map[local_id] = feature
    
    # Build inference map
    inference_map = {}
    all_inference_localIds = set()
    for inf in inferences:
        for local_id in inf.get('localIds', []):
            inference_map[local_id] = inf
            all_inference_localIds.add(local_id)
    
    # Check which inference localIds have GT
    missing_gt_localIds = all_inference_localIds - set(gt_dict.keys())
    if missing_gt_localIds:
        if logger:
            logger.warning(f"Found {len(missing_gt_localIds)} localIds in inference results without GT: {sorted(missing_gt_localIds)[:10]}{'...' if len(missing_gt_localIds) > 10 else ''}")
            logger.info(f"These cases will be skipped during evaluation. Only {len(all_inference_localIds) - len(missing_gt_localIds)}/{len(all_inference_localIds)} cases have GT and will be evaluated.")
        print(f"[!] Warning: {len(missing_gt_localIds)} localIds in inference results don't have GT and will be skipped")
        print(f"[+] Will evaluate {len(all_inference_localIds) - len(missing_gt_localIds)}/{len(all_inference_localIds)} cases that have GT")
    
    metrics = EvaluationMetrics()
    detailed_results = []
    
    # Track all evaluated localIds
    evaluated_localIds = set()
    
    # Convert target_localIds to set for fast lookup
    target_localIds_set = set(target_localIds) if target_localIds else None
    
    # Initialize workaround detector
    workaround_detector = WorkaroundDetector(logger)
    
    # Limit number of cases if specified
    if num_cases is not None and num_cases > 0:
        if logger:
            logger.info(f"Limiting evaluation to {num_cases} cases")
    if target_localIds_set:
        if logger:
            logger.info(f"Evaluating only specified localIds: {sorted(target_localIds_set)}")
    
    # Evaluate each Sub-Group inference
    for inference in inferences:
        # Check if we've reached the limit
        if num_cases is not None and num_cases > 0 and len(evaluated_localIds) >= num_cases:
            if logger:
                logger.info(f"Reached evaluation limit of {num_cases} cases, stopping")
            break
        sub_group_id = inference.get('sub_group_id', 0)
        localIds = inference.get('localIds', [])
        
        # Try to parse individual case inferences from llm_reasoning_process
        reasoning_text = inference.get('llm_reasoning_process', '')
        individual_inferences = parse_individual_inferences(reasoning_text)
        
        # Fallback to group-level inference if individual inferences are not available
        group_level_type = inference.get('group_level_root_cause_type', 'Unknown')
        group_level_dependency = inference.get('group_level_root_cause_dependency')
        
        dependency_matching_ratio = inference.get('dependency_matching_ratio', 0.0)
        dependency_matching_count = inference.get('dependency_matching_count', 0)
        
        metrics.sub_group_count += 1
        metrics.dependency_matching_count_total += dependency_matching_count
        
        # Evaluate each localId in the sub-group
        sub_group_correct_type = 0
        sub_group_correct_dependency = 0
        sub_group_correct_both = 0
        sub_group_evaluated_count = 0  # Count only cases with GT
        
        for localId in localIds:
            # Check if we've reached the limit
            if num_cases is not None and num_cases > 0 and len(evaluated_localIds) >= num_cases:
                break
            
            # Filter by target_localIds if specified
            if target_localIds_set is not None and localId not in target_localIds_set:
                continue
            
            if localId not in gt_dict:
                # Skip silently - we already logged a summary at the beginning
                continue
            
            evaluated_localIds.add(localId)
            metrics.total_cases += 1
            sub_group_evaluated_count += 1  # Count evaluated cases (with GT)
            
            gt_entry = gt_dict[localId]
            gt_type = gt_entry.get('Heuristically_Root_Cause_Type', 'Unknown')
            gt_dependency = gt_entry.get('Heuristically_Root_Cause_Dependency', {})
            
            # Use individual inference if available, otherwise fall back to group-level
            if localId in individual_inferences:
                llm_type = individual_inferences[localId]['type']
                llm_dependency = individual_inferences[localId]['dependency']
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Using individual inference for localId {localId}: {llm_type} ({llm_dependency or 'N/A'})")
            else:
                llm_type = group_level_type
                llm_dependency = group_level_dependency
                if logger and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Using group-level inference for localId {localId}: {llm_type} ({llm_dependency or 'N/A'})")
            
            # Compare type
            type_match = (llm_type == gt_type)
            if type_match:
                metrics.correct_type += 1
                sub_group_correct_type += 1
            
            # Compare dependency
            # Do not compare dependency for Main_Project_Specific (according to GT normalization)
            dep_match = False
            if llm_type == "Dependency_Specific" and gt_type == "Dependency_Specific":
                # Only compare dependency for Dependency_Specific cases
                dep_match = compare_dependencies(llm_dependency, gt_dependency)
            elif llm_type == "Main_Project_Specific" and gt_type == "Main_Project_Specific":
                # For Main_Project_Specific, consider match if both are N/A/None
                llm_dep_is_none = (llm_dependency is None or llm_dependency == 'N/A' or llm_dependency == '')
                gt_dep_is_none = (
                    gt_dependency is None or 
                    gt_dependency == 'N/A' or 
                    gt_dependency == '' or
                    (isinstance(gt_dependency, dict) and not gt_dependency.get('name'))
                )
                if llm_dep_is_none and gt_dep_is_none:
                    dep_match = True
            
            if dep_match:
                metrics.correct_dependency += 1
                sub_group_correct_dependency += 1
            
            # Both correct
            if type_match and dep_match:
                metrics.correct_both += 1
                sub_group_correct_both += 1
            
            # Update TP/FP/FN for Main_Project_Specific
            if gt_type == 'Main_Project_Specific':
                if llm_type == 'Main_Project_Specific':
                    metrics.main_project_true_positives += 1
                else:
                    metrics.main_project_false_negatives += 1
            else:
                if llm_type == 'Main_Project_Specific':
                    metrics.main_project_false_positives += 1
            
            # Update TP/FP/FN for Dependency_Specific
            if gt_type == 'Dependency_Specific':
                if llm_type == 'Dependency_Specific':
                    metrics.dependency_true_positives += 1
                else:
                    metrics.dependency_false_negatives += 1
            else:
                if llm_type == 'Dependency_Specific':
                    metrics.dependency_false_positives += 1
            
            # Store detailed result
            detailed_results.append({
                'localId': localId,
                'sub_group_id': sub_group_id,
                'llm_type': llm_type,
                'llm_dependency': llm_dependency,
                'gt_type': gt_type,
                'gt_dependency': gt_dependency.get('name', 'N/A') if isinstance(gt_dependency, dict) else str(gt_dependency),
                'type_match': type_match,
                'dependency_match': dep_match,
                'both_match': type_match and dep_match,
                'used_individual_inference': localId in individual_inferences
            })
        
        # Sub-Group level evaluation
        # Use sub_group_evaluated_count (only cases with GT) instead of len(localIds)
        if sub_group_evaluated_count > 0:
            sub_group_type_accuracy = sub_group_correct_type / sub_group_evaluated_count
            sub_group_dep_accuracy = sub_group_correct_dependency / sub_group_evaluated_count
            sub_group_both_accuracy = sub_group_correct_both / sub_group_evaluated_count
            
            # Perfect matching (existing method)
            if sub_group_type_accuracy == 1.0:
                metrics.sub_group_correct_type += 1
            if sub_group_dep_accuracy == 1.0:
                metrics.sub_group_correct_dependency += 1
            if sub_group_both_accuracy == 1.0:
                metrics.sub_group_correct_both += 1
            
            # Partial Matching: Accumulate match ratio within Sub-Group
            metrics.sub_group_partial_type_accuracy_sum += sub_group_type_accuracy
            metrics.sub_group_partial_dep_accuracy_sum += sub_group_dep_accuracy
            
            # Representative Matching: Compare LLM's group_level_root_cause with GT's most frequent value
            # Find most frequent root cause in GT
            gt_type_counts = defaultdict(int)
            gt_dep_counts = defaultdict(str)
            for localId in localIds:
                gt_entry = gt_dict.get(localId)
                if gt_entry:
                    gt_type = gt_entry.get('Heuristically_Root_Cause_Type', 'Unknown')
                    gt_type_counts[gt_type] += 1
                    
                    gt_dep_info = gt_entry.get('Heuristically_Root_Cause_Dependency', {})
                    if isinstance(gt_dep_info, dict):
                        gt_dep_name = gt_dep_info.get('name', '')
                        if gt_dep_name:
                            gt_dep_counts[gt_type] = gt_dep_name
            
            # LLM's group-level inference
            llm_group_type = inference.get('group_level_root_cause_type', 'Unknown')
            llm_group_dep = inference.get('group_level_root_cause_dependency')
            
            # Find GT's most frequent value
            if gt_type_counts:
                most_common_gt_type = max(gt_type_counts.items(), key=lambda x: x[1])[0]
                most_common_gt_dep = gt_dep_counts.get(most_common_gt_type, '')
                
                # Representative matching: Check type match
                if llm_group_type == most_common_gt_type:
                    # Also check dependency (for Dependency_Specific cases)
                    if llm_group_type == "Dependency_Specific":
                        if compare_dependencies(llm_group_dep, {'name': most_common_gt_dep} if most_common_gt_dep else None):
                            metrics.sub_group_representative_matches += 1
                    else:
                        # For Main_Project_Specific, match if both are None
                        llm_dep_is_none = (llm_group_dep is None or llm_group_dep == 'N/A' or llm_group_dep == '')
                        gt_dep_is_none = (not most_common_gt_dep or most_common_gt_dep == 'N/A')
                        if llm_dep_is_none and gt_dep_is_none:
                            metrics.sub_group_representative_matches += 1
            
            # Beyond Heuristic Accuracy (BHA) calculation
            # Detect cases where GT is Main_Project_Specific but actually Dependency_Specific
            # (e.g., qtbase case where GT classified as Main because not in srcmap)
            for localId in localIds:
                gt_entry = gt_dict.get(localId)
                if not gt_entry:
                    continue
                
                gt_type = gt_entry.get('Heuristically_Root_Cause_Type', 'Unknown')
                gt_project_name = gt_entry.get('project_name', '').lower().strip()
                
                # Get submodule_bug and repo_addr information from database
                import sqlite3
                DB_PATH = DEFAULT_ARVO_DB_PATH
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.execute("SELECT submodule_bug, repo_addr FROM arvo WHERE localId = ?", (localId,))
                    row = cursor.fetchone()
                    conn.close()
                    
                    if row:
                        submodule_bug = bool(row[0])
                        repo_addr = row[1] if row[1] else ''
                    else:
                        submodule_bug = False
                        repo_addr = ''
                except Exception:
                    submodule_bug = False
                    repo_addr = ''
                
                # BHA case conditions:
                # 1. GT is Main_Project_Specific but
                # 2. submodule_bug=True or repo_addr differs from project
                # 3. LLM correctly inferred as Dependency_Specific
                is_bha_case = False
                if gt_type == "Main_Project_Specific":
                    # Check submodule_bug flag
                    if submodule_bug:
                        is_bha_case = True
                    # Check if repo_addr differs from project
                    elif repo_addr and gt_project_name:
                        # Extract project name from repo_addr
                        repo_addr_lower = repo_addr.lower()
                        if gt_project_name not in repo_addr_lower:
                            is_bha_case = True
                
                if is_bha_case:
                    metrics.bha_cases += 1
                    # Check if LLM correctly inferred as Dependency_Specific
                    if llm_group_type == "Dependency_Specific":
                        # Check if LLM inferred a dependency
                        if llm_group_dep:
                            metrics.bha_correct += 1
    
    # Calculate average dependency matching ratio
    if metrics.sub_group_count > 0:
        total_ratio = sum(inf.get('dependency_matching_ratio', 0.0) for inf in inferences)
        metrics.dependency_matching_ratio_avg = total_ratio / metrics.sub_group_count
    
    # Workaround Detection Rate (WDR) calculation
    # Priority 1: Quantitative measurement for RQ1 validation
    #
    # WDR measures the ability to detect workaround patches (patches that don't fix root cause).
    # This directly validates RQ1: semantic reasoning capability
    #
    # Expected results:
    # - WDR Phase 1 (Heuristic): ~30-40% (heuristic misses semantic intent)
    # - WDR Phase 2 (LLM): ~70-80% (LLM detects semantic intent)
    #
    # Current limitation: Requires manual annotation of workaround cases
    # Ground truth workarounds are detected using multiple signals (≥2 signals needed):
    # 1. Phase 1 structural detection (patch_crash_distance >= 2, module_mismatch, control_flow_only)
    # 2. Commit message keywords (workaround, bypass, defensive, mitigation, etc.)
    # 3. Patch-crash distance and module mismatch
    # 4. Patch summary analysis (does not fix, does not address, etc.)
    # 5. Module mismatch evidence (crash in dependency, patch in main)
    #
    # Improvement needed for quantitative measurement:
    # 1. Manual annotation of at least 100 cases:
    #    - 50 clear workaround cases
    #    - 50 clear non-workaround cases
    #    - 50 ambiguous cases (for expert disagreement analysis)
    # 2. Store annotations in ground_truth.json or separate annotation file
    # 3. Compare Phase 1 vs Phase 2 detection rates on annotated set
    #
    # Why it matters:
    # - Direct quantitative evidence for "LLM surpasses heuristic" claim
    # - Validates semantic reasoning capability (RQ1)
    
    if logger:
        logger.info("Calculating Workaround Detection Rate (WDR)...")
    
    for local_id in evaluated_localIds:
        gt_item = gt_dict.get(local_id)
        feature = feature_map.get(local_id)
        inference = inference_map.get(local_id)
        
        if not gt_item or not feature:
            continue
        
        # Phase 1 detection
        phase1_result = workaround_detector.detect_phase1_workaround(gt_item)
        if phase1_result:
            metrics.wdr_phase1_detected += 1
        
        # Phase 2 detection
        phase2_result = False
        if inference:
            # Try to get individual inference from bug_type_groups first
            individual_rc = None
            for bg in inference_data.get('bug_type_groups', []):
                individual_root_causes = bg.get('individual_root_causes', {})
                if isinstance(individual_root_causes, dict):
                    rc_data = individual_root_causes.get(local_id)
                    if rc_data:
                        individual_rc = rc_data
            
            # Check IndividualRootCause.patch_intent if available
            if individual_rc:
                if isinstance(individual_rc, dict):
                    patch_intent = individual_rc.get('patch_intent', '')
                else:
                    patch_intent = getattr(individual_rc, 'patch_intent', '')
                
                if patch_intent in ['WORKAROUND', 'DEFENSIVE']:
                    phase2_result = True
            
            # Fallback to existing detection method
            if not phase2_result:
                phase2_result = workaround_detector.detect_phase2_workaround(feature, inference)
            
            if phase2_result:
                metrics.wdr_phase2_detected += 1
        
        # Ground truth detection
        commit_message = gt_item.get('commit_message', '')
        gt_workaround = workaround_detector.detect_ground_truth_workaround(
            gt_item, feature, commit_message
        )
        if gt_workaround:
            metrics.wdr_ground_truth_workarounds += 1
        
        # Calculate TP/FP/FN for Phase 1
        if phase1_result and gt_workaround:
            metrics.wdr_tp_phase1 += 1
        elif phase1_result and not gt_workaround:
            metrics.wdr_fp_phase1 += 1
        elif not phase1_result and gt_workaround:
            metrics.wdr_fn_phase1 += 1
        
        # Calculate TP/FP/FN for Phase 2
        if phase2_result and gt_workaround:
            metrics.wdr_tp_phase2 += 1
        elif phase2_result and not gt_workaround:
            metrics.wdr_fp_phase2 += 1
        elif not phase2_result and gt_workaround:
            metrics.wdr_fn_phase2 += 1
    
    # Cross-Project Validation Rate (CPVR) calculation
    # Priority 1: Quantitative measurement for RQ2 validation
    # 
    # CPVR measures how effectively LLM corrects outliers using cross-project pattern analysis.
    # Outlier: Case that disagrees with group majority (≥60%)
    # Success: LLM re-evaluates outlier using group context and corrects it to match group pattern
    #
    # Expected CPVR: ~60-70% (group consensus corrects individual errors)
    # This directly validates RQ2: cross-project pattern discovery capability
    #
    # Current limitation: Requires individual inferences from Module 1 to identify outliers
    # If individual inferences are not available, CPVR cannot be calculated
    #
    # Improvement needed:
    # 1. Ensure Module 1 outputs individual inferences for all cases
    # 2. Use bug_type_groups data if available
    # 3. Fallback: Use GT individual types if LLM individual inferences unavailable
    
    if logger:
        logger.info("Calculating Cross-Project Validation Rate (CPVR)...")
    
    outliers = identify_outliers(inference_data, logger)
    metrics.cpvr_total_outliers = len(outliers)
    metrics.cpvr_corrected_outliers = sum(1 for o in outliers if o.get('corrected') is True)
    metrics.cpvr_uncorrected_outliers = sum(1 for o in outliers if o.get('corrected') is False)
    
    if logger:
        if metrics.cpvr_total_outliers == 0:
            logger.warning("CPVR: No outliers identified. This may be due to:")
            logger.warning("  - Individual inferences not available from Module 1")
            logger.warning("  - All cases in sub-groups have same type (no outliers)")
            logger.warning("  - Sub-groups too small (<2 cases) or no clear majority (≥60%)")
            logger.warning("  - Improvement: Ensure Module 1 outputs individual inferences")
        else:
            cpvr_rate = (metrics.cpvr_corrected_outliers / metrics.cpvr_total_outliers * 100) if metrics.cpvr_total_outliers > 0 else 0
            logger.info(f"CPVR: {cpvr_rate:.2f}% ({metrics.cpvr_corrected_outliers}/{metrics.cpvr_total_outliers} corrected)")
    
    # Transitive Dependency Tracing calculation
    if logger:
        logger.info("Calculating Transitive Dependency Tracing...")
    
    phase1_success = 0
    phase2_success = 0
    phase2_deeper = 0
    
    for local_id in evaluated_localIds:
        gt_item = gt_dict.get(local_id)
        feature = feature_map.get(local_id)
        inference = inference_map.get(local_id)
        
        if not gt_item or not feature:
            continue
        
        # Get dependency chain from extracted data
        extracted = extracted_data.get(str(local_id), {})
        srcmap = extracted.get('srcmap', {})
        
        # Extract crash location
        stack_summary = feature.get('stack_trace_summary', '')
        crash_location = ''
        if stack_summary:
            # Try to extract file path from stack trace
            lines = stack_summary.split('\n')
            for line in lines:
                if '/src/' in line or 'src/' in line:
                    # Extract path
                    match = re.search(r'(src/[^\s:]+)', line)
                    if match:
                        crash_location = match.group(1)
                        break
        
        # Build dependency chain
        chain_path, depth = extract_dependency_chain(srcmap, crash_location, logger)
        
        if depth > 0:
            metrics.transitive_total_chains += 1
            
            # Phase 1: Can only detect 1-hop (direct dependencies)
            phase1_detected = depth <= 1
            
            # Phase 2: Check if LLM traced deeper
            phase2_detected = False
            phase2_depth = 0
            
            if inference:
                # Check if LLM identified dependency
                llm_dep = inference.get('group_level_root_cause_dependency', '')
                if llm_dep:
                    phase2_detected = True
                    # Try to estimate depth from LLM reasoning
                    reasoning = inference.get('llm_reasoning_process', '')
                    # Look for transitive chain mentions
                    if '→' in reasoning or '->' in reasoning or 'transitive' in reasoning.lower():
                        # Count arrows to estimate depth
                        arrows = reasoning.count('→') + reasoning.count('->')
                        phase2_depth = arrows + 1 if arrows > 0 else 1
                    else:
                        phase2_depth = 1  # Assume 1-hop if no explicit chain
            
            if phase1_detected:
                phase1_success += 1
            if phase2_detected:
                phase2_success += 1
            if phase2_depth > depth:
                phase2_deeper += 1
    
    metrics.transitive_phase1_success = phase1_success
    metrics.transitive_phase2_success = phase2_success
    metrics.transitive_phase2_deeper = phase2_deeper
    
    if logger:
        logger.info(f"Evaluated {len(evaluated_localIds)} localIds across {metrics.sub_group_count} sub-groups")
        logger.info(f"WDR: Phase1={metrics.wdr_phase1_detected}, Phase2={metrics.wdr_phase2_detected}, GT={metrics.wdr_ground_truth_workarounds}")
        logger.info(f"CPVR: Total={metrics.cpvr_total_outliers}, Corrected={metrics.cpvr_corrected_outliers}")
        logger.info(f"Transitive: Total={metrics.transitive_total_chains}, Phase1={phase1_success}, Phase2={phase2_success}")
    
    return metrics, detailed_results


def print_evaluation_summary(metrics: EvaluationMetrics, detailed_results: List[Dict], logger: Optional[logging.Logger] = None, arvo_baseline: Optional[Dict] = None):
    """Print evaluation summary"""
    derived_metrics = metrics.calculate_metrics()
    
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    
    print(f"\n[Overall Metrics]")
    print(f"  Total Cases Evaluated: {metrics.total_cases}")
    print(f"  Sub-Groups Evaluated: {metrics.sub_group_count}")
    
    print(f"\n[Accuracy]")
    print(f"  Type Accuracy: {derived_metrics['accuracy_type']:.2%} ({metrics.correct_type}/{metrics.total_cases})")
    print(f"  Dependency Accuracy: {derived_metrics['accuracy_dependency']:.2%} ({metrics.correct_dependency}/{metrics.total_cases})")
    print(f"  Both Correct: {derived_metrics['accuracy_both']:.2%} ({metrics.correct_both}/{metrics.total_cases})")
    
    print(f"\n[Main_Project_Specific Metrics]")
    print(f"  Precision: {derived_metrics['main_project_precision']:.2%}")
    print(f"  Recall: {derived_metrics['main_project_recall']:.2%}")
    print(f"  F1-Score: {derived_metrics['main_project_f1']:.2%}")
    print(f"  True Positives: {metrics.main_project_true_positives}")
    print(f"  False Positives: {metrics.main_project_false_positives}")
    print(f"  False Negatives: {metrics.main_project_false_negatives}")
    
    print(f"\n[Dependency_Specific Metrics]")
    print(f"  Precision: {derived_metrics['dependency_precision']:.2%}")
    print(f"  Recall: {derived_metrics['dependency_recall']:.2%}")
    print(f"  F1-Score: {derived_metrics['dependency_f1']:.2%}")
    print(f"  True Positives: {metrics.dependency_true_positives}")
    print(f"  False Positives: {metrics.dependency_false_positives}")
    print(f"  False Negatives: {metrics.dependency_false_negatives}")
    
    print(f"\n[Sub-Group Level Metrics]")
    print(f"  Perfect Matching:")
    print(f"    Type Accuracy: {derived_metrics['sub_group_accuracy_type']:.2%} ({metrics.sub_group_correct_type}/{metrics.sub_group_count})")
    print(f"    Dependency Accuracy: {derived_metrics['sub_group_accuracy_dependency']:.2%} ({metrics.sub_group_correct_dependency}/{metrics.sub_group_count})")
    print(f"    Both Correct: {derived_metrics['sub_group_accuracy_both']:.2%} ({metrics.sub_group_correct_both}/{metrics.sub_group_count})")
    print(f"  Partial Matching:")
    print(f"    Type Accuracy: {derived_metrics['sub_group_partial_type_accuracy']:.2%} (average match ratio)")
    print(f"    Dependency Accuracy: {derived_metrics['sub_group_partial_dep_accuracy']:.2%} (average match ratio)")
    print(f"  Representative Matching:")
    print(f"    Accuracy: {derived_metrics['sub_group_representative_accuracy']:.2%} ({metrics.sub_group_representative_matches}/{metrics.sub_group_count})")
    
    print(f"\n[Dependency Matching Analysis]")
    print(f"  Average Dependency Matching Ratio: {derived_metrics['dependency_matching_ratio_avg']:.2%}")
    print(f"  Total Dependency Matches: {metrics.dependency_matching_count_total}")
    
    print(f"\n[Beyond Heuristic Accuracy (BHA)]")
    if derived_metrics['bha_cases'] > 0:
        print(f"  BHA Cases: {derived_metrics['bha_cases']}")
        print(f"  BHA Correct: {derived_metrics['bha_correct']}")
        print(f"  BHA Accuracy: {derived_metrics['bha_accuracy']:.2%}")
        print(f"  Meaning: Ratio of cases where GT incorrectly classified as Main but LLM correctly inferred as Dependency")
    else:
        print(f"  BHA Cases: 0 (no evaluation targets)")
    
    print(f"\n[Workaround Detection Rate (WDR)]")
    if metrics.wdr_ground_truth_workarounds > 0:
        print(f"  Ground Truth Workarounds: {metrics.wdr_ground_truth_workarounds}")
        print(f"  Phase 1 (Heuristic) Detection:")
        print(f"    Detected: {metrics.wdr_phase1_detected}")
        print(f"    WDR: {derived_metrics['wdr_phase1']:.2f}%")
        print(f"    Precision: {derived_metrics['wdr_phase1_precision']:.3f}")
        print(f"    Recall: {derived_metrics['wdr_phase1_recall']:.3f}")
        print(f"  Phase 2 (LLM) Detection:")
        print(f"    Detected: {metrics.wdr_phase2_detected}")
        print(f"    WDR: {derived_metrics['wdr_phase2']:.2f}%")
        print(f"    Precision: {derived_metrics['wdr_phase2_precision']:.3f}")
        print(f"    Recall: {derived_metrics['wdr_phase2_recall']:.3f}")
        print(f"  Improvement: {derived_metrics['wdr_phase2'] - derived_metrics['wdr_phase1']:.2f}%")
    else:
        print(f"  Ground Truth Workarounds: 0 (no evaluation targets)")
    
    print(f"\n[Cross-Project Validation Rate (CPVR)]")
    if metrics.cpvr_total_outliers > 0:
        print(f"  Total Outliers: {metrics.cpvr_total_outliers}")
        print(f"  Corrected Outliers: {metrics.cpvr_corrected_outliers}")
        print(f"  Uncorrected Outliers: {metrics.cpvr_uncorrected_outliers}")
        print(f"  CPVR: {derived_metrics['cpvr']:.2f}%")
        print(f"  Meaning: Ratio of outliers that LLM corrected to match group majority")
    else:
        print(f"  Total Outliers: 0 (no evaluation targets)")
    
    print(f"\n[Transitive Dependency Tracing]")
    if metrics.transitive_total_chains > 0:
        print(f"  Total Dependency Chains: {metrics.transitive_total_chains}")
        print(f"  Phase 1 (1-hop only):")
        print(f"    Successfully detected: {metrics.transitive_phase1_success}")
        print(f"    Detection rate: {derived_metrics['transitive_phase1_rate']:.2f}%")
        print(f"  Phase 2 (n-hop):")
        print(f"    Successfully detected: {metrics.transitive_phase2_success}")
        print(f"    Detection rate: {derived_metrics['transitive_phase2_rate']:.2f}%")
        print(f"    Traced deeper than Phase 1: {metrics.transitive_phase2_deeper}")
        print(f"  Improvement: {derived_metrics['transitive_phase2_rate'] - derived_metrics['transitive_phase1_rate']:.2f}%")
    else:
        print(f"  Total Dependency Chains: 0 (no evaluation targets)")
    
    # Error analysis
    print(f"\n[Error Analysis]")
    type_errors = [r for r in detailed_results if not r['type_match']]
    dep_errors = [r for r in detailed_results if not r['dependency_match']]
    
    print(f"  Type Mismatches: {len(type_errors)}")
    if type_errors:
        print(f"    Sample errors:")
        for err in type_errors[:5]:
            print(f"      localId {err['localId']}: LLM={err['llm_type']}, GT={err['gt_type']}")
    
    print(f"  Dependency Mismatches: {len(dep_errors)}")
    if dep_errors:
        print(f"    Sample errors:")
        for err in dep_errors[:5]:
            print(f"      localId {err['localId']}: LLM={err['llm_dependency']}, GT={err['gt_dependency']}")
    
    print("\n" + "="*80)
    
    # Print paper metrics summary
    print_paper_metrics_summary(metrics, derived_metrics, arvo_baseline=arvo_baseline)
    
    if logger:
        logger.info("Evaluation summary printed")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM inference results against Ground Truth'
    )
    parser.add_argument(
        '--inference-file',
        type=str,
        default='llm_inference_results.json',
        help='Path to LLM inference results JSON file (default: llm_inference_results.json)'
    )
    parser.add_argument(
        '--gt-file',
        type=str,
        default='ground_truth.json',
        help='Path to Ground Truth JSON file (default: ground_truth.json)'
    )
    parser.add_argument(
        '--extracted-data',
        type=str,
        default=None,
        help='Path to extracted data JSON file (for transitive dependency analysis, default: None)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Path to save evaluation results (default: evaluation_results.json)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='evaluation.log',
        help='Path to log file (default: evaluation.log)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--num',
        type=int,
        default=None,
        help='Limit number of cases to evaluate (default: evaluate all)'
    )
    parser.add_argument(
        '--localIds',
        type=str,
        default=None,
        help='Comma-separated list of localIds to evaluate (e.g., "383170478,377977949,42538984")'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("GT-based Evaluation Script")
    logger.info("="*80)
    logger.info(f"Inference file: {args.inference_file}")
    logger.info(f"GT file: {args.gt_file}")
    if args.extracted_data:
        logger.info(f"Extracted data file: {args.extracted_data}")
    logger.info(f"Output file: {args.output}")
    
    # Evaluate
    try:
        # Parse localIds if provided
        target_localIds = None
        if args.localIds:
            try:
                target_localIds = [int(lid.strip()) for lid in args.localIds.split(',')]
                logger.info(f"Evaluating specific localIds: {target_localIds}")
            except ValueError:
                logger.error(f"Invalid localIds format: {args.localIds}. Expected comma-separated integers.")
                return
        
        metrics, detailed_results = evaluate_llm_inference(
            args.inference_file,
            args.gt_file,
            extracted_data_file=args.extracted_data,
            num_cases=args.num,
            target_localIds=target_localIds,
            logger=logger
        )
        
        # Calculate derived metrics
        derived_metrics = metrics.calculate_metrics()

        # Compute ARVO baseline metrics from ARVO DB for the evaluated localIds
        arvo_baseline = compute_arvo_submodule_baseline_metrics(
            detailed_results,
            db_path=DEFAULT_ARVO_DB_PATH,
            logger=logger
        )
        
        # Print summary
        print_evaluation_summary(metrics, detailed_results, logger=logger, arvo_baseline=arvo_baseline)
        
        # Save results
        evaluation_results = {
            'summary': {
                'total_cases': metrics.total_cases,
                'sub_group_count': metrics.sub_group_count,
                'metrics': derived_metrics,
                'arvo_baseline': arvo_baseline,
                'raw_counts': {
                    'correct_type': metrics.correct_type,
                    'correct_dependency': metrics.correct_dependency,
                    'correct_both': metrics.correct_both,
                    'main_project_tp': metrics.main_project_true_positives,
                    'main_project_fp': metrics.main_project_false_positives,
                    'main_project_fn': metrics.main_project_false_negatives,
                    'dependency_tp': metrics.dependency_true_positives,
                    'dependency_fp': metrics.dependency_false_positives,
                    'dependency_fn': metrics.dependency_false_negatives,
                    'wdr_phase1_detected': metrics.wdr_phase1_detected,
                    'wdr_phase2_detected': metrics.wdr_phase2_detected,
                    'wdr_ground_truth_workarounds': metrics.wdr_ground_truth_workarounds,
                    'wdr_tp_phase1': metrics.wdr_tp_phase1,
                    'wdr_fp_phase1': metrics.wdr_fp_phase1,
                    'wdr_fn_phase1': metrics.wdr_fn_phase1,
                    'wdr_tp_phase2': metrics.wdr_tp_phase2,
                    'wdr_fp_phase2': metrics.wdr_fp_phase2,
                    'wdr_fn_phase2': metrics.wdr_fn_phase2,
                    'cpvr_total_outliers': metrics.cpvr_total_outliers,
                    'cpvr_corrected_outliers': metrics.cpvr_corrected_outliers,
                    'cpvr_uncorrected_outliers': metrics.cpvr_uncorrected_outliers,
                    'transitive_total_chains': metrics.transitive_total_chains,
                    'transitive_phase1_success': metrics.transitive_phase1_success,
                    'transitive_phase2_success': metrics.transitive_phase2_success,
                    'transitive_phase2_deeper': metrics.transitive_phase2_deeper,
                }
            },
            'detailed_results': detailed_results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {args.output}")
        print(f"\n[+] Evaluation results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


def print_paper_metrics_summary(metrics: EvaluationMetrics, derived_metrics: Dict, arvo_baseline: Optional[Dict] = None):
    """Print paper metrics summary in a readable format"""
    print("\n" + "=" * 80)
    print("📊 PAPER METRICS SUMMARY (Phase 2 - LLM Evaluation)")
    print("=" * 80)
    
    print(f"\n📈 Overall Performance:")
    print(f"  • Type Accuracy: {derived_metrics['accuracy_type']:.2%} ({metrics.correct_type}/{metrics.total_cases})")
    print(f"  • Dependency Name Accuracy: {derived_metrics['accuracy_dependency']:.2%} ({metrics.correct_dependency}/{metrics.total_cases})")
    print(f"  • Both Correct: {derived_metrics['accuracy_both']:.2%} ({metrics.correct_both}/{metrics.total_cases})")
    if arvo_baseline and arvo_baseline.get("available"):
        print(f"\n🧱 ARVO Baseline (DB-derived):")
        print(f"  • Definition: {arvo_baseline.get('definition')}")
        print(f"  • Type Accuracy: {arvo_baseline['accuracy_type']:.2%} ({arvo_baseline['raw_counts']['correct_type']}/{arvo_baseline['total_cases']})")
        print(f"  • Balanced Type Accuracy: {arvo_baseline['balanced_accuracy_type']:.2%}")
        print(f"  • Dependency Recall: {arvo_baseline['dependency_recall']:.2%} (predicted Dependency: {arvo_baseline['predicted_dependency']})")
    
    print(f"\n📝 Paper Values (Overall):")
    print(f"  • **{derived_metrics['accuracy_type']:.2%}** - Type accuracy")
    print(f"  • **{derived_metrics['accuracy_dependency']:.2%}** - Dependency name accuracy")
    print(f"  • **{metrics.correct_type}/{metrics.total_cases}** - Correct type classifications")
    print(f"  • **{metrics.correct_dependency}/{metrics.total_cases}** - Correct dependency matches")
    
    # Per-Type Performance
    print(f"\n📊 Per-Type Performance:")
    print(f"  Main_Project_Specific:")
    print(f"    • Precision: {derived_metrics['main_project_precision']:.2%} ({metrics.main_project_true_positives}/{metrics.main_project_true_positives + metrics.main_project_false_positives if metrics.main_project_true_positives + metrics.main_project_false_positives > 0 else 0})")
    print(f"    • Recall: {derived_metrics['main_project_recall']:.2%} ({metrics.main_project_true_positives}/{metrics.main_project_true_positives + metrics.main_project_false_negatives if metrics.main_project_true_positives + metrics.main_project_false_negatives > 0 else 0})")
    print(f"    • F1: {derived_metrics['main_project_f1']:.2%}")
    print(f"  Dependency_Specific:")
    print(f"    • Precision: {derived_metrics['dependency_precision']:.2%} ({metrics.dependency_true_positives}/{metrics.dependency_true_positives + metrics.dependency_false_positives if metrics.dependency_true_positives + metrics.dependency_false_positives > 0 else 0})")
    print(f"    • Recall: {derived_metrics['dependency_recall']:.2%} ({metrics.dependency_true_positives}/{metrics.dependency_true_positives + metrics.dependency_false_negatives if metrics.dependency_true_positives + metrics.dependency_false_negatives > 0 else 0})")
    print(f"    • F1: {derived_metrics['dependency_f1']:.2%}")
    
    print(f"\n📝 Paper Values (Per-Type):")
    print(f"  • **{derived_metrics['main_project_precision']:.2%}** - Main precision")
    print(f"  • **{derived_metrics['main_project_recall']:.2%}** - Main recall")
    print(f"  • **{derived_metrics['main_project_f1']:.2%}** - Main F1")
    print(f"  • **{derived_metrics['dependency_precision']:.2%}** - Dependency precision")
    print(f"  • **{derived_metrics['dependency_recall']:.2%}** - Dependency recall")
    print(f"  • **{derived_metrics['dependency_f1']:.2%}** - Dependency F1")
    
    # Sub-Group Metrics
    print(f"\n🔗 Sub-Group Level Metrics:")
    print(f"  • Perfect Type Matching: {derived_metrics['sub_group_accuracy_type']:.2%} ({metrics.sub_group_correct_type}/{metrics.sub_group_count})")
    print(f"  • Perfect Dependency Matching: {derived_metrics['sub_group_accuracy_dependency']:.2%} ({metrics.sub_group_correct_dependency}/{metrics.sub_group_count})")
    print(f"  • Partial Type Accuracy: {derived_metrics['sub_group_partial_type_accuracy']:.2%}")
    print(f"  • Partial Dependency Accuracy: {derived_metrics['sub_group_partial_dep_accuracy']:.2%}")
    
    print(f"\n📝 Paper Values (Sub-Group):")
    print(f"  • **{derived_metrics['sub_group_accuracy_type']:.2%}** - Sub-group type matching")
    print(f"  • **{derived_metrics['sub_group_accuracy_dependency']:.2%}** - Sub-group dependency matching")
    
    # BHA
    if derived_metrics['bha_cases'] > 0:
        print(f"\n🎯 Beyond Heuristic Accuracy (BHA):")
        print(f"  • BHA Cases: {derived_metrics['bha_cases']}")
        print(f"  • LLM Corrected GT Errors: {derived_metrics['bha_correct']}")
        print(f"  • BHA Accuracy: {derived_metrics['bha_accuracy']:.2%}")
        print(f"\n📝 Paper Values (BHA):")
        print(f"  • **{derived_metrics['bha_cases']}** - LLM-GT disagreement cases")
        print(f"  • **{derived_metrics['bha_correct']}** - LLM corrected GT errors")
        print(f"  • **{derived_metrics['bha_accuracy']:.2%}** - BHA (conservative estimate)")
    
    # CPVR
    if metrics.cpvr_total_outliers > 0:
        print(f"\n🔄 Cross-Project Validation Rate (CPVR):")
        print(f"  • Total Outliers: {metrics.cpvr_total_outliers}")
        print(f"  • Corrected: {metrics.cpvr_corrected_outliers}")
        print(f"  • CPVR: {derived_metrics['cpvr']:.2f}%")
        print(f"\n📝 Paper Values (CPVR):")
        print(f"  • **{derived_metrics['cpvr']:.2f}%** - CPVR (target: >60%)")
    
    # WDR
    if metrics.wdr_ground_truth_workarounds > 0:
        print(f"\n🛡️  Workaround Detection Rate (WDR):")
        print(f"  • Phase 1 Detection: {derived_metrics['wdr_phase1']:.2f}%")
        print(f"  • Phase 2 Detection: {derived_metrics['wdr_phase2']:.2f}%")
        print(f"  • Improvement: {derived_metrics['wdr_phase2'] - derived_metrics['wdr_phase1']:.2f}%")
        print(f"\n📝 Paper Values (WDR):")
        print(f"  • **{derived_metrics['wdr_phase1']:.2f}%** - Phase 1 WDR")
        print(f"  • **{derived_metrics['wdr_phase2']:.2f}%** - Phase 2 WDR")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()


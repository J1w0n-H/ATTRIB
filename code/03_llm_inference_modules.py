#!/usr/bin/env python3
"""
LLM-Based Inference Modules: Orchestrating Semantic Reasoning

This script implements a hierarchical LLM-based inference pipeline:

Module 3.1: Vulnerability Feature Extraction & Summarization
    - Extracts and summarizes vulnerability features from OSSFuzz reports

Module 1: Bug Type Group Analysis (Macro-level Pattern Recognition)
    - Groups vulnerabilities by bug_type
    - Performs macro-level pattern recognition
    - Identifies common dependencies and initial patterns

Module 2: Fine-Grained Sub-Grouping within Bug Type Groups
    - Performs detailed sub-grouping within each bug type group
    - Infers initial root cause for each sub-group

Module 3: Cross-Group Root Cause Inference & Validation
    - Performs final root cause inference based on Module 1 and 2 outputs
    - Compares patterns across bug type groups and sub-groups
    - Performs discrepancy analysis with heuristic Ground Truth
"""

import sys
import json
import sqlite3
import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
from dataclasses import dataclass, asdict

# Import extract_from_db functions
sys.path.insert(0, str(Path(__file__).parent))
try:
    # Try importing from 01_extract_from_db.py (file name starts with number)
    import importlib.util
    extract_module_path = Path(__file__).parent / "01_extract_from_db.py"
    if extract_module_path.exists():
        spec = importlib.util.spec_from_file_location("extract_from_db", extract_module_path)
        extract_from_db = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extract_from_db)
        get_report_from_db = extract_from_db.get_report_from_db
        extract_data = extract_from_db.extract_data
    else:
        raise ImportError(f"Could not find 01_extract_from_db.py at {extract_module_path}")
except Exception as e:
    print(f"[!] Warning: Could not import extract_from_db functions: {e}")
    # Define fallback functions
    def get_report_from_db(local_id: int):
        raise RuntimeError("extract_from_db module not available. Please ensure 01_extract_from_db.py exists.")
    def extract_data(local_id: int, include_code_snippets: bool = False, auto_fetch: bool = False):
        raise RuntimeError("extract_from_db module not available. Please ensure 01_extract_from_db.py exists.")

DB_PATH = os.environ.get("ARVO_DB_PATH") or str((Path(__file__).resolve().parents[1] / "arvo.db"))

@dataclass
class PatchSemanticClassification:
    """LLM-based patch semantic classification (Semantic Evidence Extractor)"""
    patch_intent: str  # "workaround" | "real_fix" | "refactor" | "version_bump"
    assumed_fault_location: str  # "dependency" | "main" | "unclear"
    fix_scope: str  # "local" | "cross_module"
    error_handling_pattern: str  # "added_check" | "changed_logic" | "memory_fix" | "none"
    confidence: float = 0.0  # 0.0-1.0
    reasoning: str = ""

@dataclass
class FrameAttribution:
    """LLM-based stack trace frame attribution"""
    logical_owner: Optional[str] = None  # e.g., "libjxl"
    confidence: float = 0.0  # 0.0-1.0
    reason: str = ""
    top_frames_analyzed: int = 0

@dataclass
class ContradictionScan:
    """LLM-based contradiction detection for root cause inference"""
    contradiction: bool = False
    contradiction_type: Optional[str] = None  # e.g., "direct_fix_in_dependency"
    severity: str = "none"  # "high" | "medium" | "low" | "none"
    explanation: str = ""
    confidence: float = 0.0  # 0.0-1.0

@dataclass
class VulnerabilityFeatures:
    """Vulnerability feature extraction results"""
    localId: int
    project_name: str
    bug_type: str
    severity: str
    stack_trace_summary: str
    patch_summary: str
    dependencies_summary: str
    code_snippets_summary: str
    llm_reasoning_summary: str
    semantic_embedding: Optional[List[float]] = None
    # Structural features (from Phase 1 or computed)
    patch_crash_distance: Optional[int] = None
    patch_semantic_type: Optional[str] = None  # VALIDATION_ONLY, ALGORITHM_CHANGE, REFACTOR, etc.
    crash_module: Optional[str] = None
    patched_module: Optional[str] = None
    control_flow_only: Optional[bool] = None
    workaround_detected: Optional[bool] = None  # from Phase 1 GT
    # LLM-based semantic evidence (NEW)
    patch_semantic_classification: Optional[PatchSemanticClassification] = None
    frame_attribution: Optional[FrameAttribution] = None

@dataclass
class ClusterInfo:
    """Cluster information"""
    cluster_id: int
    localIds: List[int]
    common_characteristics: str
    common_dependencies: List[str]
    llm_cluster_summary: str

@dataclass
class IndividualRootCause:
    """Individual root cause inference result from Module 1"""
    localId: int
    root_cause_type: str  # Main_Project_Specific | Dependency_Specific
    root_cause_dependency: Optional[str] = None
    patch_intent: Optional[str] = None  # ACTUAL_FIX, WORKAROUND, DEFENSIVE
    is_workaround_patch: Optional[bool] = None  # True if workaround, False if real fix, None if unknown
    patch_semantic_llm_opinion: Optional[str] = None
    main_project_score: float = 0.0
    dependency_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    evidence: str = ""

@dataclass
class BugTypeGroupInfo:
    """Module 1: Bug Type Group Analysis results"""
    bug_type: str
    localIds: List[int]
    bug_type_group_summary: str
    common_dependencies_in_group: List[str]
    initial_pattern_observation: str
    group_embedding: Optional[List[float]] = None
    confidence_score: float = 0.0
    individual_root_causes: Optional[Dict[int, 'IndividualRootCause']] = None
    
    def __post_init__(self):
        if self.individual_root_causes is None:
            self.individual_root_causes = {}

@dataclass
class SubGroupInfo:
    """Module 2: Fine-Grained Sub-Grouping results"""
    sub_group_id: int
    bug_type_group: str
    localIds: List[int]
    pattern_description: str = ""  # Description of the vulnerability pattern shared by this sub-group
    grouping_reasoning: str = ""  # Explanation of why these vulnerabilities form a sub-group (CoT reasoning)
    inferred_root_cause_type: str = "Unknown"  # "Main_Project_Specific" or "Dependency_Specific"
    inferred_root_cause_dependency: Optional[str] = None
    reasoning: str = ""  # Root cause inference reasoning (deprecated, use grouping_reasoning)
    common_dependency_versions: Optional[List[str]] = None
    confidence_score: float = 0.0

@dataclass
class RootCauseInference:
    """Root cause inference results - per Sub-Group (Group-based inference)"""
    sub_group_id: int  # Sub-Group ID from Module 2
    bug_type_group: str  # Bug type group name
    localIds: List[int]  # All localIds in this Sub-Group
    group_level_root_cause_type: str  # "Main_Project_Specific" or "Dependency_Specific"
    group_pattern_justification: str  # Justification based on group patterns and shared dependencies
    group_level_root_cause_dependency: Optional[str] = None  # Dependency name if Dependency_Specific
    group_level_root_cause_dependency_version: Optional[str] = None  # Dependency version if available
    dependency_matching_ratio: float = 0.0  # Ratio of localIds in group that share the root cause dependency (0.0-1.0)
    dependency_matching_count: int = 0  # Number of localIds that share the root cause dependency
    cross_project_propagation_insight: Optional[str] = None  # Cross-project propagation analysis
    cve_validation: Optional[str] = None  # CVE pattern validation result
    llm_reasoning_process: str = ""  # Full LLM reasoning process
    confidence_score: float = 0.0  # Overall confidence score (0.0-1.0)
    main_project_score: float = 0.0  # Score for Main_Project_Specific (0.0-1.0)
    dependency_score: float = 0.0  # Score for Dependency_Specific (0.0-1.0)
    evidence_sources: List[str] = None
    module1_confidence: Optional[float] = None  # Confidence from Module 1 analysis
    module2_confidence: Optional[float] = None  # Confidence from Module 2 sub-grouping
    module3_confidence: Optional[float] = None  # Confidence from Module 3 final inference
    discrepancy_analysis: Optional[str] = None  # Analysis of discrepancy with heuristic GT
    discrepancy_type: Optional[str] = None  # Type of discrepancy: "heuristic_error", "llm_error", "borderline_case"
    corrective_reasoning: Optional[str] = None  # Rebuttal reasoning when disagreeing with GT
    per_localId_discrepancies: List[Dict] = None  # Per-localId discrepancy details
    contradiction_scan: Optional['ContradictionScan'] = None  # Contradiction scan result (NEW)
    causal_flow_explanation: Optional[str] = None  # Causal flow narrative explanation (NEW)
    
    def __post_init__(self):
        if self.evidence_sources is None:
            self.evidence_sources = []
        if self.per_localId_discrepancies is None:
            self.per_localId_discrepancies = []

class LLMInferenceModules:
    """Integrated LLM-based inference module class"""
    
    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "o4-mini"):
        """
        Args:
            llm_api_key: LLM API key (if None, tries to get from OPENAI_API_KEY env var)
            llm_model: LLM model name to use (default: o4-mini)
        """
        # Try to get API key from environment variable if not provided
        if llm_api_key is None:
            llm_api_key = os.getenv('OPENAI_API_KEY')
        
        if llm_api_key is None:
            raise ValueError(
                "LLM API key is required. "
                "Please set OPENAI_API_KEY environment variable or use --llm-api-key option."
            )
        
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, logger: Optional[logging.Logger] = None, max_retries: int = 3) -> str:
        """
        LLM API call with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            logger: Optional logger for logging LLM requests and responses
            max_retries: Maximum number of retries for connection errors (default: 3)
        
        Returns:
            LLM response text
        """
        import time
        import openai
        
        # Set timeout for API calls (60 seconds default, 120 seconds for long requests)
        timeout = 120.0
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=self.llm_api_key, timeout=timeout)
                
                # Log request with immediate flush
                if attempt == 0:
                    if logger:
                        logger.info(f"[LLM] Starting API call (model: {self.llm_model}, timeout: {timeout}s)")
                        logger.debug(f"LLM Request (model: {self.llm_model}):")
                        logger.debug(f"Prompt length: {len(prompt)} chars")
                        logger.debug(f"Prompt preview: {prompt[:200]}...")
                        for handler in logger.handlers:
                            if isinstance(handler, logging.FileHandler):
                                handler.flush()
                    else:
                        print(f"[*] Calling LLM API (model: {self.llm_model}, timeout: {timeout}s)...")
                elif logger:
                    logger.info(f"[LLM] Retry attempt {attempt + 1}/{max_retries}")
                else:
                    print(f"[*] Retry attempt {attempt + 1}/{max_retries}...")
                
                # Use appropriate API format based on model type
                if self.llm_model.startswith('o'):
                    # o-series models (o4-mini, etc.) - no temperature parameter
                    try:
                        # Try with reasoning_effort first (for o1, o3 series)
                        if logger and attempt == 0:
                            logger.debug("Trying with reasoning_effort='medium'")
                        response = client.chat.completions.create(
                            model=self.llm_model,
                            messages=messages,
                            reasoning_effort="medium"
                        )
                    except Exception as e:
                        # Fallback: o4-mini doesn't support reasoning_effort or temperature
                        if logger and attempt == 0:
                            logger.debug(f"Trying o-series model without reasoning_effort: {e}")
                        response = client.chat.completions.create(
                            model=self.llm_model,
                            messages=messages
                        )
                else:
                    # Standard models (gpt-4, gpt-5, etc.) - support temperature
                    response = client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        temperature=0.7
                    )
                
                llm_response = response.choices[0].message.content
                
                # Log response with immediate flush
                if logger:
                    logger.info(f"[LLM] API call successful: {len(llm_response)} chars response")
                    logger.info(f"LLM Response (model: {self.llm_model}):")
                    logger.info(f"Response length: {len(llm_response)} chars")
                    logger.info(f"Response preview: {llm_response[:500]}...")
                    # Log full response if not too long (limit to 5000 chars)
                    if len(llm_response) <= 5000:
                        logger.info(f"Full LLM Response:\n{llm_response}")
                    else:
                        logger.info(f"Full LLM Response (first 5000 chars):\n{llm_response[:5000]}...")
                    # Force flush to ensure immediate write
                    for handler in logger.handlers:
                        if isinstance(handler, logging.FileHandler):
                            handler.flush()
                else:
                    print(f"[+] LLM API call successful: {len(llm_response)} chars response")
                
                return llm_response
                
            except (openai.APIConnectionError, openai.APITimeoutError, ConnectionError) as e:
                # Connection errors - retry with exponential backoff
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    if logger:
                        logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    else:
                        print(f"[!] Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    break
            except Exception as e:
                # Other errors - don't retry
                error_msg = f"LLM API call failed: {e}"
                error_details = str(e)
                if logger:
                    logger.error(error_msg)
                    logger.error(f"Error details: {error_details}")
                    # Log more details if available
                    if hasattr(e, 'response'):
                        try:
                            error_body = e.response.json() if hasattr(e.response, 'json') else str(e.response)
                            logger.error(f"API error response: {error_body}")
                        except:
                            pass
                print(f"[!] {error_msg}")
                print(f"[!] Error details: {error_details}")
                raise RuntimeError(f"LLM API call failed: {error_msg}") from e
        
        # All retries exhausted
        error_msg = f"LLM API call failed after {max_retries} attempts: {last_error}"
        if logger:
            logger.error(error_msg)
        print(f"[!] {error_msg}")
        raise RuntimeError(error_msg) from last_error
    
    def generate_embedding(self, text: str, logger: Optional[logging.Logger] = None) -> Optional[List[float]]:
        """
        Generate semantic embedding from text
        
        Args:
            text: Text to embed
            logger: Optional logger for logging
        
        Returns:
            Embedding vector or None if failed
        """
        try:
            import openai
            client = openai.OpenAI(api_key=self.llm_api_key)
            
            # Try different embedding models in order of preference
            models_to_try = [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002"
            ]
            
            last_error = None
            for model in models_to_try:
                try:
                    response = client.embeddings.create(
                        model=model,
                        input=text
                    )
                    if logger:
                        logger.debug(f"Successfully generated embedding using model: {model}")
                    return response.data[0].embedding
                except Exception as e:
                    last_error = e
                    if logger:
                        logger.debug(f"Failed to use embedding model {model}: {e}")
                    continue
            
            # All models failed
            error_msg = f"All embedding models failed. Last error: {last_error}"
            if logger:
                logger.warning(error_msg)
            print(f"[!] Warning: {error_msg}")
            return None
            
        except Exception as e:
            error_msg = f"Embedding API call failed: {e}"
            if logger:
                logger.warning(error_msg)
            print(f"[!] Warning: {error_msg}")
            return None

    # ========================================================================
    # Feature Extraction Module
    # ========================================================================
    
    def extract_vulnerability_features(self, localId: int, data: Optional[Dict] = None, include_code_snippets: bool = False, 
                                      skip_patch_summary: bool = False, skip_dependency_description: bool = False, 
                                      skip_reasoning_summary: bool = False, ground_truth: Optional[Dict] = None,
                                      logger: Optional[logging.Logger] = None) -> VulnerabilityFeatures:
        """
        Extract and summarize vulnerability features
        
        Args:
            localId: localId to analyze
            data: Pre-extracted data (auto-extracted if None)
            include_code_snippets: Whether to include code snippets (default: False, can be slow/timeout)
            skip_patch_summary: Skip patch summary generation (experiment mode)
            skip_dependency_description: Skip dependency description generation (experiment mode)
            skip_reasoning_summary: Skip LLM reasoning summary generation (experiment mode)
        
        Returns:
            VulnerabilityFeatures object
        """
        print(f"[Feature Extraction] Extracting features for localId {localId}...")
        
        # Extract data
        if data is None:
            try:
                data = extract_data(localId, include_code_snippets=include_code_snippets, auto_fetch=False)
            except Exception as e:
                # If code snippet extraction fails, try without it
                if include_code_snippets:
                    print(f"[-] Error extracting data with code snippets: {e}")
                    print(f"[*] Retrying without code snippets...")
                    try:
                        data = extract_data(localId, include_code_snippets=False, auto_fetch=False)
                    except Exception as e2:
                        print(f"[-] Error extracting data: {e2}")
                        return None
                else:
                    print(f"[-] Error extracting data: {e}")
                    return None
        
        if not data:
            return None
        
        # Prepare data
        osssfuzz_report = data.get('osssfuzz_report', {})
        project_name = osssfuzz_report.get('project_name', '')
        bug_type = osssfuzz_report.get('bug_type', '')
        severity = osssfuzz_report.get('severity', '')
        stack_trace = data.get('stack_trace', '')
        patch_info = data.get('patch_info', {})
        patch_diff = patch_info.get('patch_diff', '')
        srcmap = data.get('srcmap', {})
        code_snippets = data.get('code_snippets', {})
        
        # Feature extraction and summarization using LLM
        print(f"  [*] Analyzing vulnerability characteristics with LLM...")
        import sys
        sys.stdout.flush()  # Ensure print is flushed
        
        # Stack trace summary
        print(f"    [*] Summarizing stack trace...")
        sys.stdout.flush()
        stack_trace_summary = self._summarize_stack_trace(stack_trace, project_name, logger=logger)
        
        # Patch summary
        if skip_patch_summary:
            patch_summary = f"Patch diff available ({len(patch_diff)} chars)" if patch_diff else "No patch diff available"
            print(f"    [-] Skipping patch summary (experiment mode)")
        else:
            print(f"    [*] Summarizing patch...")
            sys.stdout.flush()
            patch_summary = self._summarize_patch(patch_diff, patch_info, logger=logger)
        
        # Dependencies summary (with stack trace for consistency with GT)
        if skip_dependency_description:
            deps_count = len(srcmap.get('vulnerable_version', {}).get('dependencies', []))
            dependencies_summary = f"Dependencies available ({deps_count} dependencies from srcmap)" if deps_count > 0 else "No dependencies found in srcmap"
            print(f"    [-] Skipping dependency description (experiment mode)")
        else:
            print(f"    [*] Summarizing dependencies...")
            sys.stdout.flush()
            dependencies_summary = self._summarize_dependencies(
                srcmap, project_name=project_name, stack_trace=stack_trace, logger=logger
            )
        
        # Code snippets summary
        print(f"    [*] Summarizing code snippets...")
        sys.stdout.flush()
        code_snippets_summary = self._summarize_code_snippets(code_snippets)
        
        # LLM reasoning summary (Chain-of-Thought) - This is the longest step
        if skip_reasoning_summary:
            llm_reasoning_summary = f"Reasoning summary skipped (experiment mode). Stack trace, patch, and dependencies summaries available."
            print(f"    [-] Skipping LLM reasoning summary (experiment mode)")
        else:
            print(f"    [*] Generating LLM reasoning summary (this may take 30-120 seconds)...")
            sys.stdout.flush()
            llm_reasoning_summary = self._generate_reasoning_summary(
                project_name, bug_type, severity,
                stack_trace_summary, patch_summary,
                dependencies_summary, code_snippets_summary,
                logger=logger
            )
            print(f"    [+] LLM reasoning summary complete")
        sys.stdout.flush()
        
        # Generate semantic embedding (DISABLED - embedding generation is disabled)
        # combined_text = f"{project_name} {bug_type} {stack_trace_summary} {patch_summary} {dependencies_summary}"
        # semantic_embedding = self.generate_embedding(combined_text, logger=None)
        semantic_embedding = None  # Embedding disabled
        # if semantic_embedding is None:
        #     print(f"  [!] Warning: Could not generate semantic embedding, continuing without it")
        
        # Compute structural features (patch-crash distance, modules, etc.)
        print(f"    [*] Computing structural features...")
        sys.stdout.flush()
        patched_files = patch_info.get('patched_files', [])
        patch_crash_distance, crash_module, patched_module = self._compute_patch_crash_distance(
            stack_trace, patched_files, project_name
        )
        
        # Analyze patch semantic type
        patch_semantic_type = self._analyze_patch_semantic_type(patch_diff, patched_files)
        control_flow_only = self._is_control_flow_only_patch(patch_diff)
        
        # Try to load workaround_detected from GT if available
        workaround_detected = None
        if ground_truth:
            gt_entry = ground_truth.get(str(localId)) or ground_truth.get(int(localId))
            if gt_entry:
                workaround_detected = gt_entry.get('workaround_detected')
        
        # Fallback to calculation if GT not available
        if workaround_detected is None and patch_crash_distance is not None and crash_module and patched_module:
            # Compute workaround_detected: patch_crash_distance >= 2 AND module mismatch
            workaround_detected = (
                patch_crash_distance >= 2 and
                crash_module != patched_module
            )
        
        features = VulnerabilityFeatures(
            localId=localId,
            project_name=project_name,
            bug_type=bug_type,
            severity=severity,
            stack_trace_summary=stack_trace_summary,
            patch_summary=patch_summary,
            dependencies_summary=dependencies_summary,
            code_snippets_summary=code_snippets_summary,
            llm_reasoning_summary=llm_reasoning_summary,
            semantic_embedding=semantic_embedding,
            patch_crash_distance=patch_crash_distance,
            patch_semantic_type=patch_semantic_type,
            crash_module=crash_module,
            patched_module=patched_module,
            control_flow_only=control_flow_only,
            workaround_detected=workaround_detected
        )
        
        print(f"  [+] Feature extraction complete")
        return features
    
    def _classify_patch_semantics(self, patch_diff: str, patch_info: Dict, patched_files: List[str],
                                  logger: Optional[logging.Logger] = None) -> Optional[PatchSemanticClassification]:
        """
        🔴 1순위 개선: LLM-based Patch Semantic Classifier
        
        LLM을 "Semantic Evidence Extractor"로 사용하여 패치의 의미를 분류합니다.
        LLM은 점수를 매기지 않고 의미만 추출합니다 (핵심).
        
        Returns:
            PatchSemanticClassification 객체 또는 None (실패 시)
        """
        if not patch_diff:
            return None
        
        fix_commit = patch_info.get('fix_commit', '')
        patch_preview = patch_diff[:2000] if len(patch_diff) > 2000 else patch_diff
        
        prompt = f"""Analyze the following patch diff and classify its semantic meaning.

Patch Diff:
{patch_preview}
{'...' if len(patch_diff) > 2000 else ''}

Patched Files: {', '.join(patched_files[:10])}
Fix Commit: {fix_commit[:200] if fix_commit else 'N/A'}

Classify the patch semantics and provide your analysis in the following JSON format:

{{
    "patch_intent": "workaround" | "real_fix" | "refactor" | "version_bump",
    "assumed_fault_location": "dependency" | "main" | "unclear",
    "fix_scope": "local" | "cross_module",
    "error_handling_pattern": "added_check" | "changed_logic" | "memory_fix" | "none",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your classification"
}}

Guidelines:
- patch_intent: What is the primary intent of this patch?
  - "workaround": Adds defensive code/workaround without fixing root cause
  - "real_fix": Directly fixes the underlying bug
  - "refactor": Code restructuring without fixing bugs
  - "version_bump": Only updates dependency versions
  
- assumed_fault_location: Where does the patch assume the fault is?
  - "dependency": Patch assumes fault is in dependency (e.g., adds validation for dependency input)
  - "main": Patch assumes fault is in main project code
  - "unclear": Cannot determine from patch alone
  
- fix_scope: Scope of the fix
  - "local": Fix is localized to a single module/file
  - "cross_module": Fix spans multiple modules/files
  
- error_handling_pattern: What type of error handling was added/changed?
  - "added_check": Added validation/check (e.g., null check, bounds check)
  - "changed_logic": Changed core logic/algorithm
  - "memory_fix": Memory-related fix (e.g., buffer overflow, use-after-free)
  - "none": No error handling pattern detected

Provide ONLY valid JSON, no additional text."""
        
        try:
            response = self.call_llm(prompt, logger=logger)
            
            # Parse JSON response
            import json
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                return PatchSemanticClassification(
                    patch_intent=parsed.get('patch_intent', 'unclear'),
                    assumed_fault_location=parsed.get('assumed_fault_location', 'unclear'),
                    fix_scope=parsed.get('fix_scope', 'local'),
                    error_handling_pattern=parsed.get('error_handling_pattern', 'none'),
                    confidence=float(parsed.get('confidence', 0.5)),
                    reasoning=parsed.get('reasoning', '')
                )
            else:
                if logger:
                    logger.warning(f"Could not parse JSON from patch semantic classification response")
                return None
        except Exception as e:
            if logger:
                logger.warning(f"Error in patch semantic classification: {e}")
            return None
    
    def _attribute_stack_trace_frames(self, stack_trace: str, project_name: str, top_k: int = 10,
                                      logger: Optional[logging.Logger] = None) -> Optional[FrameAttribution]:
        """
        🔴 2순위 개선: LLM-based Frame Attribution
        
        Stack Trace의 상위 K개 프레임만 LLM에게 물어봐서 logical owner를 결정합니다.
        휴리스틱 COC와 결합하여 사용 (full replacement 아님 → 보수적 + 안전).
        
        Args:
            stack_trace: Full stack trace
            project_name: Project name for context
            top_k: Number of top frames to analyze (default: 10)
            logger: Optional logger
            
        Returns:
            FrameAttribution 객체 또는 None (실패 시)
        """
        if not stack_trace:
            return None
        
        # Extract top K frames
        lines = stack_trace.split('\n')
        top_frames = []
        frame_count = 0
        for line in lines[:50]:  # Check first 50 lines
            if re.search(r'(/src/[^\s:]+):(\d+)', line):
                top_frames.append(line.strip())
                frame_count += 1
                if frame_count >= top_k:
                    break
        
        if not top_frames:
            return None
        
        frames_text = '\n'.join(top_frames[:top_k])
        
        prompt = f"""Analyze the following stack trace frames and determine the logical owner of this crash.

Project: {project_name}
Top {len(top_frames)} Stack Trace Frames:
{frames_text}

Determine the logical owner of this crash. Consider:
- Wrapper/adapter/glue code patterns
- Template/inline/macro code
- Generated code
- Which component is actually responsible for the crash logic?

Provide your analysis in the following JSON format:

{{
    "logical_owner": "dependency_name_or_main_project" | null,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation of why this is the logical owner"
}}

Guidelines:
- logical_owner: The component/library that logically owns this crash
  - Use dependency name (e.g., "libjxl", "libpng") if crash originates in dependency
  - Use project name if crash originates in main project
  - null if unclear
  
- confidence: How confident are you in this attribution?
  - 0.9-1.0: Very clear ownership (e.g., all frames in dependency decoder)
  - 0.7-0.9: Clear ownership with some wrapper code
  - 0.5-0.7: Ambiguous (wrapper/adapter code present)
  - <0.5: Unclear ownership
  
- reason: Explain your reasoning (e.g., "frames show decoder internals, caller is thin wrapper")

Provide ONLY valid JSON, no additional text."""
        
        try:
            response = self.call_llm(prompt, logger=logger)
            
            # Parse JSON response
            import json
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                logical_owner = parsed.get('logical_owner')
                if logical_owner == 'null' or logical_owner is None:
                    logical_owner = None
                
                return FrameAttribution(
                    logical_owner=logical_owner,
                    confidence=float(parsed.get('confidence', 0.5)),
                    reason=parsed.get('reason', ''),
                    top_frames_analyzed=len(top_frames)
                )
            else:
                if logger:
                    logger.warning(f"Could not parse JSON from frame attribution response")
                return None
        except Exception as e:
            if logger:
                logger.warning(f"Error in frame attribution: {e}")
            return None
    
    def _format_patch_semantic_info(self, feature: VulnerabilityFeatures) -> str:
        """Format patch semantic classification info for prompts"""
        if not feature.patch_semantic_classification:
            return ""
        ps = feature.patch_semantic_classification
        return f"""
**LLM Patch Semantic Classification (Semantic Evidence Extractor):**
- Patch Intent: {ps.patch_intent} (workaround | real_fix | refactor | version_bump)
- Assumed Fault Location: {ps.assumed_fault_location} (dependency | main | unclear)
- Fix Scope: {ps.fix_scope} (local | cross_module)
- Error Handling Pattern: {ps.error_handling_pattern} (added_check | changed_logic | memory_fix | none)
- Confidence: {ps.confidence:.2f}
- Reasoning: {ps.reasoning}

**Usage Guidelines:**
- If patch_intent == "workaround" → This suggests Rule 1 penalty should be strengthened
- If assumed_fault_location == "dependency" → This supports Rule 2 amplification
- LLM provides semantic evidence only (no scores) - use this to guide heuristic rules
"""
    
    def _format_frame_attribution_info(self, feature: VulnerabilityFeatures) -> str:
        """Format frame attribution info for prompts"""
        if not feature.frame_attribution:
            return ""
        fa = feature.frame_attribution
        return f"""
**LLM Frame Attribution (Top-{fa.top_frames_analyzed} frames analyzed):**
- Logical Owner: {fa.logical_owner or 'unclear'}
- Confidence: {fa.confidence:.2f}
- Reasoning: {fa.reason}

**Usage Guidelines:**
- Use as COC prior for Rule 2 (combine with heuristic COC: average or max)
- Conservative approach: Use as supporting evidence, not full replacement
- Helps identify wrapper/adapter/glue code patterns that path-based matching misses
"""
    
    def _compute_patch_crash_distance(self, stack_trace: str, patched_files: List[str], 
                                     project_name: str = '') -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Compute structural distance between patch location and crash location.
        Based on 02_build_ground_truth.py logic.
        
        Returns:
            (distance, crash_module, patched_module):
            - distance: 0 (same file/function), 1 (same module), 2 (different modules), 3+ (wrapper/validation)
            - crash_module: Module name where crash occurred
            - patched_module: Module name where patch was applied
        """
        if not stack_trace or not patched_files:
            return None, None, None
        
        # Extract crash location from stack trace (top frame)
        crash_file_path = None
        crash_module = None
        lines = stack_trace.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            match = re.search(r'(/src/[^\s:]+):(\d+)', line)
            if match:
                crash_file_path = match.group(1)
                # Extract module name (project name from path)
                parts = crash_file_path.split('/')
                if len(parts) >= 3:
                    crash_module = parts[2]  # /src/{module}/...
                break
        
        if not crash_file_path:
            return None, None, None
        
        # Extract patched module from patched files
        patched_module = None
        main_project_path = f"/src/{project_name}" if project_name else None
        
        for patched_file in patched_files:
            if patched_file.startswith('/src/'):
                parts = patched_file.split('/')
                if len(parts) >= 3:
                    patched_module = parts[2]
                    break
            elif main_project_path:
                # Relative path, assume main project
                patched_module = project_name
        
        if not patched_module:
            return None, crash_module, None
        
        # Calculate distance
        distance = None
        
        # Check if same file
        for patched_file in patched_files:
            if crash_file_path == patched_file or crash_file_path.endswith(patched_file) or patched_file.endswith(crash_file_path):
                distance = 0
                break
        
        if distance is None:
            # Check if same module
            if crash_module and patched_module and crash_module.lower() == patched_module.lower():
                distance = 1
            else:
                # Different modules
                # Check if patched file is in main project but crash is in dependency (or vice versa)
                if main_project_path:
                    crash_in_main = crash_file_path.startswith(main_project_path)
                    patched_in_main = any(pf.startswith(main_project_path) if pf.startswith('/src/') else True for pf in patched_files)
                    
                    if crash_in_main != patched_in_main:
                        # One is in main project, other is in dependency
                        distance = 2
                    else:
                        # Both in same category, but different modules
                        distance = 2
                else:
                    distance = 2
        
        return distance, crash_module, patched_module
    
    def _analyze_patch_semantic_type(self, patch_diff: str, patched_files: List[str]) -> Optional[str]:
        """
        Analyze patch semantic type from patch diff.
        Returns: VALIDATION_ONLY, ALGORITHM_CHANGE, REFACTOR, etc.
        """
        if not patch_diff:
            return None
        
        patch_lower = patch_diff.lower()
        
        # Check for validation-only patterns
        validation_patterns = ['null check', 'null pointer', 'buffer size', 'bound check', 
                              'range check', 'validation', 'assert', 'if (!', 'if (', 'guard']
        if any(pattern in patch_lower for pattern in validation_patterns):
            # Check if it's mostly validation
            if patch_diff.count('+') < 10:  # Small patch, likely validation
                return "VALIDATION_ONLY"
        
        # Check for algorithm changes
        algorithm_patterns = ['algorithm', 'logic', 'calculation', 'compute', 'calculate']
        if any(pattern in patch_lower for pattern in algorithm_patterns):
            return "ALGORITHM_CHANGE"
        
        # Check for refactoring
        refactor_patterns = ['refactor', 'rename', 'extract', 'move', 'restructure']
        if any(pattern in patch_lower for pattern in refactor_patterns):
            return "REFACTOR"
        
        # Default: assume code fix
        return "CODE_FIX"
    
    def _is_control_flow_only_patch(self, patch_diff: str) -> Optional[bool]:
        """
        Check if patch only adds control flow (if statements, early returns, etc.)
        without modifying core logic.
        """
        if not patch_diff:
            return None
        
        patch_lower = patch_diff.lower()
        
        # Count control flow additions
        control_flow_keywords = ['if (', 'if(', 'return', 'goto', 'break', 'continue', 'assert']
        control_flow_count = sum(patch_lower.count(keyword) for keyword in control_flow_keywords)
        
        # Count other code additions (assignments, function calls, etc.)
        other_code_patterns = ['=', '(', '->', '.', '++', '--']
        other_code_count = sum(patch_lower.count(pattern) for pattern in other_code_patterns)
        
        # If mostly control flow, likely control-flow-only
        if control_flow_count > 0 and other_code_count < control_flow_count * 2:
            return True
        
        return False
    
    def _summarize_stack_trace(self, stack_trace: str, project_name: str, logger: Optional[logging.Logger] = None) -> str:
        """Summarize stack trace"""
        if not stack_trace:
            return "No stack trace available"
        
        # Use full stack trace and let LLM summarize
        prompt = f"""Analyze the following stack trace and summarize the key vulnerability characteristics:

Project: {project_name}
Stack Trace:
{stack_trace}

Provide a concise summary focusing on:
1. Where the crash occurs (function and file)
2. The call chain leading to the crash
3. Any patterns or notable characteristics"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _summarize_patch(self, patch_diff: str, patch_info: Dict, logger: Optional[logging.Logger] = None) -> str:
        """Summarize patch"""
        if not patch_diff:
            return "No patch diff available"
        
        fix_commit = patch_info.get('fix_commit', '')
        submodule_bug = patch_info.get('submodule_bug', False)
        
        # Estimate tokens: roughly 1 token = 4 characters for code/diff
        # Reserve ~10,000 tokens for prompt, system message, and response overhead
        # Max ~190,000 tokens for patch diff to stay under 200k total
        # That's roughly 760,000 characters, but be more conservative
        # Use 600,000 chars to account for prompt text and safety margin
        MAX_PATCH_CHARS = 600000  # Conservative limit to stay under 200k tokens
        
        patch_to_use = patch_diff
        if len(patch_diff) > MAX_PATCH_CHARS:
            # Truncate but try to keep it meaningful
            # Keep the beginning (usually has commit info and key changes)
            # and a sample from the middle/end
            first_part = patch_diff[:MAX_PATCH_CHARS // 2]
            last_part = patch_diff[-(MAX_PATCH_CHARS // 2):]
            patch_to_use = f"{first_part}\n\n[... {len(patch_diff) - MAX_PATCH_CHARS:,} characters truncated ...]\n\n{last_part}"
            if logger:
                logger.warning(f"Patch diff too large ({len(patch_diff):,} chars), truncated to {len(patch_to_use):,} chars for localId")
            else:
                print(f"[!] Warning: Patch diff too large ({len(patch_diff):,} chars), truncated to {len(patch_to_use):,} chars")
        
        # Use patch diff (possibly truncated) and let LLM summarize
        prompt = f"""Analyze the following patch diff and summarize what was fixed:

Fix Commit: {fix_commit}
Submodule Bug: {submodule_bug}
Patch Diff:
{patch_to_use}

Provide a concise summary focusing on:
1. What files were changed
2. What the fix does
3. Whether this is a dependency update or main project fix"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _is_internal_module_by_path(self, dep_path: str, project_name: Optional[str] = None) -> bool:
        """Determine if a dependency is an internal module based on path structure
        
        Path-based filtering (more fundamental than keyword-based):
        1. If dep_path == /src/{project_name} -> main project (exclude)
        2. If dep_path starts with /src/{project_name}/ -> internal module (exclude)
        3. Otherwise -> external dependency (include)
        
        Args:
            dep_path: Dependency path from srcmap (e.g., "/src/testsuite" or "/src/upx/testsuite")
            project_name: Project name (e.g., "upx")
            
        Returns:
            True if internal module, False if external dependency
        """
        if not dep_path or dep_path == 'N/A':
            return False
            
        dep_path_normalized = dep_path.rstrip('/')
        
        # Determine main project path
        if not project_name:
            return False
            
        main_project_path = f"/src/{project_name}"
        main_project_path_normalized = main_project_path.rstrip('/')
        
        # Rule 1: Exact match with main project path -> main project (exclude)
        if dep_path_normalized == main_project_path_normalized:
            return True
            
        # Rule 2: Path starts with main project path + "/" -> internal module (exclude)
        # e.g., /src/upx/testsuite, /src/imagemagick/magickcore
        if dep_path_normalized.startswith(main_project_path_normalized + '/'):
            return True
            
        # Rule 3: External dependency (include)
        return False
    
    def _is_fuzzer_tool(self, dep_name: str) -> bool:
        """Check if dependency is a fuzzing tool (not a runtime dependency)
        
        Fuzzer tools should be excluded regardless of path, as they are build-time
        testing tools, not runtime dependencies.
        """
        if not dep_name or dep_name == 'N/A':
            return False
            
        dep_name_lower = dep_name.lower()
        FUZZER_KEYWORDS = {
            'aflplusplus', 'afl', 'libfuzzer', 'fuzzer', 'fuzz-targets', 
            'fuzzing-headers', 'cryptofuzz', 'honggfuzz', 'fuzz', 'fuzzing'
        }
        return any(fuzzer in dep_name_lower for fuzzer in FUZZER_KEYWORDS)
    
    def _is_test_framework(self, dep_name: str, dep_path: str) -> bool:
        """Check if dependency is a test framework (not a runtime dependency)
        
        Test frameworks are build-time/testing tools, not runtime dependencies.
        Common patterns: testsuite, testsuite2, testsuite3, test-*, *-test, etc.
        """
        if not dep_name or dep_name == 'N/A':
            return False
            
        dep_name_lower = dep_name.lower()
        dep_path_lower = (dep_path or '').lower()
        
        # Check name patterns
        test_patterns = [
            'testsuite', 'test-suite', 'test_suite',
            'testframework', 'test-framework', 'test_framework',
            'unittest', 'testunit'
        ]
        
        # Check if name starts with test- or ends with -test
        if dep_name_lower.startswith('test-') or dep_name_lower.endswith('-test'):
            return True
            
        # Check if name contains test patterns
        if any(pattern in dep_name_lower for pattern in test_patterns):
            return True
            
        # Check path patterns (e.g., /src/testsuite*)
        if dep_path_lower and '/testsuite' in dep_path_lower:
            return True
            
        return False
    
    def _filter_dependencies(self, dependencies: List[Dict], project_name: Optional[str] = None) -> List[Dict]:
        """Filter dependencies using path-based approach (more fundamental than keyword-based)
        
        Args:
            dependencies: List of dependency dicts with 'name', 'path', 'version'
            project_name: Project name for path-based filtering
            
        Returns:
            Filtered list of external dependencies
        """
        filtered = []
        for dep in dependencies:
            dep_name = dep.get('name', 'N/A')
            dep_path = dep.get('path', 'N/A')
            
            # Filter 1: Exclude fuzzer tools (build-time, not runtime)
            if self._is_fuzzer_tool(dep_name):
                continue
            
            # Filter 2: Exclude test frameworks (build-time/testing, not runtime)
            if self._is_test_framework(dep_name, dep_path):
                continue
                
            # Filter 3: Exclude internal modules (path-based)
            if self._is_internal_module_by_path(dep_path, project_name):
                continue
                
            filtered.append(dep)
        return filtered
    
    def _extract_dependencies_from_stack_trace(self, stack_trace: str, srcmap_dependencies: List[Dict], 
                                               project_name: Optional[str] = None, top_n: int = 10) -> List[Dict]:
        """Extract dependencies from stack trace (consistent with GT Rule 2 logic)
        
        This function extracts dependencies that are actually used at runtime based on
        stack trace frames, ensuring consistency with Ground Truth filtering.
        
        Args:
            stack_trace: Stack trace string
            srcmap_dependencies: List of dependencies from srcmap
            project_name: Project name for path-based filtering
            top_n: Number of top frames to analyze
            
        Returns:
            List of dependency dicts that appear in stack trace
        """
        if not stack_trace:
            return []
        
        import re
        from collections import defaultdict
        
        # Extract file paths from stack trace (supports various formats)
        stack_frames = []
        lines = stack_trace.split('\n')
        
        for line in lines:
            # Pattern 1: #0 0x... in function /path/to/file.c:line format
            match = re.search(r'(/src/[^\s:]+):(\d+)', line)
            if match:
                file_path = match.group(1)
                stack_frames.append(file_path)
                continue
            
            # Pattern 2: #0 0x... in function_name /path/to/file.c:line format
            match = re.search(r'in\s+\S+\s+(/src/[^\s:]+):(\d+)', line)
            if match:
                file_path = match.group(1)
                stack_frames.append(file_path)
                continue
            
            # Pattern 3: #0 0x... /path/to/file.c:line format
            match = re.search(r'^\s*#\d+\s+0x[0-9a-fA-F]+\s+(/src/[^\s:]+):(\d+)', line)
            if match:
                file_path = match.group(1)
                stack_frames.append(file_path)
                continue
        
        if len(stack_frames) == 0:
            return []
        
        if len(stack_frames) < top_n:
            top_n = len(stack_frames)
        
        top_frames = stack_frames[:top_n]
        
        # Find main project path
        main_project_path = None
        if project_name:
            main_project_path = f"/src/{project_name}"
        
        # Map each frame to a dependency (using path-based filtering)
        dependency_mapping = defaultdict(set)  # Use set to avoid duplicates
        
        for file_path in top_frames:
            matched = False
            
            # Step 1: Match with srcmap dependencies first (most accurate)
            for dep in srcmap_dependencies:
                dep_path = dep.get('path', '')
                dep_name = dep.get('name', '')
                
                if dep_path and file_path.startswith(dep_path):
                    # Only map dependencies that are not the main project
                    if dep_path != main_project_path:
                        # Apply path-based filtering (consistent with GT)
                        if main_project_path and dep_path.startswith(main_project_path + '/'):
                            # Internal module - skip
                            continue
                        
                        # Apply fuzzer/test framework filtering
                        if self._is_fuzzer_tool(dep_name) or self._is_test_framework(dep_name, dep_path):
                            continue
                        
                        # Valid external dependency
                        dependency_mapping[dep_name].add(file_path)
                        matched = True
                        break
            
            # Step 2: Submodule detection (submodules within main project path)
            if not matched and main_project_path and file_path.startswith(main_project_path):
                remaining_path = file_path[len(main_project_path):]
                if remaining_path.startswith('/'):
                    remaining_path = remaining_path[1:]
                
                parts = remaining_path.split('/')
                if len(parts) >= 1:
                    potential_submodule = parts[0]
                    submodule_path = f"{main_project_path}/{potential_submodule}"
                    
                    # Check if this submodule is registered as dependency in srcmap
                    for dep in srcmap_dependencies:
                        dep_path = dep.get('path', '')
                        if dep_path == submodule_path or dep_path.endswith(f'/{potential_submodule}'):
                            dep_name = dep.get('name', '')
                            if not self._is_fuzzer_tool(dep_name) and not self._is_test_framework(dep_name, dep_path):
                                dependency_mapping[dep_name].add(file_path)
                                matched = True
                                break
            
            # Step 3: Extract project name (last resort)
            if not matched and file_path.startswith('/src/'):
                # Path-based check: skip if in main project path
                if main_project_path and file_path.startswith(main_project_path):
                    continue
                
                parts = file_path.split('/')
                if len(parts) >= 3:
                    potential_project = parts[2]
                    # Check if this project is in srcmap dependencies
                    for dep in srcmap_dependencies:
                        dep_name = dep.get('name', '')
                        dep_path = dep.get('path', '')
                        if dep_name.lower() == potential_project.lower() or dep_path.endswith(f'/{potential_project}'):
                            if not self._is_fuzzer_tool(dep_name) and not self._is_test_framework(dep_name, dep_path):
                                dependency_mapping[dep_name].add(file_path)
                                break
        
        # Convert to list of dependency dicts (with file paths for reference)
        result = []
        seen_deps = set()
        for dep_name, file_paths in dependency_mapping.items():
            if dep_name in seen_deps:
                continue
            seen_deps.add(dep_name)
            
            # Find full dependency info from srcmap
            dep_info = None
            for dep in srcmap_dependencies:
                if dep.get('name', '').lower() == dep_name.lower():
                    dep_info = dep.copy()
                    break
            
            if dep_info:
                result.append(dep_info)
            else:
                # Fallback: create minimal dict from extracted info
                result.append({
                    'name': dep_name,
                    'path': f"/src/{dep_name}",
                    'version': 'N/A'
                })
        
        return result
    
    def _summarize_dependencies(self, srcmap: Dict, project_name: Optional[str] = None, 
                                stack_trace: Optional[str] = None, logger: Optional[logging.Logger] = None) -> str:
        """Summarize dependencies using path-based filtering and stack trace analysis
        
        This function combines:
        1. srcmap dependencies (build-time dependencies) - filtered
        2. stack trace dependencies (runtime dependencies) - extracted from actual execution
        
        This ensures consistency with Ground Truth filtering approach.
        """
        vulnerable_deps = srcmap.get('vulnerable_version', {}).get('dependencies', [])
        
        if not vulnerable_deps:
            return "No dependencies found"
        
        # Apply path-based filtering to srcmap dependencies
        filtered_srcmap_deps = self._filter_dependencies(vulnerable_deps, project_name)
        
        # Extract dependencies from stack trace (if available)
        stack_trace_deps = []
        if stack_trace:
            stack_trace_deps = self._extract_dependencies_from_stack_trace(
                stack_trace, vulnerable_deps, project_name, top_n=10
            )
        
        # Combine: prioritize stack trace dependencies (actual runtime usage)
        # but also include srcmap dependencies for context
        combined_deps = {}
        
        # First, add stack trace dependencies (runtime - higher priority)
        for dep in stack_trace_deps:
            dep_name = dep.get('name', 'N/A')
            combined_deps[dep_name] = {
                'dep': dep,
                'source': 'runtime'  # From stack trace
            }
        
        # Then, add srcmap dependencies (build-time - for context)
        for dep in filtered_srcmap_deps:
            dep_name = dep.get('name', 'N/A')
            if dep_name not in combined_deps:
                combined_deps[dep_name] = {
                    'dep': dep,
                    'source': 'build-time'  # From srcmap
                }
        
        if not combined_deps:
            return "No external dependencies found (all filtered as internal modules or fuzzer tools)"
        
        # Prepare dependency info - use all dependencies and let LLM summarize
        deps_info = []
        runtime_deps = []
        buildtime_deps = []
        
        for dep_name, info in combined_deps.items():  # Use all dependencies
            dep = info['dep']
            source = info['source']
            dep_info_str = f"- {dep.get('name', 'N/A')} (path: {dep.get('path', 'N/A')}, version: {dep.get('version', 'N/A')})"
            
            if source == 'runtime':
                runtime_deps.append(dep_info_str)
            else:
                buildtime_deps.append(dep_info_str)
            
            deps_info.append(dep_info_str)
        
        deps_text = '\n'.join(deps_info)
        
        # Add source information if available
        source_info = ""
        if runtime_deps:
            source_info += f"\n\nRuntime dependencies (from stack trace - actually used):\n" + '\n'.join(runtime_deps)
        if buildtime_deps:
            source_info += f"\n\nBuild-time dependencies (from srcmap - may not be used):\n" + '\n'.join(buildtime_deps)
        
        prompt = f"""Analyze the following dependencies and summarize key characteristics:

Dependencies:
{deps_text}{source_info}

Provide a concise summary focusing on:
1. Main dependencies used (especially runtime dependencies from stack trace)
2. Any notable versions or patterns
3. Potential security-relevant dependencies
4. Distinguish between runtime dependencies (actually used) and build-time dependencies (may not be used)"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _summarize_code_snippets(self, code_snippets: Dict) -> str:
        """Summarize code snippets"""
        if not code_snippets:
            return "No code snippets available"
        
        main_project_snippets = code_snippets.get('main_project', {}).get('snippets', [])
        dependency_snippets = code_snippets.get('dependencies', {})
        
        snippet_count = len(main_project_snippets) + sum(len(d.get('snippets', [])) for d in dependency_snippets.values())
        
        return f"Code snippets available: {len(main_project_snippets)} main project, {sum(len(d.get('snippets', [])) for d in dependency_snippets.values())} dependency snippets"
    
    def _generate_reasoning_summary(self, project_name: str, bug_type: str, severity: str,
                                   stack_trace_summary: str, patch_summary: str,
                                   dependencies_summary: str, code_snippets_summary: str,
                                   logger: Optional[logging.Logger] = None) -> str:
        """Generate LLM reasoning summary (Chain-of-Thought)"""
        
        prompt = f"""Using Chain-of-Thought reasoning, analyze the following vulnerability information and explain your reasoning process:

Project: {project_name}
Bug Type: {bug_type}
Severity: {severity}

Stack Trace Summary:
{stack_trace_summary}

Patch Summary:
{patch_summary}

Dependencies Summary:
{dependencies_summary}

Code Snippets Summary:
{code_snippets_summary}

Provide a detailed reasoning process that:
1. Identifies key vulnerability characteristics
2. Explains why certain features are important
3. Suggests potential root causes (main project vs dependency)
4. Provides evidence-based reasoning"""
        
        return self.call_llm(prompt, logger=logger)

    # ========================================================================
    # Module 1: Bug Type Group Analysis (Macro-level Pattern Recognition)
    # ========================================================================
    
    def analyze_bug_type_groups(self, features_list: List[VulnerabilityFeatures],
                               logger: Optional[logging.Logger] = None) -> List[BugTypeGroupInfo]:
        """
        Module 1: Bug Type Group Analysis
        
        Groups vulnerabilities by bug_type and performs macro-level pattern recognition.
        This establishes the foundation for efficient analysis paths when the benchmark
        collects large volumes of new vulnerability tasks.
        
        Args:
            features_list: List of VulnerabilityFeatures
            logger: Optional logger for logging
        
        Returns:
            List of BugTypeGroupInfo objects
        """
        print(f"[Module 1] Analyzing bug type groups for {len(features_list)} vulnerabilities...")
        
        # Group by bug_type
        features_by_bug_type = defaultdict(list)
        for f in features_list:
            features_by_bug_type[f.bug_type].append(f)
        
        bug_type_groups = []
        
        for bug_type, type_features in features_by_bug_type.items():
            localIds = [f.localId for f in type_features]
            
            print(f"  [*] Analyzing bug type group: {bug_type} ({len(type_features)} vulnerabilities)")
            
            # Extract common dependencies from srcmap
            common_deps = self._extract_common_dependencies(type_features)
            
            # Generate group embedding (average of individual embeddings)
            group_embedding = None
            valid_embeddings = [f.semantic_embedding for f in type_features 
                              if f.semantic_embedding and isinstance(f.semantic_embedding, list)]
            if valid_embeddings:
                group_embedding = np.mean(valid_embeddings, axis=0).tolist()
            
            # LLM analysis for macro-level pattern recognition
            llm_summary = self._generate_bug_type_group_summary(
                bug_type, type_features, common_deps, logger
            )
            
            # Parse LLM response to extract summary, patterns, and confidence
            parsed_result = self._parse_bug_type_group_analysis(llm_summary)
            
            bug_type_group = BugTypeGroupInfo(
                bug_type=bug_type,
                localIds=localIds,
                bug_type_group_summary=parsed_result.get('summary', llm_summary),
                common_dependencies_in_group=common_deps,
                initial_pattern_observation=parsed_result.get('pattern_observation', ''),
                group_embedding=group_embedding,
                confidence_score=parsed_result.get('confidence', 0.0),
                individual_root_causes={}
            )
            
            # Perform individual root cause inference for each vulnerability
            print(f"    [*] Performing individual root cause inference for {len(type_features)} vulnerabilities...")
            for feature in type_features:
                try:
                    individual_rc = self.infer_individual_root_cause(feature, bug_type_group, logger)
                    bug_type_group.individual_root_causes[feature.localId] = individual_rc
                except Exception as e:
                    if logger:
                        logger.error(f"Error inferring individual root cause for localId {feature.localId}: {e}")
                    print(f"      [-] Error inferring individual root cause for localId {feature.localId}: {e}")
            
            print(f"    [+] Completed individual inference for {len(bug_type_group.individual_root_causes)}/{len(type_features)} vulnerabilities")
            bug_type_groups.append(bug_type_group)
        
        print(f"  [+] Analyzed {len(bug_type_groups)} bug type groups")
        return bug_type_groups
    
    def _generate_bug_type_group_summary(self, bug_type: str,
                                        type_features: List[VulnerabilityFeatures],
                                        common_deps: List[str],
                                        logger: Optional[logging.Logger] = None) -> str:
        """Generate LLM summary for bug type group"""
        
        # Prepare summarized information for Module 1 (token-efficient)
        features_summary = []
        for f in type_features[:20]:  # Limit to first 20 for token efficiency
            features_summary.append(f"""
localId {f.localId} (Project: {f.project_name}):
- Stack Trace Summary: {f.stack_trace_summary[:200]}
- Patch Summary: {f.patch_summary[:200]}
- Dependencies Summary: {f.dependencies_summary[:150]}
- Initial Reasoning: {f.llm_reasoning_summary[:150]}...
""")
        
        prompt = f"""Perform macro-level pattern recognition for this bug type group.

Bug Type: {bug_type}
Number of vulnerabilities: {len(type_features)}
Common dependencies: {', '.join(common_deps) if common_deps else 'None'}

Vulnerability Summaries (using summarized information for token efficiency):
{chr(10).join(features_summary)}

Analyze:
1. Where do the crashes consistently occur (main project code vs dependency code)?
2. What do the stack traces reveal about the crash location?
3. What do the patches fix and where are the fixes applied?
4. Are there patterns within this bug type group?
5. Which dependencies should be prioritized for attention in Module 2?

Provide your analysis in the following format:
BUG_TYPE_GROUP_SUMMARY: [Summary of the bug type group's general characteristics]
INITIAL_PATTERN_OBSERVATION: [Key patterns observed in this group]
COMMON_DEPENDENCIES_PRIORITY: [List of dependencies that should be prioritized in Module 2]
MODULE1_CONFIDENCE: [0.0-1.0 confidence score for this analysis]
"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _parse_bug_type_group_analysis(self, llm_response: str) -> Dict:
        """Parse Module 1 LLM response"""
        result = {
            'summary': '',
            'pattern_observation': '',
            'confidence': 0.0
        }
        
        lines = llm_response.split('\n')
        current_section = None
        
        for line in lines:
            line_upper = line.upper()
            if 'BUG_TYPE_GROUP_SUMMARY:' in line_upper:
                current_section = 'summary'
                result['summary'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'INITIAL_PATTERN_OBSERVATION:' in line_upper:
                current_section = 'pattern'
                result['pattern_observation'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'MODULE1_CONFIDENCE:' in line_upper:
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    result['confidence'] = float(conf_str)
                except:
                    pass
            elif current_section == 'summary' and line.strip():
                result['summary'] += ' ' + line.strip()
            elif current_section == 'pattern' and line.strip():
                result['pattern_observation'] += ' ' + line.strip()
        
        return result
    
    def infer_individual_root_cause(self, feature: VulnerabilityFeatures, 
                                   bug_type_group: Optional[BugTypeGroupInfo] = None,
                                   logger: Optional[logging.Logger] = None) -> IndividualRootCause:
        """
        Module 1: Individual Root Cause Inference
        
        Performs GT-independent semantic analysis for a single vulnerability.
        Addresses Gap 1: Semantic Reasoning (workaround detection, signal conflict resolution, intent analysis).
        
        Args:
            feature: VulnerabilityFeatures to analyze
            bug_type_group: Optional BugTypeGroupInfo for context
            logger: Optional logger for logging
        
        Returns:
            IndividualRootCause object
        """
        # Use deterministic workaround detection if clear
        use_deterministic = False
        if (feature.patch_crash_distance is not None and 
            feature.patch_crash_distance >= 2 and
            feature.crash_module and feature.patched_module and
            feature.crash_module != feature.patched_module and
            feature.control_flow_only):
            use_deterministic = True
        
        # Generate LLM prompt for individual root cause inference
        prompt = self._generate_individual_root_cause_prompt(feature, bug_type_group, use_deterministic)
        
        # Call LLM
        llm_response = self.call_llm(prompt, logger=logger)
        
        # Parse LLM response
        individual_rc = self._parse_individual_root_cause_response(llm_response, feature.localId)
        
        return individual_rc
    
    def _generate_individual_root_cause_prompt(self, feature: VulnerabilityFeatures,
                                               bug_type_group: Optional[BugTypeGroupInfo] = None,
                                               use_deterministic_workaround: bool = False) -> str:
        """Generate prompt for individual root cause inference"""
        
        # Get submodule_bug from database
        submodule_bug = None
        try:
            conn = sqlite3.connect(DB_PATH)
            try:
                cursor = conn.execute("SELECT submodule_bug FROM arvo WHERE localId = ?", (feature.localId,))
                row = cursor.fetchone()
                if row:
                    submodule_bug = bool(row[0])
            except:
                pass
            finally:
                conn.close()
        except:
            pass
        
        # Structural features summary
        structural_features = []
        if feature.patch_crash_distance is not None:
            structural_features.append(f"- Patch-Crash Distance: {feature.patch_crash_distance} (0=same file, 1=same module, 2=different modules, 3+=wrapper)")
        if feature.patch_semantic_type:
            structural_features.append(f"- Patch Semantic Type: {feature.patch_semantic_type}")
        if feature.crash_module:
            structural_features.append(f"- Crash Module: {feature.crash_module}")
        if feature.patched_module:
            structural_features.append(f"- Patched Module: {feature.patched_module}")
        if feature.control_flow_only is not None:
            structural_features.append(f"- Control Flow Only: {feature.control_flow_only}")
        if feature.workaround_detected is not None:
            structural_features.append(f"- Workaround Detected (heuristic): {feature.workaround_detected}")
        if submodule_bug is not None:
            structural_features.append(f"- Submodule Bug Flag: {submodule_bug} {'(STRONG indicator of Dependency_Specific)' if submodule_bug else '(NOT a submodule bug)'}")
        
        structural_summary = '\n'.join(structural_features) if structural_features else "None available"
        
        # Bug type group context
        group_context = ""
        if bug_type_group:
            group_context = f"""
Bug Type Group Context:
- Bug Type: {bug_type_group.bug_type}
- Common Dependencies: {', '.join(bug_type_group.common_dependencies_in_group[:5]) if bug_type_group.common_dependencies_in_group else 'None'}
- Pattern Observation: {bug_type_group.initial_pattern_observation[:200] if bug_type_group.initial_pattern_observation else 'None'}
"""
        
        prompt = f"""You are an expert in vulnerability analysis. Analyze the root cause of this vulnerability independently (ignore any ground truth labels).

**Task:**
Determine if the bug is in the main project code or in an external dependency.

**Vulnerability Information:**
- LocalId: {feature.localId}
- Project: {feature.project_name}
- Bug Type: {feature.bug_type}
- Severity: {feature.severity}

**Stack Trace Summary:**
{feature.stack_trace_summary[:500]}

**Patch Summary:**
{feature.patch_summary[:500]}

**Dependencies Summary:**
{feature.dependencies_summary[:300]}

**Structural Features (Quantitative Evidence):**
{structural_summary}

{self._format_patch_semantic_info(feature)}
{self._format_frame_attribution_info(feature)}

{group_context}

**Critical Questions to Answer:**
1. **Workaround Detection**: Is this a workaround patch or a real fix?
   - Workaround: Main project adds defensive code because it cannot fix third-party dependency
   - Real Fix: Main project fixes its own bug
   - Evidence: patch-crash distance, module mismatch, patch semantic type, control-flow-only

2. **Signal Conflict Resolution**: Which signal is most reliable?
   - Stack trace location (crash location) - usually most reliable
   - Patch location - may be misleading if workaround
   - Commit message - often vague
   - Assessment: Consider patch-crash distance and module mismatch

3. **Intent Analysis**: What does the patch actually do?
   - Logic fix: Corrects algorithm or logic error
   - Defensive validation: Adds null checks, bounds checks, etc. (workaround pattern)
   - Refactoring: Code restructuring without fixing bug

**Decision Tree:**
1. Check submodule_bug flag: If True, this is STRONG indicator of Dependency_Specific
2. Check stack trace: Where does crash occur? (main project vs dependency)
   - If submodule_bug=False and crash in /src/ path, it may be bundled dependency or main project module (NOT submodule)
   - If submodule_bug=True and crash in /src/ path, it is actual submodule
3. Check patch location: Where is patch applied? (main project vs dependency)
4. Check patch-crash distance: If distance >= 2 AND module mismatch → likely workaround
5. Check patch semantic type: VALIDATION_ONLY → likely workaround
6. If signals conflict, trust stack trace location (crash location) over patch location

**Dependency Naming Requirements:**
- Use exact dependency name from srcmap or dependencies summary
- Examples: "libpng", "openssl", "libjxl", "react-native"
- Do NOT use generic names like "library", "dependency", "external"
- If Main_Project_Specific, set dependency to null

**Output Format (JSON):**
{{
  "root_cause_type": "Main_Project_Specific" | "Dependency_Specific",
  "root_cause_dependency": "dependency_name" | null,
  "patch_intent": "ACTUAL_FIX" | "WORKAROUND" | "DEFENSIVE",
  "is_workaround_patch": true | false,
  "patch_semantic_llm_opinion": "description",
  "main_project_score": 0.0-1.0,
  "dependency_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reasoning": "Step-by-step explanation...",
  "evidence": "Key evidence supporting the decision"
}}

**Important Notes:**
- This is GT-independent analysis (do not use ground truth labels)
- Focus on semantic understanding (workaround vs real fix)
- Use quantitative evidence (structural features) to support reasoning
- If workaround detected deterministically, explain why
"""
        
        if use_deterministic_workaround:
            prompt += "\n**Deterministic Workaround Detected:** Patch-crash distance >= 2, module mismatch, and control-flow-only. This is a clear workaround pattern."
        
        return prompt
    
    def _parse_individual_root_cause_response(self, llm_response: str, localId: int) -> IndividualRootCause:
        """Parse LLM response for individual root cause inference"""
        
        # Initialize defaults
        root_cause_type = "Unknown"
        root_cause_dependency = None
        patch_intent = None
        is_workaround_patch = None  # Initialize is_workaround_patch
        patch_semantic_llm_opinion = None
        main_project_score = 0.0
        dependency_score = 0.0
        confidence = 0.0
        reasoning = ""
        evidence = ""
        
        # Try to parse JSON from response
        import json
        try:
            # Find JSON block in response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                root_cause_type = parsed.get('root_cause_type', 'Unknown')
                root_cause_dependency = parsed.get('root_cause_dependency')
                patch_intent = parsed.get('patch_intent')
                # Parse is_workaround_patch (can be boolean or derived from patch_intent)
                is_workaround_patch = parsed.get('is_workaround_patch')
                if is_workaround_patch is None and patch_intent:
                    # Derive from patch_intent if not explicitly provided
                    is_workaround_patch = (patch_intent in ['WORKAROUND', 'DEFENSIVE'])
                patch_semantic_llm_opinion = parsed.get('patch_semantic_llm_opinion')
                main_project_score = float(parsed.get('main_project_score', 0.0))
                dependency_score = float(parsed.get('dependency_score', 0.0))
                confidence = float(parsed.get('confidence', 0.0))
                reasoning = parsed.get('reasoning', '')
                evidence = parsed.get('evidence', '')
        except Exception as e:
            # Fallback: parse from text
            lines = llm_response.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'root_cause_type' in line_lower:
                    if 'main_project_specific' in line_lower:
                        root_cause_type = 'Main_Project_Specific'
                    elif 'dependency_specific' in line_lower:
                        root_cause_type = 'Dependency_Specific'
                elif 'root_cause_dependency' in line_lower and ':' in line:
                    dep = line.split(':', 1)[1].strip().strip('"').strip("'")
                    if dep.lower() not in ['null', 'none', 'n/a', '']:
                        root_cause_dependency = dep
                elif 'is_workaround_patch' in line_lower and ':' in line:
                    try:
                        val_str = line.split(':', 1)[1].strip().lower()
                        if 'true' in val_str:
                            is_workaround_patch = True
                        elif 'false' in val_str:
                            is_workaround_patch = False
                    except:
                        pass
                elif 'patch_intent' in line_lower and ':' in line:
                    intent_str = line.split(':', 1)[1].strip().upper()
                    patch_intent = intent_str
                    # Derive is_workaround_patch from patch_intent if not set
                    if is_workaround_patch is None and intent_str in ['WORKAROUND', 'DEFENSIVE']:
                        is_workaround_patch = True
                elif 'confidence' in line_lower and ':' in line:
                    try:
                        conf_str = line.split(':', 1)[1].strip()
                        confidence = float(conf_str)
                    except:
                        pass
            
            reasoning = llm_response[:1000]  # Use first 1000 chars as reasoning
        
        # Normalize root_cause_type
        if 'main' in root_cause_type.lower() or 'project' in root_cause_type.lower():
            root_cause_type = 'Main_Project_Specific'
        elif 'dependency' in root_cause_type.lower():
            root_cause_type = 'Dependency_Specific'
        
        return IndividualRootCause(
            localId=localId,
            root_cause_type=root_cause_type,
            root_cause_dependency=root_cause_dependency,
            patch_intent=patch_intent,
            is_workaround_patch=is_workaround_patch,
            patch_semantic_llm_opinion=patch_semantic_llm_opinion,
            main_project_score=main_project_score,
            dependency_score=dependency_score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence
        )
    
    # ========================================================================
    # Module 2: Fine-Grained Sub-Grouping within Bug Type Groups
    # ========================================================================
    
    def perform_fine_grained_sub_grouping(self, bug_type_group: BugTypeGroupInfo,
                                         features_list: List[VulnerabilityFeatures],
                                         logger: Optional[logging.Logger] = None) -> List[SubGroupInfo]:
        """
        Module 2: Fine-Grained Sub-Grouping within Bug Type Groups
        
        Implements hierarchical filtering strategy optimized for gpt-4o-mini (no embeddings/temperature):
        1. Step A: Deterministic structural pattern matching (function names, file names, CWE IDs)
        2. Step B: LLM-based semantic clustering (seed-based incremental clustering)
        
        Args:
            bug_type_group: BugTypeGroupInfo from Module 1
            features_list: List of all VulnerabilityFeatures
            logger: Optional logger for logging
        
        Returns:
            List of SubGroupInfo objects
        """
        print(f"  [*] Performing fine-grained sub-grouping for bug type: {bug_type_group.bug_type}")
        
        # Get features for this bug type group
        group_features = [f for f in features_list if f.localId in bug_type_group.localIds]
        
        if not group_features:
            print(f"    [-] No features found for bug type group")
            return []
        
        # Step A-0: Group by dependency name from individual_root_causes (if available)
        dependency_groups = defaultdict(list)
        if bug_type_group.individual_root_causes:
            print(f"    [*] Step A-0: Grouping by dependency name from individual root causes...")
            for localId in bug_type_group.localIds:
                individual_rc = bug_type_group.individual_root_causes.get(localId)
                if individual_rc and individual_rc.root_cause_dependency:
                    dep_name = individual_rc.root_cause_dependency
                    dependency_groups[dep_name].append(localId)
            print(f"    [+] Step A-0: Created {len(dependency_groups)} dependency-based groups")
        
        # Step A: Deterministic structural pattern matching
        # If dependency groups exist, perform structural grouping within each dependency group
        if dependency_groups:
            print(f"    [*] Step A: Deterministic structural pattern matching within dependency groups...")
            all_structural_groups = []
            all_ungrouped_features = []
            
            for dep_name, dep_localIds in dependency_groups.items():
                dep_features = [f for f in group_features if f.localId in dep_localIds]
                if dep_features:
                    structural_groups, ungrouped_features = self._deterministic_structural_grouping(
                        dep_features, logger
                    )
                    all_structural_groups.extend(structural_groups)
                    all_ungrouped_features.extend(ungrouped_features)
            
            # Also process features without dependency assignment
            no_dep_features = [f for f in group_features 
                             if f.localId not in [lid for dep_localIds in dependency_groups.values() for lid in dep_localIds]]
            if no_dep_features:
                structural_groups, ungrouped_features = self._deterministic_structural_grouping(
                    no_dep_features, logger
                )
                all_structural_groups.extend(structural_groups)
                all_ungrouped_features.extend(ungrouped_features)
            
            structural_groups = all_structural_groups
            ungrouped_features = all_ungrouped_features
        else:
            print(f"    [*] Step A: Deterministic structural pattern matching...")
            structural_groups, ungrouped_features = self._deterministic_structural_grouping(
                group_features, logger
            )
        
        print(f"    [+] Step A: Created {len(structural_groups)} structural groups, {len(ungrouped_features)} ungrouped")
        
        # Step B: LLM-based semantic clustering for ungrouped features
        llm_groups = []
        if ungrouped_features:
            print(f"    [*] Step B: LLM-based semantic clustering for {len(ungrouped_features)} ungrouped vulnerabilities...")
            llm_groups = self._llm_seed_based_clustering(
                bug_type_group, ungrouped_features, logger
            )
            print(f"    [+] Step B: Created {len(llm_groups)} LLM-based groups")
        
        # Combine all groups
        all_sub_groups = structural_groups + llm_groups
        
        # Assign sequential sub_group_id
        for idx, sg in enumerate(all_sub_groups):
            sg.sub_group_id = idx + 1
        
        print(f"    [+] Total: {len(all_sub_groups)} sub-groups identified")
        return all_sub_groups
    
    def _deterministic_structural_grouping(self, features: List[VulnerabilityFeatures],
                                          logger: Optional[logging.Logger] = None) -> tuple[List[SubGroupInfo], List[VulnerabilityFeatures]]:
        """
        Step A: Deterministic structural pattern matching
        
        Groups vulnerabilities by exact matches in:
        - Top function names from stack trace
        - Most patched file names
        - CWE IDs (if specific enough)
        
        Returns:
            Tuple of (structural_groups, ungrouped_features)
        """
        import re
        from collections import defaultdict
        
        # Extract structural signatures for each feature
        feature_signatures = {}
        for f in features:
            sig = self._extract_structural_signature(f)
            feature_signatures[f.localId] = sig
        
        # Group by exact signature match
        signature_to_localIds = defaultdict(list)
        for localId, sig in feature_signatures.items():
            # Create a hashable key from signature
            sig_key = (
                tuple(sorted(sig['top_functions'])) if sig['top_functions'] else None,
                tuple(sorted(sig['patched_files'])) if sig['patched_files'] else None,
                sig['cwe_id'] if sig['cwe_id'] else None
            )
            signature_to_localIds[sig_key].append(localId)
        
        # Create SubGroupInfo for groups with 2+ vulnerabilities
        structural_groups = []
        ungrouped_localIds = set()
        
        for sig_key, localIds in signature_to_localIds.items():
            if len(localIds) >= 2:  # Only group if 2+ vulnerabilities match
                # Get features for this group
                group_features = [f for f in features if f.localId in localIds]
                
                # Create pattern description from signature
                pattern_parts = []
                if sig_key[0]:  # top_functions
                    pattern_parts.append(f"Functions: {', '.join(sig_key[0][:3])}")
                if sig_key[1]:  # patched_files
                    file_names = [f.split('/')[-1] for f in sig_key[1][:2]]
                    pattern_parts.append(f"Files: {', '.join(file_names)}")
                if sig_key[2]:  # cwe_id
                    pattern_parts.append(f"CWE: {sig_key[2]}")
                
                pattern_description = " | ".join(pattern_parts) if pattern_parts else "Structural pattern match"
                
                # Infer root cause (will be refined in Module 3)
                inferred_type = "Unknown"
                inferred_dep = None
                
                # Create SubGroupInfo (sub_group_id will be assigned later)
                sub_group = SubGroupInfo(
                    sub_group_id=0,  # Will be reassigned
                    bug_type_group=group_features[0].bug_type if group_features else "",
                    localIds=localIds,
                    pattern_description=pattern_description,
                    grouping_reasoning=f"Grouped by deterministic structural pattern matching: {pattern_description}. These vulnerabilities share exact matches in function names, patched files, or CWE IDs.",
                    inferred_root_cause_type=inferred_type,
                    inferred_root_cause_dependency=inferred_dep,
                    reasoning=f"Grouped by structural pattern matching: {pattern_description}",
                    confidence_score=0.7  # High confidence for exact structural matches
                )
                structural_groups.append(sub_group)
            else:
                # Single vulnerability - will be handled by LLM clustering
                ungrouped_localIds.update(localIds)
        
        ungrouped_features = [f for f in features if f.localId in ungrouped_localIds]
        
        if logger:
            logger.info(f"Structural grouping: {len(structural_groups)} groups, {len(ungrouped_features)} ungrouped")
        
        return structural_groups, ungrouped_features
    
    def _extract_structural_signature(self, feature: VulnerabilityFeatures) -> dict:
        """
        Extract structural signature from a feature:
        - Top 1-2 function names from stack trace
        - Most patched file names (from patch summary)
        - CWE ID (if specific)
        """
        import re
        
        sig = {
            'top_functions': [],
            'patched_files': [],
            'cwe_id': None
        }
        
        # Extract top functions from stack trace
        stack_trace = feature.stack_trace_summary or ""
        # Look for function names (common patterns: "in function_name", "at function_name", "function_name(")
        function_patterns = [
            r'\bin\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'\bat\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        ]
        
        seen_functions = set()
        for pattern in function_patterns:
            matches = re.finditer(pattern, stack_trace)
            for match in matches:
                func_name = match.group(1)
                # Filter out common system/library functions
                if func_name not in ['malloc', 'free', 'memcpy', 'strcpy', 'strcmp', 'printf', 'fprintf']:
                    if func_name not in seen_functions:
                        sig['top_functions'].append(func_name)
                        seen_functions.add(func_name)
                        if len(sig['top_functions']) >= 2:
                            break
            if len(sig['top_functions']) >= 2:
                break
        
        # Extract patched files from patch summary
        patch_summary = feature.patch_summary or ""
        # Look for file paths (common patterns: "+++ b/path/to/file", "--- a/path/to/file", "/path/to/file")
        file_patterns = [
            r'[+-]{3}\s+[ab]/([^\s\n]+)',
            r'/([a-zA-Z0-9_/]+\.(c|cpp|h|hpp|cc|cxx))',
        ]
        
        seen_files = set()
        for pattern in file_patterns:
            matches = re.finditer(pattern, patch_summary)
            for match in matches:
                file_path = match.group(1)
                # Extract just the filename or last 2 path components
                path_parts = file_path.split('/')
                if len(path_parts) >= 2:
                    file_key = '/'.join(path_parts[-2:])
                else:
                    file_key = path_parts[-1]
                
                if file_key not in seen_files:
                    sig['patched_files'].append(file_key)
                    seen_files.add(file_key)
                    if len(sig['patched_files']) >= 3:
                        break
            if len(sig['patched_files']) >= 3:
                break
        
        # Extract CWE ID if available (from reasoning summary or other fields)
        # This is optional and may not always be available
        
        return sig
    
    def _llm_seed_based_clustering(self, bug_type_group: BugTypeGroupInfo,
                                   ungrouped_features: List[VulnerabilityFeatures],
                                   logger: Optional[logging.Logger] = None) -> List[SubGroupInfo]:
        """
        Step B: LLM-based seed-based incremental clustering
        
        Uses seed-based approach to reduce LLM calls:
        1. Select a seed vulnerability
        2. LLM finds similar vulnerabilities (5-10)
        3. Form a group
        4. Repeat until all vulnerabilities are grouped
        """
        if not ungrouped_features:
            return []
        
        all_groups = []
        remaining_features = ungrouped_features.copy()
        sub_group_id_base = 1000  # Start from 1000 to avoid conflicts with structural groups
        
        max_iterations = len(ungrouped_features)  # Safety limit
        iteration = 0
        
        while remaining_features and iteration < max_iterations:
            iteration += 1
            
            # Select seed (first remaining vulnerability)
            seed_feature = remaining_features[0]
            seed_localId = seed_feature.localId
            
            # Find similar vulnerabilities using LLM
            similar_result = self._llm_find_similar_vulnerabilities(
                seed_feature, remaining_features, bug_type_group, logger
            )
            if isinstance(similar_result, tuple):
                similar_localIds, grouping_reasoning = similar_result
            else:
                similar_localIds = similar_result
                grouping_reasoning = ""
            
            # Form group (include seed + similar ones)
            group_localIds = [seed_localId] + similar_localIds
            group_features = [f for f in remaining_features if f.localId in group_localIds]
            
            # Infer root cause for this group (pass grouping_reasoning from similarity search)
            sub_group = self._infer_group_root_cause(
                group_features, bug_type_group, sub_group_id_base + len(all_groups), 
                grouping_reasoning, logger
            )
            
            if sub_group:
                all_groups.append(sub_group)
            
            # Remove grouped features from remaining
            remaining_features = [f for f in remaining_features if f.localId not in group_localIds]
            
            if logger:
                logger.info(f"LLM clustering iteration {iteration}: seed {seed_localId}, group size {len(group_localIds)}, remaining {len(remaining_features)}")
        
        return all_groups
    
    def _llm_find_similar_vulnerabilities(self, seed_feature: VulnerabilityFeatures,
                                         candidate_features: List[VulnerabilityFeatures],
                                         bug_type_group: BugTypeGroupInfo,
                                         logger: Optional[logging.Logger] = None) -> tuple[List[int], str]:
        """
        Use LLM to find vulnerabilities similar to the seed vulnerability
        
        Returns tuple of (list of localIds, grouping_reasoning)
        """
        # Limit candidates to avoid token overflow (max 20 candidates per call)
        max_candidates = min(20, len(candidate_features))
        candidates = candidate_features[1:max_candidates]  # Exclude seed itself
        
        if not candidates:
            return [], ""
        
        # Prepare seed information (detailed for seed)
        seed_info = f"""
Seed Vulnerability (localId {seed_feature.localId}):
- Project: {seed_feature.project_name}
- Stack Trace Summary: {seed_feature.stack_trace_summary[:500]}
- Patch Summary: {seed_feature.patch_summary[:400]}
- Dependencies Summary: {seed_feature.dependencies_summary[:300]}
- Reasoning Summary: {seed_feature.llm_reasoning_summary[:400]}
"""
        
        # Prepare candidate information (compressed - key features only)
        candidate_info = []
        for f in candidates:
            stack_preview = f.stack_trace_summary[:150] if f.stack_trace_summary else "N/A"
            patch_preview = f.patch_summary[:100] if f.patch_summary else "N/A"
            deps_preview = f.dependencies_summary[:100] if f.dependencies_summary else "N/A"
            reasoning_preview = f.llm_reasoning_summary[:200] if f.llm_reasoning_summary else "N/A"
            
            candidate_info.append(f"""
localId {f.localId} (Project: {f.project_name}):
- Key Pattern: {reasoning_preview}
- Stack Trace Key: {stack_preview}
- Patch Key: {patch_preview}
- Dependencies: {deps_preview}
""")
        
        prompt = f"""Find vulnerabilities that share SIMILAR VULNERABILITY PATTERNS with the seed vulnerability.

Bug Type Group: {bug_type_group.bug_type}
Module 1 Summary: {bug_type_group.bug_type_group_summary[:500]}...

{seed_info}

Candidate Vulnerabilities (find similar ones - analyze semantic similarity):
{chr(10).join(candidate_info)}

**Task:** Using your semantic understanding, identify vulnerabilities that share similar patterns with the seed:
- **Similar crash patterns**: Same stack trace functions, same crash locations, same memory corruption patterns
- **Similar code patterns**: Similar buffer operations, similar loop structures, similar data structure manipulations
- **Similar patch patterns**: Similar fix strategies, similar code changes, similar validation additions
- **Similar trigger conditions**: Similar input types, similar parsing stages, similar execution paths

**Important:** Focus on semantic similarity in vulnerability patterns, NOT on project names or dependency names alone.

**Output Format:**
SIMILAR_LOCALIDS: [list of localIds that are semantically similar to the seed, comma-separated, max 10]
GROUPING_REASONING: [detailed explanation of why these vulnerabilities form a sub-group, focusing on shared vulnerability patterns. Use Chain-of-Thought reasoning to explain the semantic similarity.]

If no similar vulnerabilities found, output:
SIMILAR_LOCALIDS: []
GROUPING_REASONING: [explanation of why no similar vulnerabilities were found]
"""
        
        try:
            llm_response = self.call_llm(prompt, logger=logger)
            
            # Parse response
            similar_localIds = []
            grouping_reasoning = ""
            lines = llm_response.split('\n')
            parsing_reasoning = False
            
            for line in lines:
                line_upper = line.upper()
                if 'SIMILAR_LOCALIDS:' in line_upper:
                    ids_str = line.split(':', 1)[1].strip() if ':' in line else ''
                    # Extract numbers
                    import re
                    ids = re.findall(r'\d+', ids_str)
                    similar_localIds = [int(id) for id in ids]
                elif 'GROUPING_REASONING:' in line_upper:
                    grouping_reasoning = line.split(':', 1)[1].strip() if ':' in line else ''
                    parsing_reasoning = True
                elif parsing_reasoning and line.strip():
                    # Continue parsing reasoning (multi-line)
                    grouping_reasoning += " " + line.strip()
                elif parsing_reasoning and not line.strip():
                    # Empty line ends reasoning
                    parsing_reasoning = False
            
            # Limit to max 10 and ensure they're in candidates
            similar_localIds = [lid for lid in similar_localIds if lid in [f.localId for f in candidates]][:10]
            
            return similar_localIds, grouping_reasoning
            
        except Exception as e:
            if logger:
                logger.error(f"Error in LLM similarity search: {e}")
            return [], ""
    
    def _infer_group_root_cause(self, group_features: List[VulnerabilityFeatures],
                                bug_type_group: BugTypeGroupInfo,
                                sub_group_id: int,
                                grouping_reasoning: str = "",
                                logger: Optional[logging.Logger] = None) -> Optional[SubGroupInfo]:
        """
        Infer root cause for a group of vulnerabilities
        """
        if not group_features:
            return None
        
        # Prepare group information
        group_info = []
        for f in group_features:
            group_info.append(f"""
localId {f.localId} (Project: {f.project_name}):
- Stack Trace: {f.stack_trace_summary[:300]}...
- Patch: {f.patch_summary[:250]}...
- Dependencies: {f.dependencies_summary[:200]}...
""")
        
        prompt = f"""Analyze this group of vulnerabilities and infer the root cause.

Bug Type Group: {bug_type_group.bug_type}
Module 1 Prioritized Dependencies: {', '.join(bug_type_group.common_dependencies_in_group[:5])}

Group Vulnerabilities:
{chr(10).join(group_info)}

**Task:**
1. Identify the shared vulnerability pattern
2. Determine root cause: Main_Project_Specific or Dependency_Specific
3. If Dependency_Specific, identify the dependency

**Output Format:**
PATTERN_DESCRIPTION: [description of shared pattern]
ROOT_CAUSE_TYPE: [Main_Project_Specific or Dependency_Specific]
ROOT_CAUSE_DEPENDENCY: [dependency name or N/A]
REASONING: [explanation]
CONFIDENCE_SCORE: [0.0-1.0]
"""
        
        try:
            llm_response = self.call_llm(prompt, logger=logger)
            
            # Parse response
            pattern_desc = ""
            root_cause_type = "Unknown"
            root_cause_dep = None
            reasoning = ""
            confidence = 0.0
            
            lines = llm_response.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'pattern_description:' in line_lower:
                    pattern_desc = line.split(':', 1)[1].strip() if ':' in line else ''
                elif 'root_cause_type:' in line_lower:
                    rc_type = line.split(':', 1)[1].strip() if ':' in line else ''
                    if 'main_project_specific' in rc_type.lower():
                        root_cause_type = 'Main_Project_Specific'
                    elif 'dependency_specific' in rc_type.lower():
                        root_cause_type = 'Dependency_Specific'
                elif 'root_cause_dependency:' in line_lower:
                    dep = line.split(':', 1)[1].strip() if ':' in line else ''
                    if dep.lower() not in ['n/a', 'none', 'null', '']:
                        root_cause_dep = dep
                elif 'reasoning:' in line_lower:
                    reasoning = line.split(':', 1)[1].strip() if ':' in line else ''
                elif 'confidence_score:' in line_lower:
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except:
                        pass
            
            return SubGroupInfo(
                sub_group_id=sub_group_id,
                bug_type_group=bug_type_group.bug_type,
                localIds=[f.localId for f in group_features],
                pattern_description=pattern_desc,
                grouping_reasoning=grouping_reasoning if grouping_reasoning else reasoning,
                inferred_root_cause_type=root_cause_type,
                inferred_root_cause_dependency=root_cause_dep,
                reasoning=reasoning,
                confidence_score=confidence
            )
            
        except Exception as e:
            if logger:
                logger.error(f"Error inferring group root cause: {e}")
            return None
    
    def _generate_sub_grouping_analysis(self, bug_type_group: BugTypeGroupInfo,
                                       group_features: List[VulnerabilityFeatures],
                                       logger: Optional[logging.Logger] = None) -> str:
        """Generate LLM analysis for sub-grouping"""
        
        # Prepare detailed information for Module 2
        features_detail = []
        for f in group_features:
            features_detail.append(f"""
localId {f.localId} (Project: {f.project_name}):
- Stack Trace Summary: {f.stack_trace_summary}
- Patch Summary: {f.patch_summary}
- Dependencies Summary: {f.dependencies_summary}
- Code Snippets Summary: {f.code_snippets_summary[:300]}
- Reasoning Summary: {f.llm_reasoning_summary}
""")
        
        prompt = f"""Perform fine-grained sub-grouping within this bug type group.

Bug Type Group: {bug_type_group.bug_type}
Module 1 Summary: {bug_type_group.bug_type_group_summary}
Module 1 Pattern Observation: {bug_type_group.initial_pattern_observation}
Module 1 Prioritized Dependencies: {', '.join(bug_type_group.common_dependencies_in_group)}

Vulnerability Details (using detailed information for precise analysis):
{chr(10).join(features_detail)}

**CRITICAL: Group by VULNERABILITY PATTERNS first, NOT by root cause dependency.**

Using your reasoning capabilities, perform detailed sub-grouping based on **similar vulnerability patterns**:

1. **PRIMARY GROUPING CRITERIA (Pattern-Based):**
   - **Group vulnerabilities that share similar crash patterns**: Same stack trace functions, same crash locations, same memory corruption patterns
   - **Group vulnerabilities that share similar code patterns**: Similar buffer operations, similar loop structures, similar data structure manipulations
   - **Group vulnerabilities that share similar patch patterns**: Similar fix strategies, similar code changes, similar validation additions
   - **Group vulnerabilities that share similar trigger conditions**: Similar input types, similar parsing stages, similar execution paths

2. **SECONDARY ANALYSIS (Root Cause Inference):**
   After grouping by patterns, THEN analyze each sub-group to determine:
   - What is the likely root cause for this sub-group? (Main project vs dependency)
   - If Dependency_Specific, which dependency? (Use Module 1's prioritized dependencies as candidates)
   - Why do these vulnerabilities share the same root cause?

3. **IMPORTANT PRINCIPLES:**
   - **Pattern similarity comes FIRST**: Vulnerabilities with similar crash/stack patterns should be grouped together, even if they're from different projects
   - **Root cause inference comes SECOND**: After grouping by patterns, infer whether the pattern points to a shared dependency or project-specific code
   - **Avoid over-grouping**: Don't create one huge "Main Project" group. Look for meaningful pattern distinctions
   - **Look for shared dependencies**: If multiple vulnerabilities share similar patterns AND use the same dependency, that dependency is likely the root cause

**Example of GOOD grouping:**
- Sub-Group A: 5 vulnerabilities all crash in string concatenation functions (strcat, sprintf) → Pattern: String buffer overflow → Root cause: Likely same string library or similar code pattern
- Sub-Group B: 3 vulnerabilities all crash in hash table lookup → Pattern: Hash table corruption → Root cause: Likely same hash table implementation

**Example of BAD grouping:**
- Sub-Group X: All vulnerabilities from project A → This is project-based, not pattern-based
- Sub-Group Y: All vulnerabilities using libpng → This is dependency-based, not pattern-based

Output your sub-grouping analysis in the following format:
Bug Type Group: {bug_type_group.bug_type}
Sub-Group 1:
- LocalIds: [list of localIds]
- Pattern Description: [What vulnerability pattern do these share? e.g., "String buffer overflow in concatenation", "Hash table corruption during lookup"]
- Root Cause Type: [Main_Project_Specific or Dependency_Specific]
- Root Cause Dependency: [dependency name or N/A]
- Reasoning: [Why these vulnerabilities form a sub-group based on patterns, and what the root cause is]
- Confidence Score: [0.0-1.0]

Sub-Group 2:
- LocalIds: [list of localIds]
...
"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _parse_sub_grouping_response(self, bug_type: str,
                                    group_features: List[VulnerabilityFeatures],
                                    llm_response: str) -> List[SubGroupInfo]:
        """Parse Module 2 LLM response to extract sub-groups"""
        sub_groups = []
        
        lines = llm_response.split('\n')
        current_subgroup = None
        sub_group_id = 1
        parsing_localids = False
        parsing_reasoning = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            line_original = line
            
            if 'sub-group' in line_lower or 'subgroup' in line_lower:
                # Save previous subgroup if exists
                if current_subgroup:
                    sub_groups.append(current_subgroup)
                
                # Start new subgroup
                current_subgroup = {
                    'sub_group_id': sub_group_id,
                    'bug_type_group': bug_type,
                    'localIds': [],
                    'pattern_description': '',
                    'grouping_reasoning': '',
                    'inferred_root_cause_type': 'Unknown',
                    'inferred_root_cause_dependency': None,
                    'reasoning': '',
                    'confidence_score': 0.0
                }
                sub_group_id += 1
                parsing_localids = False
                parsing_reasoning = False
            elif current_subgroup:
                if 'localids:' in line_lower or 'local ids:' in line_lower:
                    # Parse localIds - start parsing mode
                    ids_str = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                    # Extract numbers from this line
                    ids = re.findall(r'\d+', ids_str)
                    current_subgroup['localIds'].extend([int(id) for id in ids])
                    # Check if line contains '[' or starts a multi-line list
                    if '[' in ids_str and ']' not in ids_str:
                        parsing_localids = True
                    elif ']' in ids_str:
                        parsing_localids = False
                elif parsing_localids:
                    # Continue parsing localIds from multiple lines
                    ids = re.findall(r'\d+', line)
                    current_subgroup['localIds'].extend([int(id) for id in ids])
                    if ']' in line:
                        parsing_localids = False
                elif 'pattern description:' in line_lower:
                    pattern_desc = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_subgroup['pattern_description'] = pattern_desc
                    parsing_reasoning = False
                elif 'grouping_reasoning:' in line_lower:
                    grouping_reason = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_subgroup['grouping_reasoning'] = grouping_reason
                    parsing_reasoning = True
                elif 'root cause type:' in line_lower:
                    rc_type = line.split(':', 1)[1].strip() if ':' in line else ''
                    if 'main_project_specific' in rc_type.lower() or 'main project-specific' in rc_type.lower():
                        current_subgroup['inferred_root_cause_type'] = 'Main_Project_Specific'
                    elif 'dependency_specific' in rc_type.lower() or 'dependency-specific' in rc_type.lower():
                        current_subgroup['inferred_root_cause_type'] = 'Dependency_Specific'
                    parsing_reasoning = False
                elif 'root cause dependency:' in line_lower:
                    dep = line.split(':', 1)[1].strip() if ':' in line else ''
                    if dep.lower() not in ['n/a', 'none', 'null', '']:
                        current_subgroup['inferred_root_cause_dependency'] = dep
                    parsing_reasoning = False
                elif 'reasoning:' in line_lower:
                    reasoning_text = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_subgroup['reasoning'] = reasoning_text
                    parsing_reasoning = True
                elif 'confidence score:' in line_lower:
                    try:
                        conf_str = line.split(':', 1)[1].strip() if ':' in line else ''
                        current_subgroup['confidence_score'] = float(conf_str)
                    except:
                        pass
                    parsing_reasoning = False
                elif parsing_reasoning:
                    # Continue parsing reasoning (until next field starts)
                    if line.strip() and not line.strip().startswith('-') and ':' not in line:
                        if current_subgroup['reasoning']:
                            current_subgroup['reasoning'] += ' ' + line.strip()
                        else:
                            current_subgroup['reasoning'] = line.strip()
                    elif ':' in line and any(keyword in line_lower for keyword in ['localids', 'root cause', 'confidence']):
                        parsing_reasoning = False
                elif line.strip() and not line.strip().startswith('-') and not parsing_localids:
                    # If we're not in any specific parsing mode and line doesn't start with '-', it might be continuation
                    # But only if we're not starting a new field
                    if ':' not in line or not any(keyword in line_lower for keyword in ['localids', 'root cause', 'reasoning', 'confidence']):
                        if current_subgroup['reasoning'] and not parsing_reasoning:
                            # Might be continuation of reasoning
                            pass
        
        # Add last subgroup
        if current_subgroup:
            sub_groups.append(current_subgroup)
        
        # Convert to SubGroupInfo objects
        result = []
        for sg_dict in sub_groups:
            if sg_dict['localIds']:  # Only add if has localIds
                result.append(SubGroupInfo(
                    sub_group_id=sg_dict['sub_group_id'],
                    bug_type_group=sg_dict['bug_type_group'],
                    localIds=sg_dict['localIds'],
                    pattern_description=sg_dict.get('pattern_description', ''),
                    grouping_reasoning=sg_dict.get('grouping_reasoning', sg_dict.get('reasoning', '')),
                    inferred_root_cause_type=sg_dict['inferred_root_cause_type'],
                    inferred_root_cause_dependency=sg_dict.get('inferred_root_cause_dependency'),
                    reasoning=sg_dict.get('reasoning', ''),
                    confidence_score=sg_dict['confidence_score']
                ))
        
        return result
    
    # ========================================================================
    # Module 3.2: Similarity-Based Initial Grouping Module (Legacy - kept for compatibility)
    # ========================================================================
    
    def group_similar_vulnerabilities(self, features_list: List[VulnerabilityFeatures], 
                                     n_clusters: Optional[int] = None) -> List[ClusterInfo]:
        """
        Cluster similar vulnerabilities
        
        Args:
            features_list: List of VulnerabilityFeatures
            n_clusters: Number of clusters (auto-determined if None)
        
        Returns:
            List of ClusterInfo
        """
        print(f"[Legacy Clustering] Grouping {len(features_list)} vulnerabilities...")
        
        if len(features_list) == 0:
            return []
        
        # Extract embedding vectors
        embeddings = []
        localIds = []
        for features in features_list:
            # Handle both dict and object access
            if isinstance(features, dict):
                emb = features.get('semantic_embedding')
            else:
                emb = features.semantic_embedding
            
            # Skip if embedding is None or a string description
            if emb and isinstance(emb, list) and len(emb) > 0:
                embeddings.append(emb)
                localId = features.get('localId') if isinstance(features, dict) else features.localId
                localIds.append(localId)
        
        if len(embeddings) == 0:
            print("[-] No embeddings available for clustering")
            return []
        
        embeddings_array = np.array(embeddings)
        
        # Perform clustering
        print(f"  [*] Performing clustering...")
        # Auto-determine number of clusters (based on data size)
        if n_clusters is None:
            n_clusters = min(10, max(2, len(features_list) // 10))
        
        cluster_labels = self._simple_clustering(embeddings_array, n_clusters)
        
        # Group by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(localIds[i])
        
        # LLM interpretation for each cluster
        print(f"  [*] Generating cluster summaries with LLM...")
        cluster_infos = []
        
        for cluster_id, cluster_localIds in clusters.items():
            # Collect features within cluster
            cluster_features = [
                f for f in features_list 
                if (f.localId if hasattr(f, 'localId') else f.get('localId')) in cluster_localIds
            ]
            
            # Extract common characteristics
            common_characteristics = self._extract_common_characteristics(cluster_features)
            
            # Extract common dependencies
            common_dependencies = self._extract_common_dependencies(cluster_features)
            
            # Generate LLM cluster summary
            llm_cluster_summary = self._generate_cluster_summary(
                cluster_id, cluster_localIds, cluster_features,
                common_characteristics, common_dependencies
            )
            
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                localIds=cluster_localIds,
                common_characteristics=common_characteristics,
                common_dependencies=common_dependencies,
                llm_cluster_summary=llm_cluster_summary
            )
            cluster_infos.append(cluster_info)
        
        print(f"  [+] Created {len(cluster_infos)} clusters")
        return cluster_infos
    
    def _simple_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple distance-based clustering (when sklearn/scipy unavailable)"""
        def cosine_distance(a, b):
            """Calculate cosine distance"""
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - (dot_product / (norm_a * norm_b))
        
        n_samples = len(embeddings)
        if n_samples <= n_clusters:
            return np.arange(n_samples)
        
        # Select initial centroids
        centers = embeddings[:n_clusters].copy()
        labels = np.zeros(n_samples, dtype=int)
        
        # Simple k-means style assignment
        for iteration in range(10):  # Maximum 10 iterations
            # Assign each point to nearest centroid
            for i in range(n_samples):
                distances = [cosine_distance(embeddings[i], center) for center in centers]
                labels[i] = np.argmin(distances)
            
            # Update centroids
            new_centers = []
            for k in range(n_clusters):
                cluster_points = embeddings[labels == k]
                if len(cluster_points) > 0:
                    new_centers.append(cluster_points.mean(axis=0))
                else:
                    new_centers.append(centers[k])
            centers = np.array(new_centers)
        
        return labels
    
    def _extract_common_characteristics(self, cluster_features: List[VulnerabilityFeatures]) -> str:
        """Extract common characteristics within cluster"""
        if not cluster_features:
            return "No common characteristics"
        
        # Bug type statistics
        bug_types = defaultdict(int)
        projects = defaultdict(int)
        for f in cluster_features:
            bug_types[f.bug_type] += 1
            projects[f.project_name] += 1
        
        common_bug_type = max(bug_types.items(), key=lambda x: x[1])[0] if bug_types else "Unknown"
        common_project = max(projects.items(), key=lambda x: x[1])[0] if projects else "Unknown"
        
        return f"Common bug type: {common_bug_type}, Common project: {common_project}, Cluster size: {len(cluster_features)}"
    
    def _extract_common_dependencies(self, cluster_features: List[VulnerabilityFeatures]) -> List[str]:
        """Extract common dependencies within cluster"""
        # Extract dependency names (from dependencies_summary)
        dependency_counts = defaultdict(int)
        
        for f in cluster_features:
            # Extract dependency names from dependencies_summary (simple parsing)
            deps_text = f.dependencies_summary.lower()
            # Find common dependency name patterns
            common_deps = ['libpng', 'libjpeg', 'zlib', 'openssl', 'libxml', 'libxslt', 
                          'qtbase', 'qtsvg', 'libheif', 'libjxl', 'libvips']
            for dep in common_deps:
                if dep in deps_text:
                    dependency_counts[dep] += 1
        
        # Return most frequently mentioned dependencies
        sorted_deps = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)
        return [dep for dep, count in sorted_deps[:5]]
    
    def _generate_cluster_summary(self, cluster_id: int, localIds: List[int],
                                 cluster_features: List[VulnerabilityFeatures],
                                 common_characteristics: str, common_dependencies: List[str]) -> str:
        """Generate LLM cluster summary"""
        
        features_summary = []
        for f in cluster_features[:5]:  # Top 5 only
            features_summary.append(f"- localId {f.localId}: {f.bug_type} in {f.project_name}")
        
        prompt = f"""Analyze the following cluster of similar vulnerabilities and provide a comprehensive summary:

Cluster ID: {cluster_id}
Number of vulnerabilities: {len(localIds)}
LocalIds: {localIds[:10]}{'...' if len(localIds) > 10 else ''}

Common Characteristics:
{common_characteristics}

Common Dependencies:
{', '.join(common_dependencies) if common_dependencies else 'None identified'}

Sample Vulnerabilities:
{chr(10).join(features_summary)}

Provide a summary that:
1. Explains what makes these vulnerabilities similar
2. Identifies common patterns or root causes
3. Suggests potential shared dependencies or code paths"""
        
        return self.call_llm(prompt)

    # ========================================================================
    # Module 3: Cross-Group Root Cause Inference & Validation
    # ========================================================================
    
    def infer_cross_group_root_cause(self, bug_type_groups: List[BugTypeGroupInfo],
                                     all_sub_groups: List[SubGroupInfo],
                                     features_list: List[VulnerabilityFeatures],
                                     ground_truth: Optional[Dict] = None,
                                     logger: Optional[logging.Logger] = None) -> List[RootCauseInference]:
        """
        Module 3: Group-Based Root Cause Inference & Cross-Project Validation
        
        Performs final root cause inference for EACH Sub-Group based on group patterns
        and shared dependency matching. This leverages LLM's strength in pattern recognition
        and cross-project validation.
        
        Args:
            bug_type_groups: List of BugTypeGroupInfo from Module 1
            all_sub_groups: List of SubGroupInfo from Module 2
            features_list: List of all VulnerabilityFeatures
            ground_truth: Ground Truth dictionary (optional)
            logger: Optional logger for logging
        
        Returns:
            List of RootCauseInference objects (one per Sub-Group)
        """
        print(f"[Module 3] Performing group-based root cause inference for {len(all_sub_groups)} sub-groups...")
        
        # Create mapping: bug_type -> bug_type_group
        bug_type_to_group = {bg.bug_type: bg for bg in bug_type_groups}
        
        # Collect individual inferences from Module 1
        individual_inferences = {}
        for bg in bug_type_groups:
            if bg.individual_root_causes:
                individual_inferences.update(bg.individual_root_causes)
        
        print(f"  [*] Collected {len(individual_inferences)} individual inferences from Module 1")
        
        # Process each Sub-Group (Group-based inference)
        root_cause_inferences = []
        
        for sub_group in all_sub_groups:
            print(f"  [*] Processing Sub-Group {sub_group.sub_group_id} (Bug Type: {sub_group.bug_type_group}, {len(sub_group.localIds)} vulnerabilities)...")
            
            # Get bug type group for this sub-group
            bug_type_group = bug_type_to_group.get(sub_group.bug_type_group)
            
            # Get features for all localIds in this sub-group
            group_features = [f for f in features_list if f.localId in sub_group.localIds]
            if not group_features:
                print(f"    [-] No features found for Sub-Group {sub_group.sub_group_id}, skipping")
                continue
            
            # Get individual inferences for this sub-group
            group_individual_inferences = {
                lid: individual_inferences.get(lid)
                for lid in sub_group.localIds
                if lid in individual_inferences
            }
            
            # Detect and correct outliers if we have individual inferences
            if len(group_individual_inferences) >= 2:
                corrected_count = self._detect_and_correct_outliers(
                    sub_group, group_individual_inferences, bug_type_group, group_features, logger
                )
                if corrected_count > 0:
                    print(f"    [+] Corrected {corrected_count} outliers in Sub-Group {sub_group.sub_group_id}")
            
            # Analyze shared dependencies across the group
            group_dependencies = self._analyze_group_dependencies(sub_group.localIds, features_list)
            
            # Generate group-based LLM reasoning (WITH GT rules for self-validation, but not GT inference results)
            # We pass GT to provide heuristic rules for self-validation, but LLM makes independent inference
            # Include individual inferences in the reasoning
            llm_reasoning_process = self._generate_group_based_reasoning(
                sub_group, bug_type_group, group_features, group_dependencies,
                individual_inferences=group_individual_inferences,
                ground_truth=ground_truth, logger=logger  # Pass GT for heuristic rules self-validation
            )
            
            # Log LLM response
            if logger:
                logger.info(f"LLM group-based reasoning for Sub-Group {sub_group.sub_group_id} ({len(llm_reasoning_process)} chars):")
                logger.info(f"Response preview: {llm_reasoning_process[:500]}...")
                if len(llm_reasoning_process) <= 10000:
                    logger.info(f"Full LLM Reasoning Response:\n{llm_reasoning_process}")
                else:
                    logger.info(f"Full LLM Reasoning Response (first 10000 chars):\n{llm_reasoning_process[:10000]}...")
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
            
            # Parse group-based inference result
            inferred_result = self._parse_group_based_inference(llm_reasoning_process)
            
            # 🔴 3순위 개선: Contradiction Scanner (반증 탐지기)
            root_cause_type = inferred_result.get('group_level_root_cause_type', 'Unknown')
            root_cause_dep = inferred_result.get('group_level_root_cause_dependency')
            contradiction_scan = None
            if root_cause_type == 'Dependency_Specific' and root_cause_dep:
                print(f"    [*] Running Contradiction Scanner (LLM)...")
                try:
                    contradiction_scan = self._scan_contradictions(
                        root_cause_type, root_cause_dep, sub_group, group_features,
                        inferred_result, ground_truth, logger=logger
                    )
                    if contradiction_scan and contradiction_scan.contradiction:
                        print(f"    [!] Contradiction detected: {contradiction_scan.contradiction_type} (severity: {contradiction_scan.severity})")
                        # Apply confidence downscale if high severity
                        if contradiction_scan.severity == 'high':
                            inferred_result['module3_confidence'] = (inferred_result.get('module3_confidence', 0.8) or 0.8) * 0.7
                            print(f"    [!] Confidence downscaled due to high-severity contradiction")
                except Exception as e:
                    if logger:
                        logger.warning(f"Contradiction scan failed: {e}")
                    print(f"    [!] Warning: Contradiction scan failed: {e}")
            
            # Calculate dependency matching ratio
            matching_ratio, matching_count = self._calculate_dependency_matching_ratio(
                sub_group.localIds, root_cause_dep, group_dependencies, group_features
            )
            
            # CVE Validation (if dependency inferred)
            cve_validation = None
            if root_cause_dep:
                cve_validation = self._validate_with_cve_patterns(
                    sub_group, group_features, root_cause_dep, logger
                )
                if cve_validation and logger:
                    logger.info(f"CVE Validation for Sub-Group {sub_group.sub_group_id}: {cve_validation}")
            
            # Perform discrepancy analysis with GT if available
            per_localId_discrepancies = []
            if ground_truth:
                per_localId_discrepancies = self._analyze_group_gt_discrepancies(
                    sub_group, inferred_result, ground_truth, logger
                )
            
            # Calculate scores
            main_project_score = inferred_result.get('main_project_score', 0.0)
            dependency_score = inferred_result.get('dependency_score', 0.0)
            confidence_score = max(main_project_score, dependency_score)
            
            # Determine final root cause type based on scores
            if dependency_score > main_project_score:
                group_level_root_cause_type = "Dependency_Specific"
                group_level_root_cause_dependency = inferred_result.get('group_level_root_cause_dependency')
            else:
                group_level_root_cause_type = "Main_Project_Specific"
                group_level_root_cause_dependency = None
            
            # 🟠 4순위 개선: Explainability Generator (인과 흐름 서술형 설명)
            # Improve reasoning with causal flow narrative
            enhanced_reasoning = None
            if llm_reasoning_process:
                try:
                    enhanced_reasoning = self._generate_causal_flow_explanation(
                        group_level_root_cause_type, group_level_root_cause_dependency, group_features, 
                        inferred_result, contradiction_scan, logger=logger
                    )
                    if enhanced_reasoning:
                        # Merge enhanced reasoning into justification
                        original_justification = inferred_result.get('group_pattern_justification', '')
                        inferred_result['group_pattern_justification'] = f"""{enhanced_reasoning}

---
Original Group Pattern Analysis:
{original_justification}"""
                except Exception as e:
                    if logger:
                        logger.warning(f"Causal flow explanation generation failed: {e}")
            
            # Gather evidence sources
            evidence_sources = []
            evidence_sources.append(f"Group pattern analysis ({len(sub_group.localIds)} vulnerabilities)")
            evidence_sources.append(f"Dependency matching analysis ({matching_ratio*100:.1f}% match)")
            if bug_type_group:
                evidence_sources.append(f"Bug type group analysis ({bug_type_group.bug_type})")
            
            # Create RootCauseInference for this Sub-Group
            root_cause_inference = RootCauseInference(
                sub_group_id=sub_group.sub_group_id,
                bug_type_group=sub_group.bug_type_group,
                localIds=sub_group.localIds,
                group_level_root_cause_type=group_level_root_cause_type,
                group_level_root_cause_dependency=group_level_root_cause_dependency,
                group_level_root_cause_dependency_version=inferred_result.get('group_level_root_cause_dependency_version'),
                group_pattern_justification=inferred_result.get('group_pattern_justification', ''),
                dependency_matching_ratio=matching_ratio,
                dependency_matching_count=matching_count,
                cross_project_propagation_insight=inferred_result.get('cross_project_propagation_insight'),
                cve_validation=cve_validation,
                llm_reasoning_process=llm_reasoning_process,
                confidence_score=confidence_score,
                main_project_score=main_project_score,
                dependency_score=dependency_score,
                evidence_sources=evidence_sources,
                module1_confidence=bug_type_group.confidence_score if bug_type_group else None,
                module2_confidence=sub_group.confidence_score,
                module3_confidence=inferred_result.get('module3_confidence'),
                discrepancy_analysis=json.dumps(per_localId_discrepancies) if per_localId_discrepancies else None,
                discrepancy_type='has_discrepancies' if per_localId_discrepancies else None,
                corrective_reasoning=f"Found {len(per_localId_discrepancies)} discrepancies with heuristic GT" if per_localId_discrepancies else None,
                per_localId_discrepancies=per_localId_discrepancies,
                contradiction_scan=contradiction_scan,  # NEW: Contradiction scan result
                causal_flow_explanation=enhanced_reasoning  # NEW: Causal flow explanation
            )
            
            root_cause_inferences.append(root_cause_inference)
            print(f"    [+] Sub-Group {sub_group.sub_group_id}: {group_level_root_cause_type} ({group_level_root_cause_dependency or 'N/A'})")
            print(f"        Dependency matching: {matching_count}/{len(sub_group.localIds)} ({matching_ratio*100:.1f}%), confidence: {confidence_score:.2f}")
        
        print(f"  [+] Group-based root cause inference complete: {len(root_cause_inferences)} sub-group inferences")
        return root_cause_inferences
    
    def _generate_per_localId_reasoning(self, localId: int,
                                       feature: VulnerabilityFeatures,
                                       bug_type_group: Optional[BugTypeGroupInfo],
                                       sub_group: Optional[SubGroupInfo],
                                       dependencies: List[Dict],
                                       ground_truth: Optional[Dict],
                                       logger: Optional[logging.Logger] = None) -> str:
        """Generate LLM reasoning for a specific localId"""
        
        # Prepare context from Module 1 (bug type group)
        module1_context = ""
        if bug_type_group:
            module1_context = f"""
Bug Type Group Context:
- Bug Type: {bug_type_group.bug_type}
- Group Summary: {bug_type_group.bug_type_group_summary[:500]}...
- Pattern Observation: {bug_type_group.initial_pattern_observation[:300]}...
- Common Dependencies in Group: {', '.join(bug_type_group.common_dependencies_in_group[:5])}
- Group Confidence: {bug_type_group.confidence_score:.2f}
"""
        
        # Prepare context from Module 2 (sub-group)
        module2_context = ""
        if sub_group:
            module2_context = f"""
Sub-Group Context:
- Sub-Group ID: {sub_group.sub_group_id}
- Root Cause Type (from sub-group): {sub_group.inferred_root_cause_type}
- Root Cause Dependency (from sub-group): {sub_group.inferred_root_cause_dependency or 'N/A'}
- Sub-Group Reasoning: {sub_group.reasoning[:500]}...
- Sub-Group Confidence: {sub_group.confidence_score:.2f}
"""
        
        # Prepare dependencies info
        deps_info = []
        for dep in dependencies[:10]:
            deps_info.append(f"- {dep.get('name', 'N/A')} (path: {dep.get('path', 'N/A')})")
        
        prompt = f"""Perform root cause inference for vulnerability localId {localId}.

**Vulnerability Details:**
- Project: {feature.project_name}
- Bug Type: {feature.bug_type}
- Severity: {feature.severity}

**Stack Trace Summary:**
{feature.stack_trace_summary[:1000]}{'...' if len(feature.stack_trace_summary) > 1000 else ''}

**Patch Summary:**
{feature.patch_summary[:500]}{'...' if len(feature.patch_summary) > 500 else ''}

**Dependencies Summary:**
{feature.dependencies_summary[:800]}{'...' if len(feature.dependencies_summary) > 800 else ''}

**LLM Reasoning Summary (from feature extraction):**
{feature.llm_reasoning_summary[:800]}{'...' if len(feature.llm_reasoning_summary) > 800 else ''}

{module1_context}

{module2_context}

**Dependencies for this vulnerability:**
{chr(10).join(deps_info) if deps_info else 'None identified'}

**Task: Determine Root Cause**

Your goal is to determine whether this vulnerability is:
1. **Main_Project_Specific**: The bug is in the main project code (e.g., {feature.project_name})
2. **Dependency_Specific**: The bug is in a dependency library

**CRITICAL: Individual Analysis First, Then Group Analysis**

**Step 1: Analyze THIS specific vulnerability's stack trace FIRST**
1. **Does the stack trace show code paths INSIDE a dependency library?**
   - **CRITICAL**: If the crash location is in paths like `lib/`, `/src/lib`, `vendor/`, `third_party/`, `deps/`, or contains dependency names (e.g., `libjxl/`, `libheif/`, `libpng/`), this is STRONG evidence for Dependency_Specific
   - Examples: `lib/jxl/enc_ans.h`, `libheif/heif_encoding.cc`, `src/libpng/`, `vendor/zlib/` → Dependency_Specific
   - Main project code typically has paths like `coders/`, `MagickCore/`, `src/main/`, or project-specific directories
   - **If the crash file path clearly indicates a dependency, classify as Dependency_Specific regardless of group patterns**

2. **Does the patch modify dependency code or main project code?**

**Step 2: Consider Sub-Group Context (but don't override individual analysis)**
3. Are similar vulnerabilities (from sub-group) pointing to the same dependency?
   - **CRITICAL**: If vulnerabilities in the sub-group have DIFFERENT dependencies (e.g., one crashes in libjxl, another crashes in libheif), they have DIFFERENT root causes
   - **DO NOT force a single classification** when cases have different crash locations in different dependencies
   - If cases have different dependencies, output individual inferences for each case
   - Group-level inference should only be used when ALL cases in the group share the same dependency

4. Is the crash happening in dependency API calls or in how the main project uses them?

**Important Rules:**
- **Stack trace path analysis is PRIMARY and OVERRIDES group patterns**: The crash file path is the most reliable indicator
- **If the crash is in a dependency path (lib/, vendor/, etc.), classify as Dependency_Specific** even if:
  - The dependency doesn't match other cases in the sub-group
  - Group-level patterns suggest otherwise
  - The dependency is vendored/shipped with the project
- Just because a dependency is used does NOT mean the bug is in the dependency - BUT if the crash is INSIDE the dependency code (not just calling it), it IS Dependency_Specific
- The bug could be in how the main project uses the dependency (Main_Project_Specific) ONLY if the crash is in main project code paths
- **If Heuristic GT indicates Dependency_Specific and the stack trace shows a dependency path, trust the stack trace over group-level patterns**
- Look for evidence: patches in dependency repos, stack traces showing dependency code paths

**Output Individual Inferences When Needed:**
If cases in the sub-group have different dependencies or crash locations, provide individual inferences:
```
Individual inferences:
localId 432073014: Dependency_Specific (libjxl)
localId 418219398: Dependency_Specific (libheif)
...
```

**Output Format (MUST follow exactly):**
ROOT_CAUSE_TYPE: [Main_Project_Specific or Dependency_Specific]
ROOT_CAUSE_DEPENDENCY: [dependency name if Dependency_Specific, or "N/A" if Main_Project_Specific]
DEPENDENCY_VERSION: [version if available, or "N/A"]
MAIN_PROJECT_SCORE: [0.0-1.0, confidence that it's Main_Project_Specific]
DEPENDENCY_SCORE: [0.0-1.0, confidence that it's Dependency_Specific]
MODULE3_CONFIDENCE: [0.0-1.0, overall confidence in this inference]
REASONING:
[Detailed Chain-of-Thought reasoning explaining:
  - Why you chose Main_Project_Specific vs Dependency_Specific
  - What evidence supports your decision (stack trace, patch, sub-group analysis)
  - How Module 1 and Module 2 findings influenced your decision
  - If Dependency_Specific, which specific dependency and why
]
EVIDENCE:
[List specific evidence: stack trace frames, patch locations, sub-group patterns, dependency usage patterns, etc.]"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _analyze_group_dependencies(self, localIds: List[int], features_list: List[VulnerabilityFeatures]) -> Dict:
        """Analyze dependencies shared across the group (with path-based filtering)"""
        # Get project name from features (assume same project for group)
        project_name = None
        if features_list:
            project_name = features_list[0].project_name
        
        # Get dependencies for all localIds in the group
        all_dependencies = []
        dependency_counts = defaultdict(int)
        dependency_details = {}
        
        for localId in localIds:
            deps = self._find_shared_dependencies([localId], project_name=project_name)
            for dep in deps:
                dep_name = dep.get('name', 'N/A')
                dep_path = dep.get('path', 'N/A')
                dep_version = dep.get('version', 'N/A')
                
                # Count occurrences
                dependency_counts[dep_name] += 1
                
                # Store details (keep first occurrence details)
                if dep_name not in dependency_details:
                    dependency_details[dep_name] = {
                        'name': dep_name,
                        'path': dep_path,
                        'version': dep_version,
                        'count': 0
                    }
                dependency_details[dep_name]['count'] = dependency_counts[dep_name]
        
        # Sort by frequency
        sorted_deps = sorted(dependency_details.items(), key=lambda x: x[1]['count'], reverse=True)
        
        return {
            'all_dependencies': [dep[1] for dep in sorted_deps],
            'dependency_counts': dict(dependency_counts),
            'total_localIds': len(localIds)
        }
    
    def _detect_and_correct_outliers(self, sub_group: SubGroupInfo,
                                     group_individual_inferences: Dict[int, IndividualRootCause],
                                     bug_type_group: Optional[BugTypeGroupInfo],
                                     group_features: List[VulnerabilityFeatures],
                                     logger: Optional[logging.Logger] = None) -> int:
        """
        Detect and correct outliers in sub-group based on group consensus.
        
        Args:
            sub_group: SubGroupInfo
            group_individual_inferences: Dict of individual inferences for this sub-group
            bug_type_group: Optional BugTypeGroupInfo
            group_features: List of VulnerabilityFeatures for this sub-group
            logger: Optional logger
        
        Returns:
            Number of corrected outliers
        """
        if len(group_individual_inferences) < 2:
            return 0
        
        # Calculate consensus (majority type, ≥60%)
        type_counts = defaultdict(int)
        for rc in group_individual_inferences.values():
            if rc:
                type_counts[rc.root_cause_type] += 1
        
        if not type_counts:
            return 0
        
        majority_type = max(type_counts.items(), key=lambda x: x[1])[0]
        consensus_ratio = type_counts[majority_type] / len(group_individual_inferences)
        
        # Only correct if consensus ≥ 60%
        if consensus_ratio < 0.6:
            return 0
        
        # Find outliers
        outliers = [
            (lid, rc) for lid, rc in group_individual_inferences.items()
            if rc and rc.root_cause_type != majority_type
        ]
        
        if not outliers:
            return 0
        
        # Re-evaluate outliers with group context
        corrected_count = 0
        for outlier_id, outlier_rc in outliers:
            # Find corresponding feature
            feature = next((f for f in group_features if f.localId == outlier_id), None)
            if not feature:
                continue
            
            # Re-evaluate with group context
            try:
                corrected_rc = self._re_evaluate_with_group_context(
                    outlier_id, feature, sub_group, majority_type, bug_type_group, logger
                )
                if corrected_rc and corrected_rc.root_cause_type == majority_type:
                    group_individual_inferences[outlier_id] = corrected_rc
                    corrected_count += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Error re-evaluating outlier {outlier_id}: {e}")
        
        return corrected_count
    
    def _re_evaluate_with_group_context(self, localId: int, feature: VulnerabilityFeatures,
                                       sub_group: SubGroupInfo, majority_type: str,
                                       bug_type_group: Optional[BugTypeGroupInfo],
                                       logger: Optional[logging.Logger] = None) -> Optional[IndividualRootCause]:
        """
        Re-evaluate an outlier case with group context.
        """
        prompt = f"""You previously inferred this vulnerability as {feature.root_cause_type}, but the sub-group consensus ({len(sub_group.localIds)} cases) suggests {majority_type}.

Re-evaluate this case considering the group context:

**Sub-Group Pattern:**
- Pattern: {sub_group.pattern_description[:300]}
- Group Reasoning: {sub_group.grouping_reasoning[:300]}
- Majority Type: {majority_type} ({len(sub_group.localIds)} cases)

**Individual Case:**
- LocalId: {localId}
- Project: {feature.project_name}
- Stack Trace: {feature.stack_trace_summary[:300]}
- Patch: {feature.patch_summary[:300]}

**Question:** Why does this case disagree with the strong group pattern? Should it be corrected to match the majority?

**Output Format (JSON):**
{{
  "root_cause_type": "Main_Project_Specific" | "Dependency_Specific",
  "root_cause_dependency": "dependency_name" | null,
  "confidence": 0.0-1.0,
  "reasoning": "Explanation of why corrected or why original was correct"
}}
"""
        
        llm_response = self.call_llm(prompt, logger=logger)
        
        # Parse response
        try:
            import json
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return IndividualRootCause(
                    localId=localId,
                    root_cause_type=parsed.get('root_cause_type', majority_type),
                    root_cause_dependency=parsed.get('root_cause_dependency'),
                    confidence=float(parsed.get('confidence', 0.0)),
                    reasoning=parsed.get('reasoning', 'Re-evaluated with group context')
                )
        except Exception as e:
            if logger:
                logger.warning(f"Error parsing re-evaluation response: {e}")
        
        return None
    
    def _validate_with_cve_patterns(self, sub_group: SubGroupInfo,
                                    group_features: List[VulnerabilityFeatures],
                                    inferred_dependency: Optional[str],
                                    logger: Optional[logging.Logger] = None) -> Optional[str]:
        """
        Validate inferred root cause against known CVE patterns.
        This is a simple pattern-based validation (full CVE DB integration would require external API).
        
        Returns:
            CVE pattern match string if found, None otherwise
        """
        if not inferred_dependency:
            return None
        
        # Known CVE patterns (examples)
        cve_patterns = {
            'libpng': ['CVE-2015-8540', 'CVE-2017-12652'],
            'openssl': ['CVE-2014-0160', 'CVE-2016-2107'],
            'libjxl': ['CVE-2023-XXXX'],  # Placeholder
            'libheif': ['CVE-2023-XXXX'],  # Placeholder
        }
        
        dep_lower = inferred_dependency.lower()
        for dep_name, cve_list in cve_patterns.items():
            if dep_name in dep_lower:
                # Check if bug type matches known patterns
                bug_type = sub_group.bug_type_group if hasattr(sub_group, 'bug_type_group') else ''
                if 'heap-buffer-overflow' in bug_type.lower() or 'use-of-uninitialized-value' in bug_type.lower():
                    return f"Matches known CVE patterns for {dep_name}: {', '.join(cve_list[:2])}"
        
        return None
    
    def _scan_contradictions(self, inferred_type: str, inferred_dependency: Optional[str],
                             sub_group: SubGroupInfo, group_features: List[VulnerabilityFeatures],
                             inferred_result: Dict, ground_truth: Optional[Dict] = None,
                             logger: Optional[logging.Logger] = None) -> Optional[ContradictionScan]:
        """
        🔴 3순위 개선: LLM-based Contradiction Scanner
        
        LLM에게 휴리스틱 결론, 패치 diff, 크래시 요약, 룰 상세를 주고
        "이 취약점이 dependency root cause라는 결론에 강하게 반대되는 증거가 있는가?"를 물어봅니다.
        
        Returns:
            ContradictionScan 객체 또는 None (실패 시)
        """
        if inferred_type != 'Dependency_Specific' or not inferred_dependency:
            return None
        
        # Prepare patch summaries
        patch_summaries = []
        for f in group_features[:5]:  # Sample first 5
            if f.patch_summary:
                patch_summaries.append(f"localId {f.localId}: {f.patch_summary[:300]}")
        
        # Prepare heuristic rules info
        heuristic_rules = []
        if ground_truth:
            for localId in sub_group.localIds[:5]:
                gt_entry = ground_truth.get(str(localId)) or ground_truth.get(int(localId))
                if gt_entry:
                    rules = gt_entry.get('Heuristic_Satisfied_Rules', [])
                    gt_type = gt_entry.get('Heuristically_Root_Cause_Type', 'Unknown')
                    heuristic_rules.append(f"localId {localId}: Rules {', '.join(rules)}, GT Type: {gt_type}")
        
        prompt = f"""Analyze if there is STRONG CONTRADICTORY EVIDENCE against the following root cause inference.

**Inferred Root Cause:**
- Type: {inferred_type}
- Dependency: {inferred_dependency}

**Patch Summaries (Sample):**
{chr(10).join(patch_summaries) if patch_summaries else 'No patch summaries available'}

**Heuristic Rules Applied:**
{chr(10).join(heuristic_rules) if heuristic_rules else 'No heuristic rules available'}

**Inference Reasoning:**
{inferred_result.get('group_pattern_justification', '')[:500]}

**Task: Contradiction Detection**

Look for STRONG contradictory evidence that suggests this vulnerability is NOT actually caused by {inferred_dependency}.

Examples of strong contradictions:
- Patch directly modifies {inferred_dependency} source code (direct fix, not workaround)
- Patch location is clearly in {inferred_dependency} directory
- Crash occurs in main project code, not dependency code
- Patch removes/changes dependency code directly

Provide your analysis in the following JSON format:

{{
    "contradiction": true | false,
    "contradiction_type": "direct_fix_in_dependency" | "patch_in_main_project" | "crash_in_main_project" | "other" | null,
    "severity": "high" | "medium" | "low" | "none",
    "explanation": "Brief explanation of the contradiction",
    "confidence": 0.0-1.0
}}

Guidelines:
- contradiction: true only if there is STRONG evidence contradicting the inference
- severity:
  - "high": Direct contradiction (e.g., patch modifies dependency source directly)
  - "medium": Indirect contradiction (e.g., patch location suggests main project fix)
  - "low": Weak contradiction (e.g., ambiguous evidence)
  - "none": No contradiction found
  
- If contradiction=false, set severity="none" and contradiction_type=null

Provide ONLY valid JSON, no additional text."""
        
        try:
            response = self.call_llm(prompt, logger=logger)
            
            # Parse JSON response
            import json
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                return ContradictionScan(
                    contradiction=bool(parsed.get('contradiction', False)),
                    contradiction_type=parsed.get('contradiction_type'),
                    severity=parsed.get('severity', 'none'),
                    explanation=parsed.get('explanation', ''),
                    confidence=float(parsed.get('confidence', 0.5))
                )
            else:
                if logger:
                    logger.warning(f"Could not parse JSON from contradiction scan response")
                return None
        except Exception as e:
            if logger:
                logger.warning(f"Error in contradiction scan: {e}")
            return None
    
    def _generate_causal_flow_explanation(self, root_cause_type: str, root_cause_dependency: Optional[str],
                                         group_features: List[VulnerabilityFeatures],
                                         inferred_result: Dict, contradiction_scan: Optional[ContradictionScan] = None,
                                         logger: Optional[logging.Logger] = None) -> Optional[str]:
        """
        🟠 4순위 개선: LLM-based Explainability Generator
        
        인과 흐름 서술형 설명을 생성합니다 (룰 나열형 → 인과 흐름 서술형).
        
        Returns:
            Causal flow explanation string 또는 None (실패 시)
        """
        if not group_features:
            return None
        
        # Prepare sample features
        sample_features = group_features[:3]
        feature_summaries = []
        for f in sample_features:
            feature_summaries.append(f"""
localId {f.localId}:
- Crash Location: {f.stack_trace_summary[:200]}
- Patch Location: {f.patch_summary[:200]}
- Patch-Crash Distance: {f.patch_crash_distance}
- Crash Module: {f.crash_module}
- Patched Module: {f.patched_module}
""")
        
        contradiction_info = ""
        if contradiction_scan and contradiction_scan.contradiction:
            contradiction_info = f"""
**Contradiction Detected:**
- Type: {contradiction_scan.contradiction_type}
- Severity: {contradiction_scan.severity}
- Explanation: {contradiction_scan.explanation}
"""
        
        prompt = f"""Generate a causal flow narrative explaining the root cause inference.

**Inferred Root Cause:**
- Type: {root_cause_type}
- Dependency: {root_cause_dependency or 'N/A'}

**Sample Vulnerability Features:**
{chr(10).join(feature_summaries)}

{contradiction_info}

**Task: Causal Flow Explanation**

Write a narrative explanation that describes the CAUSAL FLOW of how this vulnerability occurs and why the root cause is {root_cause_type}.

Structure your explanation as a narrative flow:
1. **Crash Origin**: Where does the crash originate? (e.g., "Crash originates in libjxl decoding logic")
2. **Patch Location**: Where was the patch applied? (e.g., "Patch in imagemagick adds input validation")
3. **Causal Relationship**: What is the causal relationship? (e.g., "The patch mitigates but does not eliminate the underlying defect")
4. **Root Cause Justification**: Why is this the root cause? (e.g., "Therefore, the root cause is in the dependency")

Example format:
"Crash originates in [dependency] [component] logic. Patch in [main_project] adds [defensive_code] but does not modify [dependency]. Therefore the patch mitigates, but does not eliminate, the underlying defect. The root cause is [dependency_specific]."

Write a clear, concise narrative (2-4 sentences) that explains the causal flow."""
        
        try:
            response = self.call_llm(prompt, logger=logger)
            # Clean up response (remove markdown formatting if present)
            response = re.sub(r'^```[\w]*\n', '', response, flags=re.MULTILINE)
            response = re.sub(r'\n```$', '', response, flags=re.MULTILINE)
            return response.strip()
        except Exception as e:
            if logger:
                logger.warning(f"Error generating causal flow explanation: {e}")
            return None
    
    def _generate_group_based_reasoning(self, sub_group: SubGroupInfo,
                                       bug_type_group: Optional[BugTypeGroupInfo],
                                       group_features: List[VulnerabilityFeatures],
                                       group_dependencies: Dict,
                                       ground_truth: Optional[Dict] = None,
                                       individual_inferences: Optional[Dict[int, IndividualRootCause]] = None,
                                       logger: Optional[logging.Logger] = None) -> str:
        """Generate LLM reasoning for a Sub-Group based on group patterns and shared dependencies"""
        
        # Prepare Module 1 context
        module1_context = ""
        if bug_type_group:
            module1_context = f"""
Bug Type Group Context:
- Bug Type: {bug_type_group.bug_type}
- Group Summary: {bug_type_group.bug_type_group_summary[:500]}...
- Pattern Observation: {bug_type_group.initial_pattern_observation[:300]}...
- Common Dependencies in Group: {', '.join(bug_type_group.common_dependencies_in_group[:5])}
- Group Confidence: {bug_type_group.confidence_score:.2f}
"""
        
        # Prepare Module 1 individual inferences summary
        individual_inferences_summary = ""
        if individual_inferences:
            type_counts = defaultdict(int)
            dep_counts = defaultdict(int)
            for rc in individual_inferences.values():
                if rc:
                    type_counts[rc.root_cause_type] += 1
                    if rc.root_cause_dependency:
                        dep_counts[rc.root_cause_dependency] += 1
            
            individual_inferences_summary = f"""
Individual Root Cause Inferences (from Module 1):
- Total Individual Inferences: {len(individual_inferences)}
- Type Distribution: {dict(type_counts)}
- Dependency Distribution: {dict(dep_counts)}
- Sample Individual Inferences:
"""
            # Add sample individual inferences
            for lid, rc in list(individual_inferences.items())[:5]:
                if rc:
                    individual_inferences_summary += f"  - LocalId {lid}: {rc.root_cause_type} ({rc.root_cause_dependency or 'N/A'}), confidence: {rc.confidence:.2f}\n"
        
        # Prepare Module 2 hypothesis
        module2_hypothesis = f"""
Sub-Group Hypothesis (from Module 2):
- Sub-Group ID: {sub_group.sub_group_id}
- Initial Root Cause Hypothesis: {sub_group.inferred_root_cause_type}
- Initial Root Cause Dependency: {sub_group.inferred_root_cause_dependency or 'N/A'}
- Sub-Group Reasoning: {sub_group.reasoning[:500]}...
- Sub-Group Confidence: {sub_group.confidence_score:.2f}
- LocalIds in this Sub-Group: {sub_group.localIds}
"""
        
        # Prepare group features summary
        projects = defaultdict(int)
        for f in group_features:
            projects[f.project_name] += 1
        
        # Get submodule_bug info from database for each feature
        submodule_bug_info = {}
        try:
            conn = sqlite3.connect(DB_PATH)
            for f in group_features[:10]:
                try:
                    cursor = conn.execute("SELECT submodule_bug FROM arvo WHERE localId = ?", (f.localId,))
                    row = cursor.fetchone()
                    if row:
                        submodule_bug_info[f.localId] = bool(row[0])
                except:
                    submodule_bug_info[f.localId] = False
            conn.close()
        except:
            pass
        
        features_summary = []
        for f in group_features[:10]:  # Limit to first 10 for token efficiency
            submodule_bug = submodule_bug_info.get(f.localId, False)
            # Extract key dependency names from dependencies_summary for quick reference
            dep_names_in_summary = []
            if f.dependencies_summary:
                # Try to extract dependency names from summary (simple heuristic)
                import re
                # Look for patterns like "libjxl", "libpng", etc.
                dep_matches = re.findall(r'\b(lib[a-z0-9_-]+|jpeg[-_]xl|qtbase|qtsvg)\b', f.dependencies_summary.lower())
                dep_names_in_summary = list(set(dep_matches))[:3]  # Limit to 3 most common
            
            # Get individual inference if available
            individual_rc_info = ""
            if individual_inferences and f.localId in individual_inferences:
                individual_rc = individual_inferences[f.localId]
                if individual_rc:
                    individual_rc_info = f"""
- Individual Root Cause (Module 1): {individual_rc.root_cause_type} ({individual_rc.root_cause_dependency or 'N/A'})
- Individual Confidence (Module 1): {individual_rc.confidence:.2f}
- Patch Intent: {individual_rc.patch_intent or 'N/A'}
"""
            
            # 🔴 1-2순위 개선: Include Patch Semantic Classification and Frame Attribution
            patch_semantic_str = ""
            if f.patch_semantic_classification:
                ps = f.patch_semantic_classification
                patch_semantic_str = f"\n- Patch Semantic: intent={ps.patch_intent}, assumed_fault={ps.assumed_fault_location}, scope={ps.fix_scope}"
            
            frame_attribution_str = ""
            if f.frame_attribution:
                fa = f.frame_attribution
                frame_attribution_str = f"\n- Frame Attribution: logical_owner={fa.logical_owner or 'unclear'}, confidence={fa.confidence:.2f}"
            
            features_summary.append(f"""
localId {f.localId} (Project: {f.project_name}):
- Stack Trace: {f.stack_trace_summary[:200]}...
- Patch: {f.patch_summary[:150]}...
- Dependencies Summary: {f.dependencies_summary[:150] if f.dependencies_summary else 'N/A'}...
- Key Dependencies Mentioned: {', '.join(dep_names_in_summary) if dep_names_in_summary else 'None explicitly mentioned'}
- Submodule Bug Flag: {submodule_bug} {'(STRONG indicator of Dependency_Specific)' if submodule_bug else ''}{patch_semantic_str}{frame_attribution_str}{individual_rc_info}
""")
        
        # Prepare dependency matching analysis
        dep_info = []
        for dep in group_dependencies['all_dependencies'][:10]:
            count = dep['count']
            ratio = count / group_dependencies['total_localIds'] * 100
            dep_info.append(f"- {dep['name']} (version: {dep.get('version', 'N/A')}): {count}/{group_dependencies['total_localIds']} vulnerabilities ({ratio:.1f}%)")
        
        # Prepare GT heuristic rules information (if available) for self-validation
        gt_rules_info = ""
        if ground_truth:
            # Extract heuristic rules from GT for each localId in the sub-group
            gt_rules_by_localId = {}
            for localId in sub_group.localIds:
                gt_entry = ground_truth.get(str(localId)) or ground_truth.get(int(localId))
                if gt_entry:
                    satisfied_rules = gt_entry.get('Heuristic_Satisfied_Rules', [])
                    confidence_score = gt_entry.get('Heuristic_Confidence_Score', 0)
                    max_score = gt_entry.get('Heuristic_Max_Score', 5)
                    gt_type = gt_entry.get('Heuristically_Root_Cause_Type', 'Unknown')
                    gt_dep = gt_entry.get('Heuristically_Root_Cause_Dependency', {})
                    if isinstance(gt_dep, dict):
                        gt_dep_name = gt_dep.get('name', 'N/A')
                    else:
                        gt_dep_name = str(gt_dep) if gt_dep else 'N/A'
                    
                    gt_rules_by_localId[localId] = {
                        'rules': satisfied_rules,
                        'confidence': confidence_score,
                        'max_score': max_score,
                        'type': gt_type,
                        'dependency': gt_dep_name
                    }
            
            if gt_rules_by_localId:
                gt_rules_info = f"""
**Heuristic Ground Truth Rules (for self-validation):**
The following heuristic rules were satisfied for vulnerabilities in this Sub-Group:
{chr(10).join([f"- localId {lid}: Rules {', '.join(info['rules']) if info['rules'] else 'None'} (Confidence: {info['confidence']}/{info['max_score']}), GT Type: {info['type']}, GT Dependency: {info['dependency']}" for lid, info in list(gt_rules_by_localId.items())[:5]])}
...
(Total: {len(gt_rules_by_localId)} vulnerabilities with GT rules)

**Self-Validation Task:**
After making your inference, compare it with the heuristic GT rules above:
- If your inference matches the GT rules (e.g., "My inference satisfies Rule 1 and Rule 2"), this strengthens your confidence
- If your inference differs from GT, explain why your semantic analysis is more accurate than the heuristic rules
- Consider: Do the heuristic rules miss important patterns that your semantic analysis captured?
"""
        
        prompt = f"""Perform group-based root cause inference for Sub-Group {sub_group.sub_group_id}.

**Sub-Group Information:**
- Bug Type Group: {sub_group.bug_type_group}
- Number of vulnerabilities: {len(sub_group.localIds)}
- LocalIds: {sub_group.localIds}
- Projects affected: {', '.join(sorted(projects.keys()))} ({len(projects)} distinct projects)

{module1_context}

{module2_hypothesis}

**Group Vulnerability Features (Sample):**
{chr(10).join(features_summary)}

**Dependency Matching Analysis:**
The following dependencies are shared across vulnerabilities in this Sub-Group:
{chr(10).join(dep_info) if dep_info else 'No shared dependencies identified'}

**Important: Dependency Matching Ratio Calculation**
After you infer the root cause dependency, the system will calculate a "dependency matching ratio" by checking:
1. **srcmap.json dependencies**: Direct dependency list from build configuration
2. **srcmap.json paths**: File paths that may reveal submodules (e.g., libpng in opencv/3rdparty/libpng)
3. **LLM-analyzed dependencies_summary**: Your semantic analysis from Module 1/2 that may identify dependencies not explicitly listed in srcmap
4. **Stack trace analysis**: Stack trace summaries that may reveal dependency usage patterns
5. **Submodule Bug Flag**: If `Submodule Bug: True` is shown for a vulnerability, this is a strong indicator that the root cause is in a dependency/submodule, not in the main project code

**Key Information Sources:**
- **Dependencies Summary**: Each vulnerability's `dependencies_summary` field contains LLM-analyzed dependency information from srcmap and stack traces
- **Submodule Bug Flag**: When `Submodule Bug: True`, the fix commit was in a submodule/dependency directory, strongly suggesting Dependency_Specific root cause
- **Patch Summary**: Contains information about patch location and whether it's a dependency update

This multi-source approach allows you to identify dependencies that heuristic GT rules might miss.
The matching ratio validates your inference: high ratio (>80%) = strong evidence, low ratio (<50%) = reconsider needed.

{gt_rules_info}

**Task: Group-Based Root Cause Inference**

Your goal is to determine the root cause for this **entire Sub-Group** based on:
1. **Group Pattern Analysis**: Do all vulnerabilities in this Sub-Group show similar patterns?
2. **Dependency Matching**: Which dependency is most strongly shared across the group?
3. **Cross-Project Validation**: Do vulnerabilities from different projects point to the same dependency?

**Key Questions:**
1. What is the **dependency matching ratio**? (e.g., "4 out of 5 vulnerabilities (80%) use libpng v1.6.34-r1")
2. Do the group patterns (stack traces, patches) consistently point to the same dependency?
3. Is this a **cross-project propagation** pattern? (same dependency affecting multiple projects)
4. Does Module 2's hypothesis align with the dependency matching analysis?
5. **Individual Case Analysis**: Are there any individual cases that differ from the group pattern?
   - Check if any individual case has `Submodule Bug: True` - this is a STRONG indicator of Dependency_Specific
   - Check if individual cases mention different dependencies in their `Dependencies Summary`
   - If individual cases differ significantly, consider whether they should be re-classified separately

**Important Principles:**
- **Group-based inference**: All vulnerabilities in this Sub-Group likely share the same root cause
- **Dependency matching strength**: Higher matching ratio = stronger evidence for dependency root cause
- **Pattern consistency**: If group patterns consistently point to a dependency, it's likely the root cause
- **Cross-project validation**: If multiple projects show the same dependency issue, it strengthens the inference

**Output Format (MUST follow exactly):**
GROUP_LEVEL_ROOT_CAUSE_TYPE: [Main_Project_Specific or Dependency_Specific]
GROUP_LEVEL_ROOT_CAUSE_DEPENDENCY: [dependency name if Dependency_Specific, or "N/A" if Main_Project_Specific]
GROUP_LEVEL_ROOT_CAUSE_DEPENDENCY_VERSION: [version if available, or "N/A"]
DEPENDENCY_MATCHING_RATIO: [0.0-1.0, ratio of vulnerabilities in group that share this dependency]
DEPENDENCY_MATCHING_COUNT: [number of vulnerabilities that share this dependency]
MAIN_PROJECT_SCORE: [0.0-1.0, confidence that it's Main_Project_Specific]
DEPENDENCY_SCORE: [0.0-1.0, confidence that it's Dependency_Specific]
MODULE3_CONFIDENCE: [0.0-1.0, overall confidence in this group-based inference]
GROUP_PATTERN_JUSTIFICATION:
[Detailed explanation:
  - Why this Sub-Group shares the same root cause
  - How dependency matching analysis supports your inference
  - How group patterns (stack traces, patches) align with the dependency
  - Cross-project propagation insights
  - How Module 2's hypothesis was validated or refined
  - If GT heuristic rules are available: Compare your inference with the heuristic rules
    * If your inference matches GT rules: Explain how your semantic analysis confirms the heuristic rules
    * If your inference differs from GT: Explain why your semantic analysis is more accurate (e.g., "Heuristic Rule 1 only checks patch file paths, but my analysis of stack traces and code patterns reveals this is actually a dependency issue")
    * Self-validation: "My inference satisfies Rule X and Rule Y" or "My inference differs because..."
]
CROSS_PROJECT_PROPAGATION_INSIGHT:
[Analysis of how this root cause affects multiple projects, if applicable]
EVIDENCE:
[List specific evidence: dependency matching statistics, group pattern consistency, cross-project patterns, etc.]"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _parse_group_based_inference(self, llm_response: str) -> Dict:
        """Parse LLM response for group-based inference"""
        result = {
            'group_level_root_cause_type': 'Unknown',
            'group_level_root_cause_dependency': None,
            'group_level_root_cause_dependency_version': None,
            'dependency_matching_ratio': 0.0,
            'dependency_matching_count': 0,
            'main_project_score': 0.0,
            'dependency_score': 0.0,
            'module3_confidence': 0.0,
            'group_pattern_justification': '',
            'cross_project_propagation_insight': None,
            'discrepancy_type': None,
            'corrective_reasoning': None
        }
        
        # Extract GROUP_LEVEL_ROOT_CAUSE_TYPE
        root_cause_match = re.search(r'GROUP_LEVEL_ROOT_CAUSE_TYPE:\s*(Main_Project_Specific|Dependency_Specific)', llm_response, re.IGNORECASE)
        if root_cause_match:
            result['group_level_root_cause_type'] = root_cause_match.group(1)
        
        # Extract GROUP_LEVEL_ROOT_CAUSE_DEPENDENCY
        dep_match = re.search(r'GROUP_LEVEL_ROOT_CAUSE_DEPENDENCY:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if dep_match:
            dep_name = dep_match.group(1).strip()
            if dep_name.upper() != 'N/A':
                result['group_level_root_cause_dependency'] = dep_name
        
        # Extract GROUP_LEVEL_ROOT_CAUSE_DEPENDENCY_VERSION
        version_match = re.search(r'GROUP_LEVEL_ROOT_CAUSE_DEPENDENCY_VERSION:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if version_match:
            version = version_match.group(1).strip()
            if version.upper() != 'N/A':
                result['group_level_root_cause_dependency_version'] = version
        
        # Extract DEPENDENCY_MATCHING_RATIO
        ratio_match = re.search(r'DEPENDENCY_MATCHING_RATIO:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if ratio_match:
            try:
                result['dependency_matching_ratio'] = float(ratio_match.group(1))
            except ValueError:
                pass
        
        # Extract DEPENDENCY_MATCHING_COUNT
        count_match = re.search(r'DEPENDENCY_MATCHING_COUNT:\s*([0-9]+)', llm_response, re.IGNORECASE)
        if count_match:
            try:
                result['dependency_matching_count'] = int(count_match.group(1))
            except ValueError:
                pass
        
        # Extract MAIN_PROJECT_SCORE
        main_score_match = re.search(r'MAIN_PROJECT_SCORE:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if main_score_match:
            try:
                result['main_project_score'] = float(main_score_match.group(1))
            except ValueError:
                pass
        
        # Extract DEPENDENCY_SCORE
        dep_score_match = re.search(r'DEPENDENCY_SCORE:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if dep_score_match:
            try:
                result['dependency_score'] = float(dep_score_match.group(1))
            except ValueError:
                pass
        
        # Extract MODULE3_CONFIDENCE
        conf_match = re.search(r'MODULE3_CONFIDENCE:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if conf_match:
            try:
                result['module3_confidence'] = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Extract GROUP_PATTERN_JUSTIFICATION
        just_match = re.search(r'GROUP_PATTERN_JUSTIFICATION:\s*(.+?)(?=\n[A-Z_]+:|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if just_match:
            result['group_pattern_justification'] = just_match.group(1).strip()
        
        # Extract CROSS_PROJECT_PROPAGATION_INSIGHT
        cross_match = re.search(r'CROSS_PROJECT_PROPAGATION_INSIGHT:\s*(.+?)(?=\n[A-Z_]+:|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if cross_match:
            result['cross_project_propagation_insight'] = cross_match.group(1).strip()
        
        return result
    
    def _normalize_dependency_name(self, name: str) -> str:
        """Normalize dependency name for comparison (same as evaluation script)"""
        if not name:
            return ""
        # Convert to lowercase and remove common prefixes/suffixes
        import re
        name = name.lower().strip()
        # Remove version suffixes (including trailing hyphens/dots)
        name = re.sub(r'[v\s]*\d+[.\d]*.*$', '', name)
        name = re.sub(r'[-_.]+$', '', name)  # Remove trailing separators
        # Remove common prefixes
        name = re.sub(r'^(lib|libs|lib-|libs-)', '', name)
        # Handle special cases (e.g., jpeg-xl = libjxl)
        name = re.sub(r'^jpeg[-_]xl$', 'jxl', name)
        name = re.sub(r'^libjxl$', 'jxl', name)
        return name.strip()
    
    def _calculate_dependency_matching_ratio(self, localIds: List[int], 
                                            root_cause_dependency: Optional[str],
                                            group_dependencies: Dict,
                                            group_features: Optional[List[VulnerabilityFeatures]] = None) -> tuple:
        """
        Calculate dependency matching ratio for the group
        
        This function is a core component of Module 3's dependency matching analysis.
        It validates LLM's inferred root cause dependency by checking how many vulnerabilities
        in the Sub-Group actually share this dependency. This process leverages multiple
        information sources beyond srcmap.json:
        
        1. **srcmap.json dependencies**: Direct dependency list from build configuration
        2. **srcmap.json paths**: File paths that may reveal submodules (e.g., libpng in opencv/3rdparty/libpng)
        3. **LLM-analyzed dependencies_summary**: LLM's semantic analysis from Module 1/2 that may identify
           dependencies not explicitly listed in srcmap (e.g., through code patterns, function names)
        4. **Stack trace analysis**: Stack trace summaries that may reveal dependency usage patterns
        
        The matching ratio serves as a key validation metric for LLM's group-based inference:
        - High ratio (e.g., >80%): Strong evidence that the inferred dependency is correct
        - Low ratio (e.g., <50%): Suggests the inference may need reconsideration
        
        This multi-source approach allows LLM to identify dependencies that heuristic GT rules
        might miss (e.g., submodules, implicit dependencies, or dependencies inferred from code patterns).
        
        Args:
            localIds: List of localIds in the Sub-Group
            root_cause_dependency: LLM-inferred root cause dependency name
            group_dependencies: Dependency analysis results from _analyze_group_dependencies
            group_features: List of VulnerabilityFeatures for the Sub-Group
        
        Returns:
            Tuple of (matching_ratio, matching_count):
            - matching_ratio: Ratio of vulnerabilities that share the inferred dependency (0.0-1.0)
            - matching_count: Number of vulnerabilities that share the inferred dependency
        """
        if not root_cause_dependency:
            return 0.0, 0
        
        # Normalize dependency name for matching (handles jpeg-xl vs libjxl, etc.)
        root_cause_dep_normalized = self._normalize_dependency_name(root_cause_dependency)
        root_cause_dep_lower = root_cause_dependency.lower()
        
        # Create mapping: localId -> feature
        feature_map = {}
        if group_features:
            feature_map = {f.localId: f for f in group_features}
        
        # Track matching methods for each localId (for debugging/explanation)
        matching_details = []
        
        # Count how many localIds have this dependency
        matching_count = 0
        for localId in localIds:
            matched = False
            match_method = None
            
            # Method 1: Check srcmap dependencies (most reliable, explicit dependency list)
            deps = self._find_shared_dependencies([localId])
            dep_names = [dep.get('name', '').lower() for dep in deps]
            dep_names_normalized = [self._normalize_dependency_name(dep.get('name', '')) for dep in deps]
            dep_paths = [dep.get('path', '').lower() for dep in deps]
            
            # Check if dependency name matches (exact match or normalized match)
            if (root_cause_dep_lower in dep_names or 
                root_cause_dep_normalized in dep_names_normalized or
                any(self._normalize_dependency_name(dep_name) == root_cause_dep_normalized for dep_name in dep_names)):
                matching_count += 1
                matched = True
                match_method = "srcmap_dependencies"
                continue
            
            # Method 2: Check if dependency name appears in paths (for submodules like libpng in opencv/3rdparty/libpng)
            # Also check normalized names
            if (any(root_cause_dep_lower in path for path in dep_paths) or
                any(root_cause_dep_normalized in self._normalize_dependency_name(path) for path in dep_paths)):
                matching_count += 1
                matched = True
                match_method = "srcmap_paths"
                continue
            
            # Method 3: Check dependencies_summary from features (LLM already analyzed this)
            # This is crucial for submodules that aren't listed in srcmap dependencies
            # LLM's semantic analysis from Module 1/2 may identify dependencies through:
            # - Code pattern recognition
            # - Function name analysis
            # - Import/include statement analysis
            if localId in feature_map:
                feature = feature_map[localId]
                deps_summary_lower = feature.dependencies_summary.lower()
                stack_trace_lower = feature.stack_trace_summary.lower()
                
                # Check if dependency name appears in dependencies_summary or stack_trace
                if root_cause_dep_lower in deps_summary_lower or root_cause_dep_lower in stack_trace_lower:
                    matching_count += 1
                    matched = True
                    match_method = "llm_analysis"  # LLM's semantic analysis from Module 1/2
                    continue
            
            # Method 4: Check srcmap file paths directly (for submodules)
            if not matched:
                try:
                    data = extract_data(localId, include_code_snippets=False, auto_fetch=False)
                    if data:
                        srcmap = data.get('srcmap', {})
                        # Check vulnerable_version file paths
                        if isinstance(srcmap, dict):
                            vul_ver = srcmap.get('vulnerable_version', {})
                            if isinstance(vul_ver, dict):
                                file_info = vul_ver.get('file', {})
                                # If file_info is a dict with paths as keys
                                if isinstance(file_info, dict):
                                    file_paths = [str(k).lower() for k in file_info.keys()]
                                    if any(root_cause_dep_lower in path for path in file_paths):
                                        matching_count += 1
                                        match_method = "srcmap_file_paths"
                                        continue
                except Exception:
                    pass  # Skip if extraction fails
            
            # Track matching details for explanation
            if matched:
                matching_details.append({
                    'localId': localId,
                    'method': match_method
                })
        
        ratio = matching_count / len(localIds) if localIds else 0.0
        
        # Log matching details for debugging (if logger available)
        # This helps explain how the matching ratio was calculated and validates LLM's inference
        
        return ratio, matching_count
    
    def _analyze_group_gt_discrepancies(self, sub_group: SubGroupInfo,
                                       inferred_result: Dict,
                                       ground_truth: Dict,
                                       logger: Optional[logging.Logger] = None) -> List[Dict]:
        """Analyze discrepancies between group inference and per-localId GT"""
        discrepancies = []
        
        llm_type = inferred_result.get('group_level_root_cause_type', 'Unknown')
        llm_dep = inferred_result.get('group_level_root_cause_dependency')
        if llm_dep is None:
            llm_dep = 'N/A'
        
        for localId in sub_group.localIds:
            # Try both str and int keys
            gt_entry = ground_truth.get(str(localId)) or ground_truth.get(localId)
            if not gt_entry:
                if logger:
                    logger.debug(f"No GT found for localId {localId} (tried both str and int keys)")
                continue
            
            # Handle both dict format (from GT file) and direct format
            if isinstance(gt_entry, dict):
                gt_type = gt_entry.get('Heuristically_Root_Cause_Type') or gt_entry.get('Heuristically_Inferred_Root_Cause_Type', 'Unknown')
                gt_dep_info = gt_entry.get('Heuristically_Root_Cause_Dependency') or gt_entry.get('Heuristically_Inferred_Root_Cause_Dependency')
                gt_project_name = gt_entry.get('project_name', '').lower().strip()
                
                # Extract dependency name
                if isinstance(gt_dep_info, dict):
                    gt_dep = gt_dep_info.get('name', 'N/A')
                elif isinstance(gt_dep_info, str):
                    gt_dep = gt_dep_info
                elif gt_dep_info is None:
                    gt_dep = 'N/A'
                else:
                    gt_dep = 'N/A'
                
                # Additional normalization: If project_name == dependency name, treat as Main_Project_Specific
                gt_dep_normalized = str(gt_dep).lower().strip() if gt_dep and gt_dep != 'N/A' else None
                if gt_project_name and gt_dep_normalized and gt_project_name == gt_dep_normalized:
                    # This is actually Main_Project_Specific, not Dependency_Specific
                    if gt_type == 'Dependency_Specific':
                        gt_type = 'Main_Project_Specific'
                        gt_dep = 'N/A'
                        gt_dep_normalized = None
            else:
                gt_type = 'Unknown'
                gt_dep = 'N/A'
                gt_dep_normalized = None
            
            # Normalize dependency names for comparison
            llm_dep_normalized = str(llm_dep).lower().strip() if llm_dep and llm_dep != 'N/A' else None
            if not gt_dep_normalized:
                gt_dep_normalized = None
            
            # Check if there's a discrepancy
            type_mismatch = (llm_type != gt_type)
            dep_mismatch = False
            if llm_dep_normalized and gt_dep_normalized:
                dep_mismatch = (llm_dep_normalized != gt_dep_normalized)
            elif llm_dep_normalized and not gt_dep_normalized:
                dep_mismatch = True  # LLM says dependency, GT says N/A
            elif not llm_dep_normalized and gt_dep_normalized:
                dep_mismatch = True  # LLM says N/A, GT says dependency
            
            if type_mismatch or dep_mismatch:
                discrepancies.append({
                    'localId': localId,
                    'llm_inference': f"{llm_type} ({llm_dep})",
                    'heuristic_gt': f"{gt_type} ({gt_dep})",
                    'type_match': not type_mismatch,
                    'dependency_match': not dep_mismatch,
                    'discrepancy_type': 'type_mismatch' if type_mismatch else 'dependency_mismatch'
                })
        
        return discrepancies
    
    def _parse_per_localId_inference(self, llm_response: str) -> Dict:
        """Parse LLM response for per-localId inference"""
        result = {
            'root_cause_type': 'Unknown',
            'root_cause_dependency': None,
            'dependency_version': None,
            'main_project_score': 0.0,
            'dependency_score': 0.0,
            'module3_confidence': 0.0,
            'discrepancy_type': None,
            'corrective_reasoning': None
        }
        
        # Extract ROOT_CAUSE_TYPE
        root_cause_match = re.search(r'ROOT_CAUSE_TYPE:\s*(Main_Project_Specific|Dependency_Specific)', llm_response, re.IGNORECASE)
        if root_cause_match:
            result['root_cause_type'] = root_cause_match.group(1)
        
        # Extract ROOT_CAUSE_DEPENDENCY
        dep_match = re.search(r'ROOT_CAUSE_DEPENDENCY:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if dep_match:
            dep_name = dep_match.group(1).strip()
            if dep_name.upper() != 'N/A':
                result['root_cause_dependency'] = dep_name
        
        # Extract DEPENDENCY_VERSION
        version_match = re.search(r'DEPENDENCY_VERSION:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if version_match:
            version = version_match.group(1).strip()
            if version.upper() != 'N/A':
                result['dependency_version'] = version
        
        # Extract MAIN_PROJECT_SCORE
        main_score_match = re.search(r'MAIN_PROJECT_SCORE:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if main_score_match:
            try:
                result['main_project_score'] = float(main_score_match.group(1))
            except ValueError:
                pass
        
        # Extract DEPENDENCY_SCORE
        dep_score_match = re.search(r'DEPENDENCY_SCORE:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if dep_score_match:
            try:
                result['dependency_score'] = float(dep_score_match.group(1))
            except ValueError:
                pass
        
        # Extract MODULE3_CONFIDENCE
        conf_match = re.search(r'MODULE3_CONFIDENCE:\s*([0-9.]+)', llm_response, re.IGNORECASE)
        if conf_match:
            try:
                result['module3_confidence'] = float(conf_match.group(1))
            except ValueError:
                pass
        
        # Extract DISCREPANCY_TYPE
        disc_type_match = re.search(r'DISCREPANCY_TYPE:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if disc_type_match:
            result['discrepancy_type'] = disc_type_match.group(1).strip()
        
        # Extract CORRECTIVE_REASONING
        corr_match = re.search(r'CORRECTIVE_REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if corr_match:
            result['corrective_reasoning'] = corr_match.group(1).strip()
        
        return result
    
    def _generate_discrepancy_analysis(self, localId: int,
                                      feature: VulnerabilityFeatures,
                                      inferred_result: Dict,
                                      llm_reasoning: str,
                                      gt_type: str,
                                      gt_dependency: str,
                                      logger: Optional[logging.Logger] = None) -> str:
        """
        Generate discrepancy analysis between LLM inference and heuristic GT.
        
        This is a meta-cognitive task: LLM has already made its inference,
        and now it compares with GT and explains why its inference might be more accurate.
        """
        llm_type = inferred_result.get('root_cause_type', 'Unknown')
        llm_dep = inferred_result.get('root_cause_dependency', 'N/A')
        
        prompt = f"""You have already performed root cause inference for vulnerability localId {localId}.

**Your LLM Inference:**
- Root Cause Type: {llm_type}
- Root Cause Dependency: {llm_dep}
- Your Reasoning: {llm_reasoning[:1000]}...

**Heuristic Ground Truth (from automated heuristic-based system):**
- Root Cause Type: {gt_type}
- Root Cause Dependency: {gt_dependency}

**Task: Meta-Cognitive Reflection & Discrepancy Analysis**

Your inference differs from the heuristic Ground Truth. Perform a meta-cognitive analysis:

1. **Compare your inference with the heuristic GT:**
   - What are the key differences?
   - What evidence did you use that the heuristic might have missed?
   - What limitations might the heuristic GT have?

2. **Classify the discrepancy type:**
   - "heuristic_error": The heuristic GT likely misidentified due to missing srcmap data, mapping failures, or limitations in heuristic rules
   - "llm_error": Your analysis may be incorrect
   - "borderline_case": Ambiguous case where both could be valid

3. **Provide corrective reasoning:**
   - If you believe your inference is more accurate, explain why with detailed evidence
   - Reference specific evidence: stack trace frames, patch locations, code snippets, dependency usage patterns
   - Explain how the heuristic GT might have failed (e.g., patch path analysis without considering actual code flow)

**Output Format:**
DISCREPANCY_TYPE: [heuristic_error, llm_error, or borderline_case]
CORRECTIVE_REASONING:
[Detailed explanation of why your inference is more accurate (if heuristic_error) or why the heuristic GT might be correct (if llm_error), or why both could be valid (if borderline_case). Include specific evidence and reasoning.]
"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _parse_discrepancy_analysis(self, llm_response: str) -> Dict:
        """Parse discrepancy analysis response"""
        result = {
            'discrepancy_type': None,
            'corrective_reasoning': None
        }
        
        # Extract DISCREPANCY_TYPE
        disc_type_match = re.search(r'DISCREPANCY_TYPE:\s*([^\n]+)', llm_response, re.IGNORECASE)
        if disc_type_match:
            disc_type = disc_type_match.group(1).strip().lower()
            if 'heuristic_error' in disc_type:
                result['discrepancy_type'] = 'heuristic_error'
            elif 'llm_error' in disc_type:
                result['discrepancy_type'] = 'llm_error'
            elif 'borderline' in disc_type:
                result['discrepancy_type'] = 'borderline_case'
        
        # Extract CORRECTIVE_REASONING
        corr_match = re.search(r'CORRECTIVE_REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)', llm_response, re.IGNORECASE | re.DOTALL)
        if corr_match:
            result['corrective_reasoning'] = corr_match.group(1).strip()
        
        return result
    
    def _generate_cross_group_reasoning(self, bug_type_groups: List[BugTypeGroupInfo],
                                        all_sub_groups: List[SubGroupInfo],
                                        features_list: List[VulnerabilityFeatures],
                                        common_dependencies: List[Dict],
                                        ground_truth: Optional[Dict],
                                        logger: Optional[logging.Logger] = None) -> str:
        """Generate LLM reasoning for Module 3 using Module 1 and 2 outputs (legacy, kept for compatibility)"""
        
        # Prepare Module 1 outputs
        module1_summary = []
        for bg in bug_type_groups:
            module1_summary.append(f"""
Bug Type Group: {bg.bug_type}
- Summary: {bg.bug_type_group_summary}
- Pattern Observation: {bg.initial_pattern_observation}
- Prioritized Dependencies: {', '.join(bg.common_dependencies_in_group)}
- Confidence: {bg.confidence_score}
- LocalIds: {bg.localIds[:10]}{'...' if len(bg.localIds) > 10 else ''}
""")
        
        # Prepare Module 2 outputs
        module2_summary = []
        for sg in all_sub_groups:
            module2_summary.append(f"""
Sub-Group {sg.sub_group_id} (Bug Type: {sg.bug_type_group}):
- LocalIds: {sg.localIds}
- Root Cause Type: {sg.inferred_root_cause_type}
- Root Cause Dependency: {sg.inferred_root_cause_dependency or 'N/A'}
- Reasoning: {sg.reasoning[:200]}...
- Confidence: {sg.confidence_score}
""")
        
        # Prepare GT information
        gt_info = self._prepare_gt_info(bug_type_groups, ground_truth)
        
        deps_info = []
        for dep in common_dependencies[:10]:
            deps_info.append(f"- {dep.get('name', 'N/A')} ({dep.get('path', 'N/A')})")
        
        prompt = f"""Perform cross-group root cause inference based on Module 1 and 2 analysis results.

**Module 1 Outputs (Bug Type Group Analysis):**
{chr(10).join(module1_summary)}

**Module 2 Outputs (Fine-Grained Sub-Grouping):**
{chr(10).join(module2_summary)}

**Shared Dependencies:**
{chr(10).join(deps_info) if deps_info else 'None identified'}
{f'''

**Heuristic Ground Truth (for reference and discrepancy analysis):**
{chr(10).join([f"- localId {gt['localId']}: {gt['gt_type']} ({gt['gt_dependency']})" for gt in gt_info]) if gt_info else 'No heuristic GT available'}
''' if gt_info else ''}

**Module 3: Cross-Group Root Cause Inference (Final Synthesis & Autonomous Task Generation)**

Based on Module 1 and 2 analysis, perform final root cause inference:

1. **Information Flow Integration:**
   - Use Module 1's common_dependencies as prioritized candidates for root cause analysis
   - Incorporate Module 2's sub-group-level root cause reasoning into your final inference
   - Consider confidence scores from Module 1 and 2: if confidence is low, request more detailed information

2. Compare patterns across bug type groups and sub-groups:
   - Do different bug types/sub-groups point to the same root cause?
   - Are there consistent patterns in stack traces or patch locations across groups?
   - Do all groups show similar dependency usage patterns?

3. Analyze shared dependencies:
   - Are the crashes occurring IN the dependency code or in how the main project uses the dependency?
   - Do patches modify dependency code or main project code?

4. Distinguish between superficial indications and true root causes:
   - Just because a dependency is used does NOT mean the bug is in the dependency
   - The bug could be in how the main project uses the dependency (Main_Project_Specific)
   - Look for evidence: Are patches in dependency repos? Do stack traces show dependency code paths?

5. Consider functional context of dependencies (e.g., image processing, networking, parsing)

6. **Cross-Project Propagation Insight:**
   - Analyze if this root cause affects multiple projects
   - Identify propagation patterns across different bug type groups

7. **Discrepancy Analysis with Heuristic Ground Truth:**
   If heuristic GT is available and differs from your inference:
   - Classify the discrepancy type:
     * "heuristic_error": Heuristic GT misidentified due to missing srcmap data or mapping failures
     * "llm_error": Your analysis may be incorrect
     * "borderline_case": Ambiguous case where both could be valid
   - Provide "Corrective Reasoning": Explain why your inference is more accurate than heuristic GT
     Format: "The heuristic GT suggests [GT's conclusion], but my analysis indicates [your conclusion] is more accurate because [detailed reasoning with evidence]"

**Final Output Format:**
ROOT_CAUSE_TYPE: [Main_Project_Specific or Dependency_Specific]
ROOT_CAUSE_DEPENDENCY: [dependency name if Dependency_Specific, or "N/A" if Main_Project_Specific]
MODULE1_CONFIDENCE: [0.0-1.0]
MODULE2_CONFIDENCE: [0.0-1.0]
MODULE3_CONFIDENCE: [0.0-1.0]
REASONING:
[Detailed Chain-of-Thought reasoning explaining why this is the root cause, structured as:
  Module 1 Analysis Summary: [Summary of Module 1 findings]
  Module 2 Sub-Grouping Summary: [Summary of Module 2 sub-group findings]
  Module 3 Cross-Group Inference: [Final root cause determination based on cross-group and sub-group analysis with confidence score]
  Cross-Project Propagation Insight: [Analysis of propagation patterns]
]
DISCREPANCY_ANALYSIS: [If heuristic GT differs, provide:
  DISCREPANCY_TYPE: [heuristic_error, llm_error, or borderline_case]
  CORRECTIVE_REASONING: [Detailed rebuttal explaining why your inference is more accurate]
]
EVIDENCE:
[List of evidence sources: Module 1 analysis, Module 2 sub-grouping, stack trace analysis, patch analysis, dependency analysis, etc.]"""
        
        return self.call_llm(prompt, logger=logger)
    
    # ========================================================================
    # Legacy Root Cause Inference (kept for backward compatibility)
    # ========================================================================
    
    def infer_root_cause(self, cluster_info: ClusterInfo, 
                        features_list: List[VulnerabilityFeatures],
                        ground_truth: Optional[Dict] = None,
                        logger: Optional[logging.Logger] = None) -> RootCauseInference:
        """
        Infer shared dependency root cause
        
        This method implements the hierarchical analysis pipeline:
        - Module 1: Macro-level pattern recognition (efficient token usage)
        - Module 2: Fine-grained sub-grouping within bug type groups (detailed analysis)
        - Module 3: Cross-group root cause inference with discrepancy analysis
        
        Args:
            cluster_info: Cluster information
            features_list: List of features within cluster
            ground_truth: Ground Truth (for evaluation and discrepancy analysis, optional)
            logger: Optional logger for logging
        
        Returns:
            RootCauseInference object with confidence scores and discrepancy analysis
        """
        print(f"[Legacy] Inferring root cause for cluster {cluster_info.cluster_id}...")
        
        # Collect features within cluster
        cluster_features = [f for f in features_list if f.localId in cluster_info.localIds]
        
        # Find common dependencies from srcmap
        common_dependencies = self._find_shared_dependencies(cluster_info.localIds)
        
        # Deep semantic inference using LLM
        print(f"  [*] Performing deep semantic inference with LLM...")
        if logger:
            logger.info(f"Generating root cause reasoning for cluster {cluster_info.cluster_id}")
        
        # Prepare ground truth information for discrepancy analysis
        gt_info = self._prepare_gt_info_for_localIds(cluster_info.localIds, ground_truth)
        
        llm_reasoning_process = self._generate_root_cause_reasoning(
            cluster_info, cluster_features, common_dependencies, logger, gt_info
        )
        
        # Log LLM response immediately
        if logger:
            logger.info(f"LLM reasoning response for cluster {cluster_info.cluster_id} ({len(llm_reasoning_process)} chars):")
            logger.info(f"Response preview: {llm_reasoning_process[:500]}...")
            # Log full response if not too long (limit to 10000 chars)
            if len(llm_reasoning_process) <= 10000:
                logger.info(f"Full LLM Reasoning Response:\n{llm_reasoning_process}")
            else:
                logger.info(f"Full LLM Reasoning Response (first 10000 chars):\n{llm_reasoning_process[:10000]}...")
            # Force flush to ensure immediate write to log file
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
        
        # Infer root cause
        inferred_result = self._parse_llm_root_cause_inference(llm_reasoning_process)
        
        # Validation via external tools (Git Blame, Code Search, etc.)
        evidence_sources = self._gather_external_evidence(
            inferred_result, cluster_info, cluster_features
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            inferred_result, cluster_info, cluster_features, evidence_sources
        )
        
        # Legacy function: create one inference per localId in cluster
        # For backward compatibility, return a list (but typically only one localId)
        if len(cluster_info.localIds) == 1:
            # Single localId: create one inference
            localId = cluster_info.localIds[0]
            dep_name = None
            dep_version = None
            if isinstance(inferred_result.get('root_cause_dependency'), dict):
                dep_name = inferred_result.get('root_cause_dependency', {}).get('name')
            elif isinstance(inferred_result.get('root_cause_dependency'), str):
                dep_name = inferred_result.get('root_cause_dependency')
            
            root_cause_inference = RootCauseInference(
                localId=localId,
                inferred_root_cause_type=inferred_result.get('root_cause_type', 'Unknown'),
                inferred_root_cause_dependency=dep_name,
                inferred_root_cause_dependency_version=dep_version,
                llm_reasoning_process=llm_reasoning_process,
                confidence_score=confidence_score,
                main_project_score=0.5 if inferred_result.get('root_cause_type') == 'Main_Project_Specific' else 0.0,
                dependency_score=0.5 if inferred_result.get('root_cause_type') == 'Dependency_Specific' else 0.0,
                evidence_sources=evidence_sources,
                module1_confidence=inferred_result.get('module1_confidence'),
                module2_confidence=inferred_result.get('module2_confidence'),
                module3_confidence=inferred_result.get('module3_confidence'),
                discrepancy_type=inferred_result.get('discrepancy_type'),
                corrective_reasoning=inferred_result.get('corrective_reasoning')
            )
        else:
            # Multiple localIds: create one inference per localId (use first one as representative)
            # This is a legacy path, should not be used in new code
            localId = cluster_info.localIds[0]
            dep_name = None
            if isinstance(inferred_result.get('root_cause_dependency'), dict):
                dep_name = inferred_result.get('root_cause_dependency', {}).get('name')
            elif isinstance(inferred_result.get('root_cause_dependency'), str):
                dep_name = inferred_result.get('root_cause_dependency')
            
            root_cause_inference = RootCauseInference(
                localId=localId,
                inferred_root_cause_type=inferred_result.get('root_cause_type', 'Unknown'),
                inferred_root_cause_dependency=dep_name,
                llm_reasoning_process=llm_reasoning_process,
                confidence_score=confidence_score,
                main_project_score=0.5 if inferred_result.get('root_cause_type') == 'Main_Project_Specific' else 0.0,
                dependency_score=0.5 if inferred_result.get('root_cause_type') == 'Dependency_Specific' else 0.0,
                evidence_sources=evidence_sources,
                module1_confidence=inferred_result.get('module1_confidence'),
                module2_confidence=inferred_result.get('module2_confidence'),
                module3_confidence=inferred_result.get('module3_confidence'),
                discrepancy_type=inferred_result.get('discrepancy_type'),
                corrective_reasoning=inferred_result.get('corrective_reasoning')
            )
        
        print(f"  [+] Root cause inference complete")
        return root_cause_inference
    
    def _find_shared_dependencies(self, localIds: List[int], project_name: Optional[str] = None) -> List[Dict]:
        """Find shared dependencies within cluster (with path-based filtering)"""
        all_dependencies = []
        
        for localId in localIds:
            try:
                data = extract_data(localId, include_code_snippets=False, auto_fetch=False)
                if data:
                    srcmap = data.get('srcmap', {})
                    vulnerable_deps = srcmap.get('vulnerable_version', {}).get('dependencies', [])
                    # Apply path-based filtering
                    filtered_deps = self._filter_dependencies(vulnerable_deps, project_name)
                    all_dependencies.append(filtered_deps)
            except Exception as e:
                print(f"  [!] Error extracting dependencies for localId {localId}: {e}")
                continue
        
        if not all_dependencies:
            return []
        
        # Find shared dependencies (by name)
        dependency_names = defaultdict(int)
        dependency_info = {}
        
        for deps in all_dependencies:
            seen_names = set()
            for dep in deps:
                dep_name = dep.get('name', '')
                if dep_name and dep_name not in seen_names:
                    dependency_names[dep_name] += 1
                    if dep_name not in dependency_info:
                        dependency_info[dep_name] = dep
                    seen_names.add(dep_name)
        
        # Select only dependencies present in all cases
        shared_deps = []
        for dep_name, count in dependency_names.items():
            if count == len(all_dependencies):  # Present in all cases
                shared_deps.append(dependency_info[dep_name])
        
        return shared_deps
    
    def _generate_root_cause_reasoning(self, cluster_info: ClusterInfo,
                                      cluster_features: List[VulnerabilityFeatures],
                                      common_dependencies: List[Dict],
                                      logger: Optional[logging.Logger] = None,
                                      gt_info: Optional[List[Dict]] = None) -> str:
        """Generate LLM Chain-of-Thought for root cause inference"""
        
        # Group features by bug_type for better analysis
        features_by_bug_type = {}
        for f in cluster_features:
            bug_type = f.bug_type
            if bug_type not in features_by_bug_type:
                features_by_bug_type[bug_type] = []
            features_by_bug_type[bug_type].append(f)
        
        # Build summary grouped by bug_type (ordered by frequency)
        bug_type_order = sorted(features_by_bug_type.items(), key=lambda x: len(x[1]), reverse=True)
        
        features_summary = []
        bug_type_distribution = []
        
        for bug_type, type_features in bug_type_order:
            count = len(type_features)
            bug_type_distribution.append(f"{bug_type} ({count} cases)")
            
            features_summary.append(f"\n{'='*80}")
            features_summary.append(f"Bug Type Group: {bug_type} ({count} vulnerabilities)")
            features_summary.append(f"{'='*80}")
            
            # Use summarized information for token efficiency
            for f in type_features:
                features_summary.append(f"""
localId {f.localId} (Project: {f.project_name}):
- Stack Trace Summary: {f.stack_trace_summary[:300]}
- Patch Summary: {f.patch_summary[:300]}
- Dependencies Summary: {f.dependencies_summary[:250]}
- Initial Reasoning: {f.llm_reasoning_summary[:250]}...
""")
        
        deps_info = []
        for dep in common_dependencies[:10]:
            deps_info.append(f"- {dep.get('name', 'N/A')} ({dep.get('path', 'N/A')})")
        
        prompt = f"""Using Chain-of-Thought reasoning, perform deep semantic inference to determine the root cause of vulnerabilities in this cluster.

Cluster Information:
- Cluster ID: {cluster_info.cluster_id}
- Number of vulnerabilities: {len(cluster_info.localIds)}
- Bug Type Distribution: {', '.join(bug_type_distribution)}
- Common characteristics: {cluster_info.common_characteristics}
- Common dependencies: {', '.join(cluster_info.common_dependencies) if cluster_info.common_dependencies else 'None'}

Vulnerability Details (Grouped by Bug Type):
{chr(10).join(features_summary)}

Shared Dependencies:
{chr(10).join(deps_info) if deps_info else 'None identified'}
{f'''

Heuristic Ground Truth (for reference and discrepancy analysis):
{chr(10).join([f"- localId {gt['localId']}: {gt['gt_type']} ({gt['gt_dependency']})" for gt in gt_info]) if gt_info else 'No heuristic GT available'}
''' if gt_info else ''}

Perform a detailed analysis following these steps:

**Module 1: Bug Type Group Analysis (Macro-level Pattern Recognition)**
*Purpose: This macro-level analysis establishes the foundation for efficient analysis paths when the benchmark collects large volumes of new vulnerability tasks. It enables initial classification and pattern recognition.*

For each bug type group provided above, analyze:
1. Where do the crashes consistently occur (main project code vs dependency code)?
2. What do the stack traces reveal about the crash location?
3. What do the patches fix and where are the fixes applied?
4. Are there patterns within this bug type group?

*Note: This module uses summarized information to identify macro-level patterns efficiently, conserving tokens for more detailed analysis in subsequent modules.*

**Output Requirements:**
- Provide a confidence score (0.0-1.0) for your Module 1 analysis
- Identify common dependencies that should be prioritized for attention in Module 2

**Module 2: Fine-Grained Sub-Grouping within Bug Type Groups (Micro-level Deep Analysis)**
*Purpose: This fine-grained sub-grouping builds deep knowledge for specific problem types, enabling the benchmark to autonomously handle similar patterns and strengthen its adaptive capabilities. This directly contributes to clustering accuracy evaluation (RQ3.3.2).*

*Context Management: This module uses detailed information (llm_reasoning_summary, affected_module_pattern, code snippets) for precise analysis. If you need additional context, you can request specific information through Tool Use.*

Using your reasoning capabilities, perform detailed sub-grouping within each bug type group:
1. For each bug type group, analyze the vulnerabilities and identify meaningful sub-groups:
   - Group vulnerabilities that share similar crash patterns (same stack trace patterns)
   - Group vulnerabilities that share similar patch locations (same files/repos modified)
   - Group vulnerabilities that share similar dependency usage patterns
   - Group vulnerabilities that point to the same root cause (main project vs dependency)
2. For each identified sub-group, determine:
   - What makes this sub-group distinct from others?
   - What is the likely root cause for this sub-group?
   - Is it Main_Project_Specific or Dependency_Specific?
3. **Prioritize dependencies** identified in Module 1's common_dependencies list when analyzing root causes
4. Output your sub-grouping analysis in the following format:
   ```
   Bug Type Group: [bug_type]
   Sub-Group 1:
   - LocalIds: [list of localIds]
   - Root Cause Type: [Main_Project_Specific or Dependency_Specific]
   - Root Cause Dependency: [dependency name or N/A]
   - Reasoning: [why these vulnerabilities form a sub-group and what the root cause is]
   
   Sub-Group 2:
   - LocalIds: [list of localIds]
   ...
   ```

**Module 3: Cross-Group Root Cause Inference (Final Synthesis & Autonomous Task Generation)**
*Purpose: This final root cause inference and cross-project pattern analysis provides the core mechanism for the benchmark to autonomously generate new evaluation tasks and continuously expand its knowledge base. This directly contributes to root cause tracking accuracy (RQ3.4.1) and reasoning quality evaluation (RQ3.4.2).*

Based on Module 1 and 2 analysis, perform final root cause inference:
1. **Information Flow Integration:**
   - Use Module 1's common_dependencies as prioritized candidates for root cause analysis
   - Incorporate Module 2's sub-group-level root cause reasoning into your final inference
   - Consider confidence scores from Module 1 and 2: if confidence is low, request more detailed information

2. Compare patterns across bug type groups and sub-groups:
   - Do different bug types/sub-groups point to the same root cause?
   - Are there consistent patterns in stack traces or patch locations across groups?
   - Do all groups show similar dependency usage patterns?

3. Analyze shared dependencies:
   - Are the crashes occurring IN the dependency code or in how the main project uses the dependency?
   - Do patches modify dependency code or main project code?

4. Distinguish between superficial indications and true root causes:
   - Just because a dependency is used does NOT mean the bug is in the dependency
   - The bug could be in how the main project uses the dependency (Main_Project_Specific)
   - Look for evidence: Are patches in dependency repos? Do stack traces show dependency code paths?

5. Consider functional context of dependencies (e.g., image processing, networking, parsing)

6. **Discrepancy Analysis with Heuristic Ground Truth:**
   If heuristic GT is available and differs from your inference:
   - Classify the discrepancy type:
     * "heuristic_error": Heuristic GT misidentified due to missing srcmap data or mapping failures
     * "llm_error": Your analysis may be incorrect
     * "borderline_case": Ambiguous case where both could be valid
   - Provide "Corrective Reasoning": Explain why your inference is more accurate than heuristic GT
     Format: "The heuristic GT suggests [GT's conclusion], but my analysis indicates [your conclusion] is more accurate because [detailed reasoning with evidence]"

**Important Decision Criteria:**
- Bug type consistency: If all vulnerabilities share the same bug type and occur in dependency code paths, it's likely Dependency_Specific
- Patch location: If patches are in dependency repositories, it's likely Dependency_Specific
- Stack trace patterns: If crashes consistently occur in dependency functions, analyze whether it's a dependency bug or misuse by the main project
- Cross-group analysis: If multiple bug types/sub-groups point to the same dependency or main project pattern, that strengthens the root cause inference
- Sub-grouping consistency: If sub-groups within a bug type group have different root causes, analyze which is more prevalent or if they should be treated separately

**Final Output Format:**
Provide your final analysis in the following format:
ROOT_CAUSE_TYPE: [Main_Project_Specific or Dependency_Specific]
ROOT_CAUSE_DEPENDENCY: [dependency name if Dependency_Specific, or "N/A" if Main_Project_Specific]
MODULE1_CONFIDENCE: [0.0-1.0]
MODULE2_CONFIDENCE: [0.0-1.0]
MODULE3_CONFIDENCE: [0.0-1.0]
REASONING:
[Detailed Chain-of-Thought reasoning explaining why this is the root cause, structured as:
  Module 1 Analysis: [Analysis of each bug type group with confidence score]
  Module 2 Sub-Grouping: [Fine-grained sub-grouping analysis within each bug type group with confidence score]
  Module 3 Cross-Group Inference: [Final root cause determination based on cross-group and sub-group analysis with confidence score]
]
DISCREPANCY_ANALYSIS: [If heuristic GT differs, provide:
  DISCREPANCY_TYPE: [heuristic_error, llm_error, or borderline_case]
  CORRECTIVE_REASONING: [Detailed rebuttal explaining why your inference is more accurate]
]
EVIDENCE:
[List of evidence sources: bug type analysis, sub-grouping analysis, stack trace analysis, patch analysis, dependency analysis, etc.]"""
        
        return self.call_llm(prompt, logger=logger)
    
    def _parse_llm_root_cause_inference(self, llm_response: str) -> Dict:
        """Parse root cause inference results from LLM response"""
        result = {
            'root_cause_type': 'Unknown',
            'root_cause_dependency': None,
            'module1_confidence': None,
            'module2_confidence': None,
            'module3_confidence': None,
            'discrepancy_type': None,
            'corrective_reasoning': None
        }
        
        llm_response_lower = llm_response.lower()
        lines = llm_response.split('\n')
        
        # Parse ROOT_CAUSE_TYPE (explicit format)
        for line in lines:
            line_upper = line.upper()
            if 'ROOT_CAUSE_TYPE:' in line_upper or 'ROOT CAUSE TYPE:' in line_upper:
                line_lower = line.lower()
                if 'main_project_specific' in line_lower or 'main project specific' in line_lower:
                    result['root_cause_type'] = 'Main_Project_Specific'
                elif 'dependency_specific' in line_lower or 'dependency specific' in line_lower:
                    result['root_cause_type'] = 'Dependency_Specific'
                break
        
        # Parse ROOT_CAUSE_DEPENDENCY (explicit format)
        for line in lines:
            line_upper = line.upper()
            if 'ROOT_CAUSE_DEPENDENCY:' in line_upper or 'ROOT CAUSE DEPENDENCY:' in line_upper:
                if ':' in line:
                    dep_name = line.split(':', 1)[1].strip()
                else:
                    dep_name = line.strip()
                
                if dep_name and dep_name.lower() not in ['n/a', 'none', 'null', '']:
                    result['root_cause_dependency'] = {'name': dep_name}
                    break
        
        # Search for more general patterns (when explicit format not found)
        if result['root_cause_type'] == 'Unknown':
            if 'main_project_specific' in llm_response_lower or 'main project specific' in llm_response_lower:
                result['root_cause_type'] = 'Main_Project_Specific'
            elif 'dependency_specific' in llm_response_lower or 'dependency specific' in llm_response_lower:
                result['root_cause_type'] = 'Dependency_Specific'
            elif 'main project' in llm_response_lower and 'specific' in llm_response_lower:
                # Check if "main project" appears before "dependency"
                main_idx = llm_response_lower.find('main project')
                dep_idx = llm_response_lower.find('dependency')
                if main_idx < dep_idx or dep_idx == -1:
                    result['root_cause_type'] = 'Main_Project_Specific'
            elif 'dependency' in llm_response_lower and 'specific' in llm_response_lower:
                result['root_cause_type'] = 'Dependency_Specific'
        
        # Extract dependency name (general patterns)
        if result['root_cause_type'] == 'Dependency_Specific' and not result['root_cause_dependency']:
            # Find common dependency name patterns
            common_deps = ['libpng', 'libjpeg', 'zlib', 'openssl', 'libxml', 'libxslt',
                          'qtbase', 'qtsvg', 'libheif', 'libjxl', 'libvips', 'libspdm',
                          'qtsvg', 'qtimageformats']
            for dep in common_deps:
                if dep.lower() in llm_response_lower:
                    result['root_cause_dependency'] = {'name': dep}
                    break
        
        # Parse confidence scores
        for line in lines:
            line_upper = line.upper()
            if 'MODULE1_CONFIDENCE:' in line_upper or 'MODULE 1 CONFIDENCE:' in line_upper:
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    result['module1_confidence'] = float(conf_str)
                except:
                    pass
            elif 'MODULE2_CONFIDENCE:' in line_upper or 'MODULE 2 CONFIDENCE:' in line_upper:
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    result['module2_confidence'] = float(conf_str)
                except:
                    pass
            elif 'MODULE3_CONFIDENCE:' in line_upper or 'MODULE 3 CONFIDENCE:' in line_upper:
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    result['module3_confidence'] = float(conf_str)
                except:
                    pass
        
        # Parse discrepancy analysis
        if 'DISCREPANCY_ANALYSIS:' in llm_response_lower or 'DISCREPANCY ANALYSIS:' in llm_response_lower:
            # Extract discrepancy type
            if 'heuristic_error' in llm_response_lower:
                result['discrepancy_type'] = 'heuristic_error'
            elif 'llm_error' in llm_response_lower:
                result['discrepancy_type'] = 'llm_error'
            elif 'borderline_case' in llm_response_lower:
                result['discrepancy_type'] = 'borderline_case'
            
            # Extract corrective reasoning
            if 'CORRECTIVE_REASONING:' in llm_response_lower or 'CORRECTIVE REASONING:' in llm_response_lower:
                # Find the section and extract text
                reasoning_start = llm_response_lower.find('corrective_reasoning:') or llm_response_lower.find('corrective reasoning:')
                if reasoning_start >= 0:
                    reasoning_text = llm_response[reasoning_start + len('corrective_reasoning:'):]
                    # Extract until next major section or end
                    next_section = min(
                        reasoning_text.find('\nEVIDENCE:'),
                        reasoning_text.find('\nEvidence:'),
                        len(reasoning_text)
                    )
                    result['corrective_reasoning'] = reasoning_text[:next_section].strip()
        
        return result
    
    def _gather_external_evidence(self, inferred_result: Dict,
                                 cluster_info: ClusterInfo,
                                 cluster_features: List[VulnerabilityFeatures]) -> List[str]:
        """Gather evidence via external tools (Git Blame, Code Search, etc.)"""
        evidence_sources = []
        
        # Stack trace analysis
        evidence_sources.append("Stack trace analysis")
        
        # Patch analysis
        evidence_sources.append("Patch diff analysis")
        
        # Dependency analysis
        evidence_sources.append("Dependency analysis via srcmap.json")
        
        return evidence_sources
    
    def _calculate_confidence_score(self, inferred_result: Dict,
                                   cluster_info: ClusterInfo,
                                   cluster_features: List[VulnerabilityFeatures],
                                   evidence_sources: List[str]) -> float:
        """Calculate confidence score"""
        score = 0.0
        
        # Score based on number of evidence sources
        score += len(evidence_sources) * 0.1
        
        # Score based on cluster size (more cases = higher confidence)
        if len(cluster_info.localIds) >= 5:
            score += 0.3
        elif len(cluster_info.localIds) >= 3:
            score += 0.2
        else:
            score += 0.1
        
        # Presence of common dependencies
        if cluster_info.common_dependencies:
            score += 0.2
        
        # Clarity of LLM inference (simple heuristic)
        if inferred_result.get('root_cause_type') != 'Unknown':
            score += 0.3
        
        return min(1.0, score)
    
    def _prepare_gt_info(self, bug_type_groups: List[BugTypeGroupInfo],
                        ground_truth: Optional[Dict]) -> Optional[List[Dict]]:
        """Prepare GT information from bug type groups"""
        if not ground_truth:
            return None
        
        all_localIds = []
        for bg in bug_type_groups:
            all_localIds.extend(bg.localIds)
        
        return self._prepare_gt_info_for_localIds(all_localIds, ground_truth)
    
    def _prepare_gt_info_for_localIds(self, localIds: List[int],
                                     ground_truth: Optional[Dict]) -> Optional[List[Dict]]:
        """Prepare GT information for a list of localIds"""
        if not ground_truth:
            return None
        
        gt_list = []
        for localId in localIds:
            if localId in ground_truth:
                gt = ground_truth[localId]
                gt_dep = gt.get('Heuristically_Root_Cause_Dependency', {})
                dep_name = 'N/A'
                if isinstance(gt_dep, dict):
                    dep_name = gt_dep.get('name', 'N/A')
                
                gt_list.append({
                    'localId': localId,
                    'gt_type': gt.get('Heuristically_Root_Cause_Type', 'Unknown'),
                    'gt_dependency': dep_name
                })
        
        return gt_list if gt_list else None

def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj

def _save_checkpoint(checkpoint_file: str, features_list: List, cluster_infos: List, 
                    root_cause_inferences: List, logger: logging.Logger,
                    bug_type_groups: Optional[List] = None, sub_groups: Optional[List] = None):
    """Save intermediate checkpoint"""
    try:
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_vulnerabilities': len(features_list),
                'total_clusters': len(cluster_infos),
                'total_inferences': len(root_cause_inferences),
                'total_bug_type_groups': len(bug_type_groups) if bug_type_groups else 0,
                'total_sub_groups': len(sub_groups) if sub_groups else 0
            },
            'features': [_convert_to_serializable(asdict(f)) if hasattr(f, '__dict__') else f for f in features_list],
            'clusters': [_convert_to_serializable(asdict(c)) if hasattr(c, '__dict__') else c for c in cluster_infos],
            'root_cause_inferences': [_convert_to_serializable(asdict(r)) if hasattr(r, '__dict__') else r for r in root_cause_inferences]
        }
        
        if bug_type_groups:
            checkpoint['bug_type_groups'] = [_convert_to_serializable(asdict(bg)) if hasattr(bg, '__dict__') else bg for bg in bug_type_groups]
        if sub_groups:
            checkpoint['sub_groups'] = [_convert_to_serializable(asdict(sg)) if hasattr(sg, '__dict__') else sg for sg in sub_groups]
        
        # Keep semantic_embedding for clustering (needed for resume)
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description='LLM-Based Inference Modules')
    parser.add_argument('--gt-file', type=str, default='ground_truth.json', help='Ground Truth JSON file')
    parser.add_argument('--localIds', type=str, nargs='+', help='Specific localIds to process (comma-separated or space-separated)')
    parser.add_argument('--project', type=str, help='Process all cases from a project')
    parser.add_argument('--bug-type', type=str, help='Process all cases with a specific bug_type (crash_type)')
    parser.add_argument('-n', '--num', type=int, default=None, help='Number of cases to process (default: None = all)')
    parser.add_argument('--offset', type=int, default=0, help='Offset for batch processing (default: 0)')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters (default: auto)')
    parser.add_argument('--llm-api-key', type=str, help='LLM API key (if not provided, tries OPENAI_API_KEY env var)')
    parser.add_argument('--llm-model', type=str, default='o4-mini', help='LLM model to use (default: o4-mini)')
    parser.add_argument('-o', '--output', type=str, default='llm_inference_results.json', help='Output file')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature extraction (use existing features)')
    parser.add_argument('--skip-clustering', action='store_true', help='Skip clustering (use existing clusters)')
    parser.add_argument('--checkpoint-interval', type=int, default=50, help='Save checkpoint every N vulnerabilities (default: 50)')
    parser.add_argument('--log-file', type=str, help='Log file path (default: {output}.log)')
    parser.add_argument('--resume-from', type=str, help='Resume from checkpoint file')
    parser.add_argument('--include-code-snippets', action='store_true', help='Include code snippets extraction (can be slow/timeout, default: False)')
    parser.add_argument('--module', type=str, choices=['1', '2', '3', 'all'], default='all',
                       help='Run specific module only: 1 (Bug Type Group Analysis), 2 (Sub-Grouping), 3 (Root Cause Inference), or all (default: all)')
    parser.add_argument('--stop-after-module', type=int, choices=[1, 2, 3],
                       help='Stop after completing specified module (1, 2, or 3)')
    # Paper mode and optimization options
    parser.add_argument('--paper-mode', action='store_true', help='Enable all LLM summaries (paper/report mode, default: all summaries enabled)')
    parser.add_argument('--no-patch-summary', action='store_true', help='Disable patch summary (experiment mode, faster)')
    parser.add_argument('--no-dependency-description', action='store_true', help='Disable dependency description (experiment mode, faster)')
    parser.add_argument('--no-reasoning-summary', action='store_true', help='Disable LLM reasoning summary (experiment mode, faster, longest step)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or args.output.replace('.json', '.log')
    
    # Create file handler with immediate flush
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # DEBUG level to capture all LLM responses
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Ensure immediate flush for file handler
    file_handler.flush()
    logger.info(f"Starting LLM inference modules")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Log file: {log_file}")
    
    # Initialize LLM module
    # API key priority: 1) --llm-api-key argument, 2) OPENAI_API_KEY env var
    api_key = args.llm_api_key or os.getenv('OPENAI_API_KEY')
    llm_modules = LLMInferenceModules(
        llm_api_key=api_key,
        llm_model=args.llm_model
    )
    logger.info(f"Using LLM API with model: {args.llm_model}")
    
    # Load Ground Truth for discrepancy analysis
    gt_dict = {}
    gt_data = None
    try:
        with open(args.gt_file, 'r') as f:
            gt_data = json.load(f)
        gt_list = gt_data.get('ground_truth', [])
        
        # Normalize GT: If project_name == dependency name, it's Main_Project_Specific
        normalized_gt_list = []
        for gt in gt_list:
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
                if logger:
                    logger.debug(f"Normalized GT for localId {normalized_gt.get('localId')}: "
                               f"project_name ({project_name}) == dependency name, "
                               f"set to Main_Project_Specific with dependency=None")
            
            normalized_gt_list.append(normalized_gt)
        
        gt_dict = {gt.get('localId'): gt for gt in normalized_gt_list}
        logger.info(f"Loaded {len(gt_dict)} Ground Truth entries (normalized) for discrepancy analysis")
    except Exception as e:
        logger.warning(f"Could not load Ground Truth file: {e}")
        gt_dict = {}
        gt_data = None
    
    # Determine localIds to process
    # Always filter by GT if available (original behavior: GT-only processing)
    if args.localIds:
        # Support both comma-separated string and space-separated list
        localIds = []
        for item in args.localIds:
            # Split by comma if it's a comma-separated string
            if ',' in item:
                localIds.extend([int(x.strip()) for x in item.split(',') if x.strip()])
            else:
                localIds.append(int(item))
        # Filter by GT if available
        if gt_dict:
            before_count = len(localIds)
            localIds = [lid for lid in localIds if lid in gt_dict]
            if before_count != len(localIds):
                print(f"[*] Filtered {before_count - len(localIds)} localIds not in GT, remaining: {len(localIds)}")
                logger.info(f"Filtered {before_count - len(localIds)} localIds not in GT")
    elif args.project:
        # Extract by project from DB, then filter by GT
        conn = sqlite3.connect(DB_PATH)
        try:
            if args.num and args.num > 0:
                query = "SELECT localId FROM arvo WHERE reproduced = 1 AND project = ? ORDER BY localId DESC LIMIT ?"
                cursor = conn.execute(query, (args.project, args.num))
            else:
                query = "SELECT localId FROM arvo WHERE reproduced = 1 AND project = ? ORDER BY localId DESC"
                cursor = conn.execute(query, (args.project,))
            db_localIds = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
        
        # Filter by GT if available (original behavior: GT-only)
        if gt_dict:
            localIds = [lid for lid in db_localIds if lid in gt_dict]
            print(f"[+] Found {len(db_localIds)} cases with project: {args.project}")
            print(f"[+] After GT filtering: {len(localIds)} cases")
            logger.info(f"Filtering by project: {args.project}, found {len(db_localIds)} in DB, {len(localIds)} in GT")
        else:
            localIds = db_localIds
            print(f"[+] Found {len(localIds)} cases with project: {args.project} (no GT file, using all)")
            logger.warning("No GT file provided, using all cases from DB")
    elif args.bug_type:
        # Extract by bug_type from DB, then filter by GT
        conn = sqlite3.connect(DB_PATH)
        try:
            if args.num and args.num > 0:
                query = "SELECT localId FROM arvo WHERE reproduced = 1 AND crash_type = ? ORDER BY localId DESC LIMIT ?"
                cursor = conn.execute(query, (args.bug_type, args.num))
            else:
                query = "SELECT localId FROM arvo WHERE reproduced = 1 AND crash_type = ? ORDER BY localId DESC"
                cursor = conn.execute(query, (args.bug_type,))
            db_localIds = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
        
        # Filter by GT if available (original behavior: GT-only)
        # Also filter by GT's bug_type field to ensure consistency
        if gt_dict:
            localIds = []
            for lid in db_localIds:
                if lid in gt_dict:
                    gt_bug_type = gt_dict[lid].get('bug_type', '')
                    # Only include if GT's bug_type matches (or is empty/None, which means not set)
                    # For bug_type filtering, we require exact match to avoid including cases with empty bug_type
                    if gt_bug_type == args.bug_type:
                        localIds.append(lid)
            print(f"[+] Found {len(db_localIds)} cases with bug_type: {args.bug_type} in DB")
            print(f"[+] After GT filtering (by localId AND bug_type match): {len(localIds)} cases")
            logger.info(f"Filtering by bug_type: {args.bug_type}, found {len(db_localIds)} in DB, {len(localIds)} in GT (with bug_type match)")
        else:
            localIds = db_localIds
            print(f"[+] Found {len(localIds)} cases with bug_type: {args.bug_type} (no GT file, using all)")
            logger.warning("No GT file provided, using all cases from DB")
    else:
        # Extract from Ground Truth (default behavior)
        if gt_data and gt_dict:
            gt_list = gt_data.get('ground_truth', [])
            # Apply offset
            start_idx = args.offset
            end_idx = args.offset + args.num if args.num and args.num > 0 else len(gt_list)
            localIds = [gt['localId'] for gt in gt_list[start_idx:end_idx]]
            
            if args.offset > 0:
                print(f"[*] Using offset: {args.offset}, processing {len(localIds)} cases")
        else:
            # GT file not available, extract from DB instead
            print(f"[!] Warning: GT file not available or failed to load. Extracting from database instead...")
            logger.warning("GT file not available, extracting from database")
            conn = sqlite3.connect(DB_PATH)
            try:
                if args.num and args.num > 0:
                    query = "SELECT localId FROM arvo WHERE reproduced = 1 ORDER BY localId DESC LIMIT ?"
                    cursor = conn.execute(query, (args.num,))
                else:
                    query = "SELECT localId FROM arvo WHERE reproduced = 1 ORDER BY localId DESC"
                    cursor = conn.execute(query)
                localIds = [row[0] for row in cursor.fetchall()]
                if args.offset > 0:
                    localIds = localIds[args.offset:]
                print(f"[+] Found {len(localIds)} cases from database")
            finally:
                conn.close()
    
    if not localIds:
        print("[-] No localIds to process")
        if gt_dict and (args.project or args.bug_type):
            print(f"[-] Note: All selected cases were filtered out because they don't exist in GT")
        return
    
    print(f"[+] Processing {len(localIds)} localIds...")
    logger.info(f"Processing {len(localIds)} localIds")
    
    # Resume from checkpoint if specified
    features_list = []
    cluster_infos = []
    root_cause_inferences = []
    bug_type_groups = []  # Initialize bug_type_groups
    all_sub_groups = []  # Initialize sub_groups
    start_idx = 0
    
    if args.resume_from:
        # Remove @ prefix if present (some shells or tools add it)
        checkpoint_path = args.resume_from.lstrip('@')
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"[*] Resuming from checkpoint: {checkpoint_path}")
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Convert dict features to VulnerabilityFeatures objects
            features_list = []
            for f_dict in checkpoint.get('features', []):
                if isinstance(f_dict, dict):
                    # Handle semantic_embedding if it's stored as a string description
                    f_dict_copy = f_dict.copy()
                    semantic_emb = f_dict_copy.get('semantic_embedding')
                    if isinstance(semantic_emb, str) and 'dimensional vector' in semantic_emb:
                        f_dict_copy['semantic_embedding'] = None  # Skip if it's just a description
                    features_list.append(VulnerabilityFeatures(**f_dict_copy))
                else:
                    features_list.append(f_dict)
            
            # Convert dict clusters to ClusterInfo objects
            cluster_infos = []
            for c_dict in checkpoint.get('clusters', []):
                if isinstance(c_dict, dict):
                    cluster_infos.append(ClusterInfo(**c_dict))
                else:
                    cluster_infos.append(c_dict)
            
            # Convert dict inferences to RootCauseInference objects
            root_cause_inferences = []
            for r_dict in checkpoint.get('root_cause_inferences', []):
                if isinstance(r_dict, dict):
                    # Support new format: Sub-Group based inference
                    if 'sub_group_id' in r_dict:
                        # New format: group-based inference
                        inference_dict = {
                            'sub_group_id': r_dict.get('sub_group_id'),
                            'bug_type_group': r_dict.get('bug_type_group', 'Unknown'),
                            'localIds': r_dict.get('localIds', []),
                            'group_level_root_cause_type': r_dict.get('group_level_root_cause_type', 'Unknown'),
                            'group_level_root_cause_dependency': r_dict.get('group_level_root_cause_dependency'),
                            'group_level_root_cause_dependency_version': r_dict.get('group_level_root_cause_dependency_version'),
                            'group_pattern_justification': r_dict.get('group_pattern_justification', ''),
                            'dependency_matching_ratio': r_dict.get('dependency_matching_ratio', 0.0),
                            'dependency_matching_count': r_dict.get('dependency_matching_count', 0),
                            'cross_project_propagation_insight': r_dict.get('cross_project_propagation_insight'),
                            'llm_reasoning_process': r_dict.get('llm_reasoning_process', ''),
                            'confidence_score': r_dict.get('confidence_score', 0.0),
                            'main_project_score': r_dict.get('main_project_score', 0.0),
                            'dependency_score': r_dict.get('dependency_score', 0.0),
                            'evidence_sources': r_dict.get('evidence_sources', []),
                            'module1_confidence': r_dict.get('module1_confidence'),
                            'module2_confidence': r_dict.get('module2_confidence'),
                            'module3_confidence': r_dict.get('module3_confidence'),
                            'discrepancy_type': r_dict.get('discrepancy_type'),
                            'corrective_reasoning': r_dict.get('corrective_reasoning'),
                            'per_localId_discrepancies': r_dict.get('per_localId_discrepancies', [])
                        }
                        root_cause_inferences.append(RootCauseInference(**inference_dict))
                    elif 'localId' in r_dict:
                        # Old format: per-localId inference (convert to Sub-Group format)
                        # Create a single-localId Sub-Group
                        inference_dict = {
                            'sub_group_id': 0,  # Dummy sub_group_id
                            'bug_type_group': 'Unknown',
                            'localIds': [r_dict.get('localId')],
                            'group_level_root_cause_type': r_dict.get('inferred_root_cause_type', 'Unknown'),
                            'group_level_root_cause_dependency': r_dict.get('inferred_root_cause_dependency'),
                            'group_level_root_cause_dependency_version': r_dict.get('inferred_root_cause_dependency_version'),
                            'group_pattern_justification': 'Converted from per-localId inference',
                            'dependency_matching_ratio': 1.0,
                            'dependency_matching_count': 1,
                            'llm_reasoning_process': r_dict.get('llm_reasoning_process', ''),
                            'confidence_score': r_dict.get('confidence_score', 0.0),
                            'main_project_score': r_dict.get('main_project_score', 0.0),
                            'dependency_score': r_dict.get('dependency_score', 0.0),
                            'evidence_sources': r_dict.get('evidence_sources', []),
                            'module1_confidence': r_dict.get('module1_confidence'),
                            'module2_confidence': r_dict.get('module2_confidence'),
                            'module3_confidence': r_dict.get('module3_confidence'),
                            'discrepancy_type': r_dict.get('discrepancy_type'),
                            'corrective_reasoning': r_dict.get('corrective_reasoning'),
                            'per_localId_discrepancies': []
                        }
                        root_cause_inferences.append(RootCauseInference(**inference_dict))
                    elif 'localIds' in r_dict:
                        # Old format: cluster-based inference (convert to Sub-Group format)
                        localIds = r_dict.get('localIds', [])
                        inference_dict = {
                            'sub_group_id': 0,  # Dummy sub_group_id
                            'bug_type_group': 'Unknown',
                            'localIds': localIds,
                            'group_level_root_cause_type': r_dict.get('inferred_root_cause_type', 'Unknown'),
                            'group_level_root_cause_dependency': r_dict.get('inferred_root_cause_dependency'),
                            'group_level_root_cause_dependency_version': r_dict.get('inferred_root_cause_dependency_version'),
                            'group_pattern_justification': 'Converted from cluster-based inference',
                            'dependency_matching_ratio': 1.0 if localIds else 0.0,
                            'dependency_matching_count': len(localIds),
                            'llm_reasoning_process': r_dict.get('llm_reasoning_process', ''),
                            'confidence_score': r_dict.get('confidence_score', 0.0),
                            'main_project_score': r_dict.get('main_project_score', 0.0),
                            'dependency_score': r_dict.get('dependency_score', 0.0),
                            'evidence_sources': r_dict.get('evidence_sources', []),
                            'module1_confidence': r_dict.get('module1_confidence'),
                            'module2_confidence': r_dict.get('module2_confidence'),
                            'module3_confidence': r_dict.get('module3_confidence'),
                            'discrepancy_type': r_dict.get('discrepancy_type'),
                            'corrective_reasoning': r_dict.get('corrective_reasoning'),
                            'per_localId_discrepancies': []
                        }
                        root_cause_inferences.append(RootCauseInference(**inference_dict))
                else:
                    root_cause_inferences.append(r_dict)
            
            # Load bug_type_groups from checkpoint if available
            bug_type_groups = []
            bug_type_groups_data = checkpoint.get('bug_type_groups', [])
            for bg_dict in bug_type_groups_data:
                if isinstance(bg_dict, dict):
                    # Convert individual_root_causes from dict to IndividualRootCause objects
                    if 'individual_root_causes' in bg_dict and isinstance(bg_dict['individual_root_causes'], dict):
                        individual_root_causes_dict = bg_dict['individual_root_causes']
                        individual_root_causes_objects = {}
                        for lid_str, rc_dict in individual_root_causes_dict.items():
                            try:
                                lid = int(lid_str) if isinstance(lid_str, str) else lid_str
                                if isinstance(rc_dict, dict):
                                    individual_root_causes_objects[lid] = IndividualRootCause(**rc_dict)
                                else:
                                    individual_root_causes_objects[lid] = rc_dict
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to convert individual_root_cause for localId {lid_str}: {e}")
                                continue
                        bg_dict['individual_root_causes'] = individual_root_causes_objects
                    bug_type_groups.append(BugTypeGroupInfo(**bg_dict))
                else:
                    bug_type_groups.append(bg_dict)
            
            # Load sub_groups from checkpoint if available
            all_sub_groups = []
            sub_groups_data = checkpoint.get('sub_groups', [])
            for sg_dict in sub_groups_data:
                if isinstance(sg_dict, dict):
                    all_sub_groups.append(SubGroupInfo(**sg_dict))
                else:
                    all_sub_groups.append(sg_dict)
            
            # Normalize localId types for comparison (int vs str)
            processed_ids = {int(f.localId) if isinstance(f.localId, str) else f.localId for f in features_list}
            original_count = len(localIds)
            
            # Normalize localIds to int for comparison
            localIds_normalized = []
            for lid in localIds:
                lid_int = int(lid) if isinstance(lid, str) else lid
                if lid_int not in processed_ids:
                    localIds_normalized.append(lid_int)
            
            localIds = localIds_normalized
            start_idx = len(features_list)
            
            filtered_count = original_count - len(localIds)
            logger.info(f"Resumed: {len(features_list)} features, {len(cluster_infos)} clusters, {len(root_cause_inferences)} inferences")
            if bug_type_groups:
                logger.info(f"Loaded {len(bug_type_groups)} bug type groups from checkpoint")
            if all_sub_groups:
                logger.info(f"Loaded {len(all_sub_groups)} sub-groups from checkpoint")
            logger.info(f"Filtered out {filtered_count} already processed localIds, remaining: {len(localIds)} localIds to process")
            print(f"[+] Resumed: {len(features_list)} features loaded")
            print(f"[+] Filtered out {filtered_count} already processed localIds, {len(localIds)} remaining to process")
            if processed_ids:
                print(f"[*] Processed localIds sample: {list(processed_ids)[:5]}...")
        except Exception as e:
            error_msg = f"Failed to load checkpoint '{checkpoint_path}': {e}"
            logger.error(error_msg)
            print(f"[!] {error_msg}")
            print(f"[!] Starting from scratch")
            logger.info("Starting from scratch")
            features_list = []
            cluster_infos = []
            root_cause_inferences = []
            bug_type_groups = []
            all_sub_groups = []
            start_idx = 0
    
    # Regenerate semantic embeddings if missing (DISABLED - embedding generation is disabled)
    # if features_list and any(f.semantic_embedding is None for f in features_list):
    #     print(f"[*] Regenerating missing semantic embeddings...")
    #     logger.info("Regenerating missing semantic embeddings")
    #     missing_count = 0
    #     success_count = 0
    #     for f in features_list:
    #         if f.semantic_embedding is None:
    #             combined_text = f"{f.project_name} {f.bug_type} {f.stack_trace_summary} {f.patch_summary} {f.dependencies_summary}"
    #             embedding = llm_modules.generate_embedding(combined_text, logger=logger)
    #             if embedding is not None:
    #                 f.semantic_embedding = embedding
    #                 success_count += 1
    #             missing_count += 1
    #     logger.info(f"Attempted to regenerate embeddings for {missing_count} features, succeeded for {success_count}")
    #     print(f"[+] Regenerated {success_count}/{missing_count} missing embeddings")
    
    # Feature Extraction (Module 3.1) - Skip if resuming and only running Module 2 or 3
    if not (args.resume_from and args.module in ['2', '3']):
        print(f"\n{'='*80}")
        print("Feature Extraction: Vulnerability Feature Extraction & Summarization")
        print(f"{'='*80}")
        logger.info("="*80)
        logger.info("Feature Extraction: Vulnerability Feature Extraction & Summarization")
        logger.info("="*80)
        
        for idx, localId in enumerate(localIds, start=start_idx):
            try:
                # Determine summary options based on paper-mode and individual flags
                # Paper mode: enable all summaries (default behavior)
                # Individual flags override paper mode
                skip_patch = args.no_patch_summary and not args.paper_mode
                skip_dep = args.no_dependency_description and not args.paper_mode
                skip_reasoning = args.no_reasoning_summary and not args.paper_mode
                
                # Try to get patch_diff from GT if available
                pre_extracted_data = None
                if gt_dict and localId in gt_dict:
                    gt_entry = gt_dict[localId]
                    gt_patch_diff = gt_entry.get('patch_diff')
                    gt_patch_file_path = gt_entry.get('patch_file_path')
                    print(f"  [*] Checking GT for localId {localId}: has_patch_diff={bool(gt_patch_diff)}, has_patch_file={bool(gt_patch_file_path)}")
                    
                    # Try to get patch_diff from GT entry or patch_file_path
                    patch_diff_to_use = None
                    if gt_patch_diff:
                        patch_diff_to_use = gt_patch_diff
                        print(f"  [+] Found patch_diff in GT for localId {localId} ({len(patch_diff_to_use)} bytes)")
                    elif gt_patch_file_path:
                        # Try to read from patch_file_path
                        try:
                            from pathlib import Path
                            patch_file = Path(gt_patch_file_path)
                            if patch_file.exists():
                                patch_diff_to_use = patch_file.read_text(encoding='utf-8', errors='ignore')
                                print(f"  [+] Loaded patch_diff from GT patch_file_path for localId {localId} ({len(patch_diff_to_use)} bytes)")
                                if logger:
                                    logger.info(f"Loaded patch_diff from GT patch_file_path for localId {localId} ({len(patch_diff_to_use)} bytes)")
                            else:
                                print(f"  [-] Patch file does not exist: {gt_patch_file_path}")
                        except Exception as e:
                            if logger:
                                logger.debug(f"Could not read patch_file_path {gt_patch_file_path} for localId {localId}: {e}")
                    
                    if patch_diff_to_use:
                        # Pre-extract data and merge GT's patch_diff
                        try:
                            pre_extracted_data = extract_data(localId, include_code_snippets=False, auto_fetch=False)
                            if pre_extracted_data and 'patch_info' in pre_extracted_data:
                                if not pre_extracted_data['patch_info'].get('patch_diff'):
                                    pre_extracted_data['patch_info']['patch_diff'] = patch_diff_to_use
                                    print(f"  [+] Using patch_diff from GT for localId {localId} ({len(patch_diff_to_use)} bytes)")
                                    if logger:
                                        logger.info(f"Using patch_diff from GT for localId {localId} ({len(patch_diff_to_use)} bytes)")
                        except Exception as e:
                            print(f"  [-] Could not pre-extract data for localId {localId}: {e}")
                            if logger:
                                logger.warning(f"Could not pre-extract data for localId {localId}: {e}")
                            pre_extracted_data = None
                elif gt_dict:
                    print(f"  [!] localId {localId} not in gt_dict (gt_dict has {len(gt_dict)} entries)")
                else:
                    print(f"  [!] gt_dict is empty or not loaded")
                
                features = llm_modules.extract_vulnerability_features(
                    localId, 
                    data=pre_extracted_data,
                    include_code_snippets=args.include_code_snippets,
                    skip_patch_summary=skip_patch,
                    skip_dependency_description=skip_dep,
                    skip_reasoning_summary=skip_reasoning,
                    ground_truth=gt_dict,
                    logger=logger
                )
                if features:
                    features_list.append(features)
                    logger.info(f"[{idx+1}/{len(localIds)+start_idx}] Extracted features for localId {localId}")
                    
                    # Save checkpoint periodically
                    if (idx + 1) % args.checkpoint_interval == 0:
                        checkpoint_file = args.output.replace('.json', f'_checkpoint_{idx+1}.json')
                        _save_checkpoint(checkpoint_file, features_list, cluster_infos, root_cause_inferences, logger)
                        logger.info(f"Checkpoint saved: {checkpoint_file}")
            except KeyboardInterrupt:
                print("\n[!] Interrupted by user")
                logger.warning("Interrupted by user")
                raise
            except Exception as e:
                error_msg = f"Error processing localId {localId}: {e}"
                print(f"[-] {error_msg}")
                logger.error(error_msg, exc_info=True)
                continue
        
        print(f"\n[+] Extracted features for {len(features_list)} vulnerabilities")
        logger.info(f"Extracted features for {len(features_list)} vulnerabilities")
        
        # Save checkpoint after feature extraction
        checkpoint_file = args.output.replace('.json', '_checkpoint_features.json')
        _save_checkpoint(checkpoint_file, features_list, cluster_infos, root_cause_inferences, logger)
        logger.info(f"Checkpoint saved after feature extraction: {checkpoint_file}")
        print(f"[+] Checkpoint saved: {checkpoint_file}")
    else:
        print(f"\n[+] Skipping Feature Extraction (resuming from checkpoint, running Module {args.module})")
        logger.info(f"Skipping Feature Extraction (resuming from checkpoint, running Module {args.module})")
        if features_list:
            print(f"[+] Using {len(features_list)} features from checkpoint")
            logger.info(f"Using {len(features_list)} features from checkpoint")
    
    # Initialize Module 1, 2 outputs
    bug_type_groups = []
    all_sub_groups = []
    
    # Load Module 1, 2 outputs from checkpoint if resuming and only running Module 3
    if args.resume_from and args.module == '3':
        try:
            with open(args.resume_from, 'r') as f:
                checkpoint = json.load(f)
            # Load bug_type_groups
            bug_type_groups_data = checkpoint.get('bug_type_groups', [])
            bug_type_groups = [BugTypeGroupInfo(**bg) if isinstance(bg, dict) else bg for bg in bug_type_groups_data]
            # Load sub_groups
            sub_groups_data = checkpoint.get('sub_groups', [])
            all_sub_groups = [SubGroupInfo(**sg) if isinstance(sg, dict) else sg for sg in sub_groups_data]
            logger.info(f"Loaded {len(bug_type_groups)} bug type groups and {len(all_sub_groups)} sub-groups from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load Module 1/2 outputs from checkpoint: {e}")
    
    # Module 1: Bug Type Group Analysis
    if args.module in ['1', 'all']:
        print(f"\n{'='*80}")
        print("Module 1: Bug Type Group Analysis (Macro-level Pattern Recognition)")
        print(f"{'='*80}")
        logger.info("="*80)
        logger.info("Module 1: Bug Type Group Analysis")
        logger.info("="*80)
        
        bug_type_groups = llm_modules.analyze_bug_type_groups(features_list, logger=logger)
        
        print(f"\n[+] Analyzed {len(bug_type_groups)} bug type groups")
        logger.info(f"Analyzed {len(bug_type_groups)} bug type groups")
        
        # Save checkpoint after Module 1
        checkpoint_file = args.output.replace('.json', '_checkpoint_module1_bugtype_groups.json')
        # Create dummy lists for compatibility
        cluster_infos = []
        _save_checkpoint(checkpoint_file, features_list, cluster_infos, root_cause_inferences, logger, bug_type_groups=bug_type_groups)
        logger.info(f"Checkpoint saved after Module 1: {checkpoint_file}")
        
        if args.stop_after_module == 1:
            print(f"\n[+] Stopping after Module 1 as requested")
            logger.info("Stopping after Module 1")
            return
    
    # Module 2: Fine-Grained Sub-Grouping
    if args.module in ['2', 'all']:
        if not bug_type_groups:
            print(f"[-] Error: Module 1 output (bug_type_groups) required for Module 2")
            print(f"    Please run Module 1 first or provide checkpoint with bug_type_groups")
            logger.error("Module 1 output required for Module 2")
            return
        
        print(f"\n{'='*80}")
        print("Module 2: Fine-Grained Sub-Grouping within Bug Type Groups")
        print(f"{'='*80}")
        logger.info("="*80)
        logger.info("Module 2: Fine-Grained Sub-Grouping")
        logger.info("="*80)
        
        all_sub_groups = []
        for bug_type_group in bug_type_groups:
            try:
                sub_groups = llm_modules.perform_fine_grained_sub_grouping(
                    bug_type_group, features_list, logger=logger
                )
                all_sub_groups.extend(sub_groups)
                logger.info(f"Generated {len(sub_groups)} sub-groups for bug type: {bug_type_group.bug_type}")
            except Exception as e:
                error_msg = f"Error performing sub-grouping for bug type {bug_type_group.bug_type}: {e}"
                print(f"[-] {error_msg}")
                logger.error(error_msg)
                continue
        
        print(f"\n[+] Generated {len(all_sub_groups)} sub-groups")
        logger.info(f"Generated {len(all_sub_groups)} sub-groups")
        
        # Save checkpoint after Module 2
        checkpoint_file = args.output.replace('.json', '_checkpoint_module2_subgroups.json')
        _save_checkpoint(checkpoint_file, features_list, cluster_infos, root_cause_inferences, logger, bug_type_groups=bug_type_groups, sub_groups=all_sub_groups)
        logger.info(f"Checkpoint saved after Module 2: {checkpoint_file}")
        
        if args.stop_after_module == 2:
            print(f"\n[+] Stopping after Module 2 as requested")
            logger.info("Stopping after Module 2")
            return
    
    # Module 3: Cross-Group Root Cause Inference
    if args.module in ['3', 'all']:
        if not bug_type_groups:
            print(f"[-] Error: Module 1 output (bug_type_groups) required for Module 3")
            print(f"    Please run Module 1 first or provide checkpoint with bug_type_groups")
            logger.error("Module 1 output required for Module 3")
            return
        if not all_sub_groups:
            print(f"[-] Warning: Module 2 output (sub_groups) not found, but continuing with Module 3")
            logger.warning("Module 2 output not found, but continuing with Module 3")
        
        print(f"\n{'='*80}")
        print("Module 3: Cross-Group Root Cause Inference & Validation")
        print(f"{'='*80}")
        logger.info("="*80)
        logger.info("Module 3: Cross-Group Root Cause Inference")
        logger.info("="*80)
        
        try:
            inferences = llm_modules.infer_cross_group_root_cause(
                bug_type_groups, all_sub_groups, features_list,
                ground_truth=gt_dict, logger=logger
            )
            root_cause_inferences.extend(inferences)
            logger.info(f"Generated {len(inferences)} Sub-Group root cause inferences")
        except Exception as e:
            error_msg = f"Error performing cross-group root cause inference: {e}"
            print(f"[-] {error_msg}")
            logger.error(error_msg)
        
        print(f"\n[+] Generated {len(root_cause_inferences)} root cause inference")
        logger.info(f"Generated {len(root_cause_inferences)} root cause inference")
        
        # Save checkpoint after Module 3
        checkpoint_file = args.output.replace('.json', '_checkpoint_module3_inference.json')
        _save_checkpoint(checkpoint_file, features_list, cluster_infos, root_cause_inferences, logger, bug_type_groups=bug_type_groups, sub_groups=all_sub_groups)
        logger.info(f"Checkpoint saved after Module 3: {checkpoint_file}")
        print(f"[+] Checkpoint saved: {checkpoint_file}")
    
    # Save results
    results = {
        'summary': {
            'total_vulnerabilities': len(features_list),
            'total_clusters': len(cluster_infos),
            'total_inferences': len(root_cause_inferences)
        },
        'features': [_convert_to_serializable(asdict(f)) for f in features_list],
        'clusters': [_convert_to_serializable(asdict(c)) for c in cluster_infos],
        'root_cause_inferences': [_convert_to_serializable(asdict(r)) for r in root_cause_inferences]
    }
    
    # Exclude semantic_embedding (too large)
    for f in results['features']:
        if 'semantic_embedding' in f and isinstance(f['semantic_embedding'], list):
            f['semantic_embedding'] = f"[{len(f['semantic_embedding'])}-dimensional vector]"
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] Results saved to: {args.output}")
    logger.info(f"Final results saved to: {args.output}")
    
    # Save final checkpoint
    final_checkpoint = args.output.replace('.json', '_checkpoint_final.json')
    _save_checkpoint(final_checkpoint, features_list, cluster_infos, root_cause_inferences, logger, bug_type_groups=bug_type_groups, sub_groups=all_sub_groups)
    logger.info(f"Final checkpoint saved: {final_checkpoint}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    print(f"Vulnerabilities processed: {len(features_list)}")
    print(f"Bug type groups: {len(bug_type_groups)}")
    print(f"Sub-groups: {len(all_sub_groups)}")
    print(f"Root cause inferences: {len(root_cause_inferences)}")
    
    # Root cause type distribution
    root_cause_types = defaultdict(int)
    for inference in root_cause_inferences:
        root_cause_types[inference.group_level_root_cause_type] += 1
    
    print(f"\nRoot Cause Type Distribution:")
    logger.info("Root Cause Type Distribution:")
    for root_type, count in sorted(root_cause_types.items()):
        percentage = (count / len(root_cause_inferences) * 100) if root_cause_inferences else 0
        print(f"  {root_type}: {count} ({percentage:.1f}%)")
        logger.info(f"  {root_type}: {count} ({percentage:.1f}%)")
    
    logger.info("="*80)
    logger.info("LLM inference modules completed successfully")
    logger.info("="*80)
    
    # Print paper metrics summary
    print_paper_metrics_summary(features_list, bug_type_groups, all_sub_groups, root_cause_inferences, logger)

def print_paper_metrics_summary(features_list: List, bug_type_groups: List, sub_groups: List, root_cause_inferences: List, logger: Optional[logging.Logger] = None):
    """Print paper metrics summary in a readable format"""
    print("\n" + "=" * 80)
    print("📊 PAPER METRICS SUMMARY (Phase 2 - LLM Inference)")
    print("=" * 80)
    
    # Basic statistics
    total_cases = len(features_list)
    num_sub_groups = len(sub_groups)
    num_bug_type_groups = len(bug_type_groups)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  • Total cases processed: {total_cases}")
    print(f"  • Bug type groups: {num_bug_type_groups}")
    print(f"  • Sub-groups formed: {num_sub_groups}")
    print(f"  • Root cause inferences: {len(root_cause_inferences)}")
    
    # Root cause type distribution
    root_cause_types = defaultdict(int)
    dependency_counts = defaultdict(int)
    for inference in root_cause_inferences:
        root_type = inference.group_level_root_cause_type
        root_cause_types[root_type] += len(inference.localIds)
        if root_type == 'Dependency_Specific':
            dep = inference.group_level_root_cause_dependency
            if dep:
                dependency_counts[dep] += len(inference.localIds)
    
    print(f"\n📊 Root Cause Type Distribution:")
    for root_type, count in sorted(root_cause_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_cases * 100) if total_cases > 0 else 0
        print(f"  • {root_type}: {count} cases ({percentage:.1f}%)")
    
    # Dependency distribution (top 5)
    if dependency_counts:
        print(f"\n📦 Top Dependencies (Dependency_Specific cases):")
        sorted_deps = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for dep, count in sorted_deps:
            print(f"  • {dep}: {count} cases")
    
    # Sub-group statistics
    sub_group_sizes = [len(sg.localIds) for sg in sub_groups]
    if sub_group_sizes:
        avg_size = sum(sub_group_sizes) / len(sub_group_sizes)
        max_size = max(sub_group_sizes)
        min_size = min(sub_group_sizes)
        print(f"\n🔗 Sub-Group Statistics:")
        print(f"  • Average sub-group size: {avg_size:.1f} cases")
        print(f"  • Largest sub-group: {max_size} cases")
        print(f"  • Smallest sub-group: {min_size} cases")
        print(f"  • Sub-groups with ≥2 cases: {sum(1 for s in sub_group_sizes if s >= 2)}")
    
    # Project distribution
    project_counts = defaultdict(int)
    for feature in features_list:
        project_counts[feature.project_name] += 1
    
    if project_counts:
        print(f"\n🏗️  Project Distribution:")
        print(f"  • Total projects: {len(project_counts)}")
        sorted_projects = sorted(project_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for project, count in sorted_projects:
            print(f"  • {project}: {count} cases")
    
    print(f"\n📝 Paper Values:")
    print(f"  • **{total_cases}** - Evaluation cases (Stage 1)")
    print(f"  • **{num_sub_groups}** - Distinct sub-groups formed")
    print(f"  • **{root_cause_types.get('Dependency_Specific', 0) / total_cases * 100:.1f}%** - Dependency_Specific prediction rate")
    print(f"  • **{root_cause_types.get('Main_Project_Specific', 0) / total_cases * 100:.1f}%** - Main_Project_Specific prediction rate")
    print(f"  • **{len(project_counts)}+** - Projects spanned")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()


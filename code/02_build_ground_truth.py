#!/usr/bin/env python3
"""
Ground Truth Building Script

Applies heuristic rules to automatically define root causes that LLM should infer.

Usage:
    python3 02_build_ground_truth.py --localId 42533949
    python3 02_build_ground_truth.py --localIds 42533949 42492074
    python3 02_build_ground_truth.py --project libspdm -n 10
    python3 02_build_ground_truth.py --all
"""

import sys
import json
import sqlite3
import re
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import os

# Import extract_from_db functions
sys.path.insert(0, str(Path(__file__).parent))
try:
    # Try importing from 01_extract_from_db (module name starts with number)
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract_from_db", Path(__file__).parent / "01_extract_from_db.py")
    extract_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract_module)
    get_report_from_db = extract_module.get_report_from_db
    get_srcmap_files = extract_module.get_srcmap_files
    get_patch_diff = extract_module.get_patch_diff
    extract_dependencies_from_srcmap = extract_module.extract_dependencies_from_srcmap
    extract_data = extract_module.extract_data
    parse_stack_trace = extract_module.parse_stack_trace
except Exception as e:
    # Fallback: try direct import
    try:
        from extract_from_db import (
            get_report_from_db,
            get_srcmap_files,
            get_patch_diff,
            extract_dependencies_from_srcmap,
            extract_data
        )
    except ImportError:
        print(f"[!] Error importing extract_from_db: {e}")
        raise

def resolve_db_path() -> str:
    """Return ARVO DB path (env override supported) as string."""
    return os.environ.get("ARVO_DB_PATH") or str((Path(__file__).resolve().parents[1] / "arvo.db"))


DB_PATH = resolve_db_path()


class GroundTruthBuilder:
    """Ground Truth building class"""
    
    def __init__(self, use_frame_attribution: bool = False, llm_modules = None):
        self.rules_satisfied = []
        self.confidence_score = 0
        self.max_score = 8.0  # Maximum score (Rule 2 amplified: 2 × 4 = 8.0, but can be higher with other rules)
        # NOTE: Frame Attribution should NOT be used in Phase 1 GT generation
        # GT must be pure heuristic baseline for fair comparison with LLM (Phase 2)
        # Frame Attribution is used in Phase 2 Stage 0 instead
        self.use_frame_attribution = use_frame_attribution
        self.llm_modules = llm_modules
        
    def rule1_patch_file_path_mapping(self, patch_diff: str, srcmap_dependencies: List[Dict], 
                                      patch_file_path: Optional[str] = None, 
                                      patched_files: Optional[List[str]] = None,
                                      project_name: str = '') -> Tuple[int, Optional[str], bool]:
        """
        Rule 1: Patch file path mapping with score system
        
        Checks changed file paths in patch diff:
        - Score 0: All files are in main project path (Main_Project_Specific)
        - Score 2: All files belong to a single dependency (Dependency_Specific)
        - Score 1: Files are mixed (some in main project, some in dependencies, or multiple dependencies)
        
        - Tries to read from patch_file_path if patch_diff is empty
        - Falls back to patched_files if patch_diff is still empty
        
        Returns:
            (score, dependency_name, indicates_main_project):
            - score: 0 (main only), 1 (mixed), or 2 (single dependency)
            - dependency_name: Name of dependency if score=2, None otherwise
            - indicates_main_project: True if score=0 (all files in main project)
        """
        # Try to read patch_diff from file if not provided
        if not patch_diff and patch_file_path:
            try:
                from pathlib import Path
                patch_file = Path(patch_file_path)
                if patch_file.exists():
                    patch_diff = patch_file.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                pass
        
        # Fallback: Use patched_files if patch_diff is still empty
        if not patch_diff and patched_files:
            # Find main project path for path-based filtering
            main_project_path = None
            if project_name:
                main_project_path = f"/src/{project_name}"
            
            # Convert relative paths to absolute and filter
            absolute_patched_files = []
            for file_path in patched_files:
                # Convert relative path to absolute if needed
                if not file_path.startswith('/src/'):
                    if project_name:
                        file_path = f"/src/{project_name}/{file_path}"
                    else:
                        continue
                absolute_patched_files.append(file_path)
            
            if not absolute_patched_files:
                return 0, None, False
            
            # CRITICAL: Path-based check first - if all files are in main project path, score 0
            if main_project_path:
                if all(f.startswith(main_project_path) for f in absolute_patched_files):
                    # All patched files are in main project path, this is Main_Project_Specific
                    return 0, None, True
            
            # Try to match patched_files with dependencies
            dependency_mapping = defaultdict(list)
            
            for file_path in absolute_patched_files:
                # CRITICAL: Skip files in main project path (already checked above)
                if main_project_path and file_path.startswith(main_project_path):
                    continue
                
                # Match with dependencies
                matched = False
                for dep in srcmap_dependencies:
                    dep_path = dep.get('path', '')
                    dep_name = dep.get('name', '')
                    
                    if dep_path and file_path.startswith(dep_path):
                        # Only map dependencies that are not the main project
                        if dep_path != main_project_path:
                            dependency_mapping[dep_name].append(file_path)
                            matched = True
                            break
                
                if not matched:
                    # Try to extract project name from path
                    if file_path.startswith('/src/'):
                        parts = file_path.split('/')
                        if len(parts) >= 3:
                            potential_project = parts[2]
                            # Only add if different from main project
                            if not main_project_path or not file_path.startswith(main_project_path):
                                dependency_mapping[potential_project].append(file_path)
            
            # Check if all files belong to a single dependency
            if len(dependency_mapping) == 1:
                dependency_name = list(dependency_mapping.keys())[0]
                matched_files = dependency_mapping[dependency_name]
                
                if len(matched_files) == len(absolute_patched_files):
                    # Additional path-based verification: check dependency path from srcmap
                    for dep in srcmap_dependencies:
                        dep_name = dep.get('name', '')
                        dep_path = dep.get('path', '')
                        
                        if dep_name.lower() == dependency_name.lower() or dep_path.endswith(f'/{dependency_name}'):
                            # Found matching dependency in srcmap
                            if dep_path == main_project_path:
                                # Dependency path matches main project path
                                return 0, None, True
                            # Verify that matched files actually start with this dependency path
                            if dep_path and all(f.startswith(dep_path) for f in matched_files):
                                return 2, dependency_name, False
                            break
                    
                    # If couldn't verify from srcmap, still return score 2 if we have a valid mapping
                    return 2, dependency_name, False
            
            # Mixed case: files belong to multiple dependencies or some unmapped
            return 1, None, False
        
        if not patch_diff:
            return 0, None, False
        
        # Find main project path for path-based filtering
        main_project_path = None
        if project_name:
            main_project_path = f"/src/{project_name}"
        
        # Extract changed file paths from patch diff
        changed_files = []
        for line in patch_diff.split('\n'):
            if line.startswith('diff --git'):
                # Format: diff --git a/path/to/file b/path/to/file
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[2].replace('a/', '', 1)
                    # Convert relative path to absolute if needed
                    if not file_path.startswith('/src/'):
                        if project_name:
                            file_path = f"/src/{project_name}/{file_path}"
                    changed_files.append(file_path)
            elif line.startswith('---') or line.startswith('+++'):
                # Format: --- a/path/to/file or +++ b/path/to/file
                file_path = line.split()[-1]
                if file_path.startswith('a/') or file_path.startswith('b/'):
                    file_path = file_path[2:]
                # Convert relative path to absolute if needed
                if not file_path.startswith('/src/'):
                    if project_name:
                        file_path = f"/src/{project_name}/{file_path}"
                if file_path.startswith('/src/'):
                    changed_files.append(file_path)
        
        if not changed_files:
            return 0, None, False
        
        # Count files in main project vs dependencies
        main_project_files = []
        dependency_files = []
        
        for file_path in changed_files:
            if main_project_path and file_path.startswith(main_project_path):
                main_project_files.append(file_path)
            else:
                dependency_files.append(file_path)
        
        # CRITICAL: Path-based check first - if all files are in main project path, score 0
        if main_project_path:
            if all(f.startswith(main_project_path) for f in changed_files):
                # All changed files are in main project path, this is Main_Project_Specific
                return 0, None, True
        
        # Map each file path to a dependency (only for dependency files, not main project files)
        dependency_mapping = defaultdict(list)
        
        for file_path in dependency_files:  # Only process dependency files
            # Match with srcmap dependencies first (most accurate)
            matched = False
            for dep in srcmap_dependencies:
                dep_path = dep.get('path', '')
                dep_name = dep.get('name', '')
                
                # Path matching
                if dep_path and file_path.startswith(dep_path):
                    # Only map dependencies that are not the main project
                    if dep_path != main_project_path:
                        dependency_mapping[dep_name].append(file_path)
                        matched = True
                        break
            
            # Fallback: Extract project name from path if not matched
            if not matched and file_path.startswith('/src/'):
                parts = file_path.split('/')
                if len(parts) >= 3:
                    potential_project = parts[2]
                    # Only add if different from main project
                    if not main_project_path or not file_path.startswith(main_project_path):
                        dependency_mapping[potential_project].append(file_path)
            
            # Additional fallback for submodule bug cases:
            # If patch_diff comes from submodule repo, files may not match srcmap paths
            # Try to match by dependency name (case-insensitive)
            if not matched:
                for dep in srcmap_dependencies:
                    dep_name = dep.get('name', '').lower()
                    dep_type = dep.get('type', '')
                    # For submodule cases, try name-based matching
                    if dep_type == 'submodule' or 'submodule' in str(dep.get('url', '')).lower():
                        # Extract potential dependency name from file path
                        if file_path.startswith('/src/'):
                            path_parts = file_path.split('/')
                            if len(path_parts) >= 3:
                                path_project = path_parts[2].lower()
                                if path_project == dep_name or dep_name in path_project or path_project in dep_name:
                                    dependency_mapping[dep.get('name', path_parts[2])].append(file_path)
                                    matched = True
                                    break
        
        # Determine score based on file distribution
        # Case 1: All files are in main project (already handled above, returns 0)
        # Case 2: All dependency files belong to a single dependency
        if len(dependency_mapping) == 1 and len(main_project_files) == 0:
            dependency_name = list(dependency_mapping.keys())[0]
            matched_files = dependency_mapping[dependency_name]
            
            # Verify all dependency files belong to this dependency
            if len(matched_files) == len(dependency_files):
                # Additional path-based verification: check dependency path from srcmap
                for dep in srcmap_dependencies:
                    dep_name = dep.get('name', '')
                    dep_path = dep.get('path', '')
                    
                    if dep_name.lower() == dependency_name.lower() or dep_path.endswith(f'/{dependency_name}'):
                        # Found matching dependency in srcmap
                        if dep_path == main_project_path:
                            # Dependency path matches main project path
                            return 0, None, True
                        # Verify that matched files actually start with this dependency path
                        if dep_path and all(f.startswith(dep_path) for f in matched_files):
                            return 2, dependency_name, False
                        break
                
                # If couldn't verify from srcmap, still return score 2 if we have a valid mapping
                return 2, dependency_name, False
        
        # Case 3: Mixed - files in main project AND dependencies, or multiple dependencies
        # Score 1 for mixed cases
        return 1, None, False
    
    def rule2_stack_trace_dominance(self, stack_trace: str, srcmap_dependencies: List[Dict], 
                                     project_name: str = '', top_n: int = 5) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Rule 2: Stack trace top frames 100% within dependency (improved version)
        
        Checks if top N frames of stack trace all belong to a single dependency.
        - Added submodule detection logic
        - Filter internal project module paths
        - Enhanced srcmap information utilization
        - Calculates COC (Crash Ownership Confidence) = dependency_frames / total_frames
        
        Args:
            stack_trace: Stack trace string
            srcmap_dependencies: List of dependencies extracted from srcmap
            project_name: Main project name (for filtering internal module paths)
            top_n: Number of top frames to analyze
        
        Returns:
            (satisfied, dependency_name, coc): Whether rule is satisfied, dependency name, and COC value
        """
        if not stack_trace:
            return False, None, None
        
        # Internal module path keywords (to filter out false dependencies)
        INTERNAL_MODULE_KEYWORDS = {
            'core', 'lib', 'src', 'include', 'test', 'tests', 'tools', 'utils', 'common',
            'epan', 'thirdparty', 'third_party', 'source', 'sources', 'internal', 'internals',
            'modules', 'module', 'components', 'component', 'submodules', 'submodule'
        }
        
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
            return False, None, None
        
        if len(stack_frames) < top_n:
            top_n = len(stack_frames)
        
        top_frames = stack_frames[:top_n]
        
        # Find main project path
        main_project_path = None
        if project_name:
            main_project_path = f"/src/{project_name}"
        
        # Map each frame to a dependency (for COC calculation, use all frames)
        dependency_mapping = defaultdict(list)
        dependency_frames_count = 0
        total_frames_count = len(stack_frames)
        
        # Process all frames for COC calculation, but only use top_n for rule satisfaction
        all_frames_processed = []
        for file_path in stack_frames:
            matched = False
            
            # Step 1: Match with srcmap dependencies first (most accurate)
            for dep in srcmap_dependencies:
                dep_path = dep.get('path', '')
                dep_name = dep.get('name', '')
                
                if dep_path and file_path.startswith(dep_path):
                    # Only map dependencies that are not the main project
                    if dep_path != main_project_path:
                        # Filter internal module paths: check if dep_name is an internal module keyword
                        dep_name_lower = dep_name.lower()
                        if dep_name_lower not in INTERNAL_MODULE_KEYWORDS:
                            dependency_mapping[dep_name].append(file_path)
                            dependency_frames_count += 1
                            matched = True
                            break
                        else:
                            # Ignore internal module paths
                            continue
            
            # Step 2: Submodule detection (submodules within main project path)
            if not matched and main_project_path and file_path.startswith(main_project_path):
                # Detect /src/{project_name}/{submodule_name}/... pattern
                remaining_path = file_path[len(main_project_path):]
                if remaining_path.startswith('/'):
                    remaining_path = remaining_path[1:]  # Remove leading '/'
                
                parts = remaining_path.split('/')
                if len(parts) >= 1:
                    potential_submodule = parts[0]
                    
                    # Check if submodule: verify if this path is registered as dependency in srcmap
                    submodule_path = f"{main_project_path}/{potential_submodule}"
                    for dep in srcmap_dependencies:
                        dep_path = dep.get('path', '')
                        if dep_path == submodule_path or dep_path.endswith(f'/{potential_submodule}'):
                            dependency_mapping[dep.get('name', potential_submodule)].append(file_path)
                            dependency_frames_count += 1
                            matched = True
                            break
                    
                    # If not recognized as submodule, treat as internal project path
                    if not matched:
                        # Check if internal module path
                        if potential_submodule.lower() not in INTERNAL_MODULE_KEYWORDS:
                            # Detect known submodule patterns
                            # 1. lib* pattern (libtommath, libpng, libjpeg, etc.)
                            # 2. Independent library name pattern (4+ chars, lowercase/number combination)
                            # 3. Path structure looks like independent library
                            is_likely_submodule = (
                                potential_submodule.startswith('lib') or
                                (len(potential_submodule) >= 4 and 
                                 potential_submodule.islower() and
                                 potential_submodule not in ['src', 'test', 'tests', 'tools', 'utils'])
                            )
                            
                            if is_likely_submodule:
                                # Estimate as submodule (e.g., libtommath, libpng)
                                dependency_mapping[potential_submodule].append(file_path)
                                dependency_frames_count += 1
                                matched = True
            
            # Step 3: Extract project name (last resort, filter internal module paths)
            if not matched and file_path.startswith('/src/'):
                # CRITICAL: Path-based check first (more accurate)
                # If file is in main project path, don't treat as dependency
                if main_project_path and file_path.startswith(main_project_path):
                    # File is in main project path, skip (this is Main_Project_Specific)
                    continue
                
                parts = file_path.split('/')
                if len(parts) >= 3:
                    potential_project = parts[2]
                    
                    # Filter internal module paths
                    if potential_project.lower() not in INTERNAL_MODULE_KEYWORDS:
                        # Only recognize as dependency if different from main project
                        # (already checked above with path-based comparison)
                        dependency_mapping[potential_project].append(file_path)
                        dependency_frames_count += 1
        
        # Calculate COC (Crash Ownership Confidence) = dependency_frames / total_frames
        coc = None
        if total_frames_count > 0:
            coc = dependency_frames_count / total_frames_count
        
        # Now check rule satisfaction using top_n frames only
        top_frames_dependency_mapping = defaultdict(list)
        for file_path in top_frames:
            # Re-map top frames to dependencies (simplified check)
            matched = False
            for dep in srcmap_dependencies:
                dep_path = dep.get('path', '')
                dep_name = dep.get('name', '')
                if dep_path and file_path.startswith(dep_path):
                    if dep_path != main_project_path:
                        dep_name_lower = dep_name.lower()
                        if dep_name_lower not in INTERNAL_MODULE_KEYWORDS:
                            top_frames_dependency_mapping[dep_name].append(file_path)
                            matched = True
                            break
            
            if not matched and main_project_path and file_path.startswith(main_project_path):
                remaining_path = file_path[len(main_project_path):]
                if remaining_path.startswith('/'):
                    remaining_path = remaining_path[1:]
                parts = remaining_path.split('/')
                if len(parts) >= 1:
                    potential_submodule = parts[0]
                    if potential_submodule.lower() not in INTERNAL_MODULE_KEYWORDS:
                        is_likely_submodule = (
                            potential_submodule.startswith('lib') or
                            (len(potential_submodule) >= 4 and 
                             potential_submodule.islower() and
                             potential_submodule not in ['src', 'test', 'tests', 'tools', 'utils'])
                        )
                        if is_likely_submodule:
                            top_frames_dependency_mapping[potential_submodule].append(file_path)
                            matched = True
            
            if not matched and file_path.startswith('/src/'):
                if main_project_path and file_path.startswith(main_project_path):
                    continue
                parts = file_path.split('/')
                if len(parts) >= 3:
                    potential_project = parts[2]
                    if potential_project.lower() not in INTERNAL_MODULE_KEYWORDS:
                        top_frames_dependency_mapping[potential_project].append(file_path)
        
        # Use top_frames_dependency_mapping for rule satisfaction check
        dependency_mapping = top_frames_dependency_mapping
        
        # Check if all frames belong to a single dependency
        if len(dependency_mapping) == 1:
            dependency_name = list(dependency_mapping.keys())[0]
            matched_files = dependency_mapping[dependency_name]
            
            # Verify all top frames belong to this dependency
            if len(matched_files) == len(top_frames):
                # CRITICAL: Path-based check (more accurate than name-based)
                # Check if all matched files are actually in the main project path
                if main_project_path:
                    if all(f.startswith(main_project_path) for f in matched_files):
                        # All files are in main project path, this is Main_Project_Specific
                        return False, None, coc
                
                # Additional check: Verify dependency path from srcmap
                # Find the dependency in srcmap and check its path
                for dep in srcmap_dependencies:
                    dep_name = dep.get('name', '')
                    dep_path = dep.get('path', '')
                    
                    if dep_name.lower() == dependency_name.lower() or dep_path.endswith(f'/{dependency_name}'):
                        # Found matching dependency in srcmap
                        if dep_path == main_project_path:
                            # Dependency path matches main project path
                            return False, None, coc
                        # Verify that matched files actually start with this dependency path
                        if dep_path and not all(f.startswith(dep_path) for f in matched_files):
                            # Files don't match the dependency path, might be wrong match
                            # But still return True if we have a valid dependency path
                            if dep_path != main_project_path:
                                return True, dependency_name, coc
                        break
                
                # If we couldn't verify from srcmap, use name-based check as fallback
                # (but this should be rare if srcmap is complete)
                if project_name and dependency_name.lower() == project_name.lower():
                    # Name matches project name, likely Main_Project_Specific
                    return False, None, coc
                
                return True, dependency_name, coc
        
        return False, None, coc
    
    def _compute_patch_crash_distance(self, stack_trace: str, patched_files: List[str], 
                                       project_name: str = '') -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Compute structural distance between patch location and crash location.
        
        Returns:
            (distance, crash_module, patched_module):
            - distance: 0 (same file/function), 1 (same module), 2 (different modules), 3+ (wrapper/validation)
            - crash_module: Module name where crash occurred
            - patched_module: Module name where patch was applied
        """
        if not stack_trace or not patched_files:
            return None, None, None
        
        # Extract crash location from stack trace (top frame)
        # IMPROVED: Check more lines (up to 50) because crash location may appear after INFO lines
        crash_file_path = None
        crash_module = None
        lines = stack_trace.split('\n')
        for line in lines[:50]:  # Increased from 10 to 50 to find crash location after INFO lines
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
    
    def rule3_dependency_update_commit(self, submodule_bug: bool, repo_addr: str, project: str) -> Tuple[bool, Optional[str]]:
        """
        Rule 3: Dependency update commit
        
        Determines dependency update if submodule_bug=True or repo_addr differs from project.
        
        Returns:
            (satisfied, dependency_name): Whether rule is satisfied and the dependency name
        """
        if submodule_bug:
            # Extract dependency name from repo_addr
            # Example: git://code.qt.io/qt/qtbase.git -> qtbase
            if repo_addr:
                # Extract last part from URL
                parts = repo_addr.rstrip('/').split('/')
                if parts:
                    dep_name = parts[-1].replace('.git', '')
                    return True, dep_name
            return True, None
        
        # Also check if repo_addr differs from project
        if repo_addr and project:
            # Extract project name from repo_addr
            repo_project = None
            if '/src/' in repo_addr or 'github.com' in repo_addr or 'code.qt.io' in repo_addr:
                parts = repo_addr.rstrip('/').split('/')
                if parts:
                    repo_project = parts[-1].replace('.git', '')
            
            if repo_project and repo_project.lower() != project.lower():
                return True, repo_project
        
        return False, None
    
    def rule4_external_cve_connection(self, local_id: int, project: str, fix_commit: str) -> Tuple[bool, Optional[str]]:
        """
        Rule 4: External CVE/NVD direct connection
        
        Checks if there is a CVE connected to this vulnerability in external CVE/NVD database.
        (Currently only basic implementation provided, actual CVE API call needed)
        
        Returns:
            (satisfied, cve_id): Whether rule is satisfied and the CVE ID
        """
        # TODO: Need actual CVE/NVD API integration
        # Currently only basic implementation provided
        return False, None
    
    def analyze_patch_type(self, patch_diff: str, patched_files: List[str]) -> Dict[str, any]:
        """
        Analyze patch content to detect patch type and whether it's a version update bypass
        
        This is critical for detecting cases where LLM might suggest just updating dependency version
        instead of fixing the actual root cause.
        
        Returns:
            {
                'patch_type': 'version_update' | 'code_fix' | 'mixed' | 'unknown',
                'is_version_update_bypass': bool,
                'version_update_files': List[str],
                'code_fix_files': List[str],
                'confidence': float  # 0.0 to 1.0
            }
        """
        import re
        
        # Files that typically indicate version updates
        VERSION_UPDATE_PATTERNS = [
            r'CMakeLists\.txt',
            r'package\.json',
            r'requirements\.txt',
            r'Pipfile',
            r'poetry\.lock',
            r'Cargo\.toml',
            r'Cargo\.lock',
            r'go\.mod',
            r'go\.sum',
            r'\.gitmodules',
            r'Makefile',
            r'configure\.ac',
            r'configure\.in',
            r'\.version',
            r'VERSION',
            r'version\.txt',
        ]
        
        # Keywords that indicate version updates
        VERSION_UPDATE_KEYWORDS = [
            r'version',
            r'bump',
            r'update',
            r'upgrade',
            r'git submodule',
            r'submodule update',
            r'^\+\s*[0-9]+\.[0-9]+\.[0-9]+',  # Version number increment
            r'^\-\s*[0-9]+\.[0-9]+\.[0-9]+',
        ]
        
        version_update_files = []
        code_fix_files = []
        version_keyword_matches = 0
        
        # Analyze patched files
        for file_path in patched_files:
            is_version_file = False
            for pattern in VERSION_UPDATE_PATTERNS:
                if re.search(pattern, file_path, re.IGNORECASE):
                    is_version_file = True
                    version_update_files.append(file_path)
                    break
            
            if not is_version_file:
                code_fix_files.append(file_path)
        
        # Analyze patch diff content
        if patch_diff:
            patch_lines = patch_diff.split('\n')
            code_change_lines = 0
            version_change_lines = 0
            
            for line in patch_lines:
                # Skip diff headers and metadata
                if line.startswith('diff --git') or line.startswith('index ') or \
                   line.startswith('---') or line.startswith('+++') or \
                   line.startswith('@@') or not line.strip():
                    continue
                
                # Check for version update keywords
                for keyword in VERSION_UPDATE_KEYWORDS:
                    if re.search(keyword, line, re.IGNORECASE):
                        version_keyword_matches += 1
                        version_change_lines += 1
                        break
                
                # Check for actual code changes (not just version numbers)
                if line.startswith('+') or line.startswith('-'):
                    # Skip lines that are only version numbers or whitespace
                    content = line[1:].strip()
                    if content and not re.match(r'^[0-9]+\.[0-9]+\.[0-9]+', content):
                        code_change_lines += 1
            
            # Determine patch type
            total_changes = code_change_lines + version_change_lines
            if total_changes == 0:
                patch_type = 'unknown'
                is_bypass = False
            elif version_change_lines > code_change_lines * 2:
                # Mostly version updates
                patch_type = 'version_update'
                is_bypass = True
            elif code_change_lines > version_change_lines * 2:
                # Mostly code fixes
                patch_type = 'code_fix'
                is_bypass = False
            else:
                # Mixed
                patch_type = 'mixed'
                is_bypass = version_change_lines > 0
            
            # Confidence calculation
            if len(version_update_files) > 0 and len(code_fix_files) == 0:
                confidence = 0.9  # High confidence it's version update
            elif len(code_fix_files) > 0 and len(version_update_files) == 0:
                confidence = 0.9  # High confidence it's code fix
            elif len(version_update_files) > 0 and len(code_fix_files) > 0:
                confidence = 0.6  # Mixed, lower confidence
            else:
                confidence = 0.5  # Unknown
            
            # Adjust confidence based on patch content analysis
            if patch_type == 'version_update' and version_keyword_matches > 3:
                confidence = min(confidence + 0.1, 1.0)
            elif patch_type == 'code_fix' and code_change_lines > 10:
                confidence = min(confidence + 0.1, 1.0)
        else:
            # No patch diff, use file analysis only
            if len(version_update_files) > 0 and len(code_fix_files) == 0:
                patch_type = 'version_update'
                is_bypass = True
                confidence = 0.8
            elif len(code_fix_files) > 0 and len(version_update_files) == 0:
                patch_type = 'code_fix'
                is_bypass = False
                confidence = 0.8
            else:
                patch_type = 'mixed' if (version_update_files and code_fix_files) else 'unknown'
                is_bypass = len(version_update_files) > 0
                confidence = 0.6
        
        return {
            'patch_type': patch_type,
            'is_version_update_bypass': is_bypass,
            'version_update_files': version_update_files,
            'code_fix_files': code_fix_files,
            'confidence': confidence
        }
    
    def build_ground_truth(self, local_id: int, data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Build Ground Truth for given local_id
        
        Args:
            local_id: Local ID to analyze
            data: Already extracted data (if None, auto-extract)
        
        Returns:
            Ground Truth dictionary
        """
        # Extract data
        if data is None:
            print(f"[*] Extracting data for localId {local_id}...")
            try:
                # First extract without code snippets, but try to auto-fetch patch diff
                # auto_fetch=True: Try to auto-download srcmap and auto-generate patch_diff if missing
                data = extract_data(local_id, include_code_snippets=False, auto_fetch=True)
                
                # CRITICAL: patch_diff is required for GT
                patch_diff = data.get('patch_info', {}).get('patch_diff', '')
                
                # If patch_diff is still empty, try to get it from patch_file_path
                if not patch_diff:
                    patch_file_path = data.get('patch_info', {}).get('patch_file_path')
                    if patch_file_path:
                        try:
                            from pathlib import Path
                            patch_file = Path(patch_file_path)
                            if patch_file.exists():
                                patch_diff = patch_file.read_text(encoding='utf-8', errors='ignore')
                                if patch_diff:
                                    # Update data with patch_diff
                                    if 'patch_info' not in data:
                                        data['patch_info'] = {}
                                    data['patch_info']['patch_diff'] = patch_diff
                                    print(f"  [*] Loaded patch_diff from patch_file_path")
                        except Exception as e:
                            print(f"  [-] Failed to read patch_diff from {patch_file_path}: {e}")
                
                # If patch_diff is still empty, try to generate it
                if not patch_diff:
                    print(f"  [*] patch_diff not found, attempting to generate...")
                    try:
                        patch_diff_file = get_patch_diff(local_id, auto_generate=True)
                        if patch_diff_file and patch_diff_file.exists():
                            patch_diff = patch_diff_file.read_text(encoding='utf-8', errors='ignore')
                            if patch_diff:
                                # Update data with patch_diff
                                if 'patch_info' not in data:
                                    data['patch_info'] = {}
                                data['patch_info']['patch_diff'] = patch_diff
                                data['patch_info']['patch_file_path'] = str(patch_diff_file)
                                print(f"  [+] Generated patch_diff successfully")
                    except Exception as e:
                        error_msg = str(e)
                        if any(mod in error_msg.lower() for mod in ['sqlalchemy', 'requests', 'tiktoken']):
                            missing_mods = [mod for mod in ['sqlalchemy', 'requests', 'tiktoken'] if mod in error_msg.lower()]
                            print(f"  [-] Failed to generate patch_diff: Missing modules: {', '.join(missing_mods)}")
                            print(f"      Please activate virtual environment: source /root/Git/ARVO/arvo-run/bin/activate")
                        else:
                            print(f"  [-] Failed to generate patch_diff: {e}")
                
                # CRITICAL: patch_diff가 없으면 GT 생성 불가능 (실제 실패)
                if not patch_diff:
                    # Determine why patch_diff is missing (명확한 실패 사유)
                    db_report = get_report_from_db(local_id)
                    failure_reason = "patch_diff is required but not available"
                    if db_report:
                        if not db_report.get('fix_commit'):
                            failure_reason += ": fix_commit not found in database"
                        elif not db_report.get('repo_addr'):
                            failure_reason += ": repo_addr not found in database (cannot clone Git repository)"
                        else:
                            repo_addr = db_report.get('repo_addr', 'unknown')
                            fix_commit = db_report.get('fix_commit', 'unknown')
                            if isinstance(fix_commit, str):
                                fix_commit = fix_commit.split('\n')[0].strip()[:20]
                            failure_reason += f": Git repository clone/checkout failed (repo: {repo_addr}, commit: {fix_commit}...) - possible network error or repository access issue"
                    print(f"[-] Failed: localId {local_id} - {failure_reason}")
                    return {
                        'localId': local_id,
                        'failed': True,
                        'failure_reason': failure_reason,
                        'skipped': True,  # Backward compatibility
                        'reason': failure_reason,  # Backward compatibility
                        'project_name': data.get('osssfuzz_report', {}).get('project_name', '')
                    }
                
                if patch_diff:
                    # Patch diff found - check if Rule 1 can be satisfied
                    vulnerable_deps = data.get('srcmap', {}).get('vulnerable_version', {}).get('dependencies', [])
                    project_name_temp = data.get('osssfuzz_report', {}).get('project_name', '')
                    rule1_score, rule1_dep, rule1_indicates_main = self.rule1_patch_file_path_mapping(patch_diff, vulnerable_deps, project_name=project_name_temp)
                    if rule1_score > 0:
                        # Rule 1 satisfied - patch diff is sufficient for GT
                        # Code snippets optional for LLM inference (can skip if patch_diff is available)
                        print(f"  [*] Patch diff found and Rule 1 satisfied - patch_diff will be used for LLM inference")
                        # Optionally extract code snippets for richer context (but not required)
                        # Uncomment below if you want code snippets even when patch_diff exists:
                        # print(f"  [*] Extracting code snippets for additional context...")
                        # data_with_snippets = extract_data(local_id, include_code_snippets=True, auto_fetch=False)
                        # if data_with_snippets:
                        #     data['code_snippets'] = data_with_snippets.get('code_snippets', {})
                    else:
                        # Patch diff exists but Rule 1 not satisfied - extract stack trace locations only (no code snippets)
                        # Code snippets are slow (Git clone per snippet), so we only extract locations for GT references
                        print(f"  [*] Patch diff found but Rule 1 not satisfied - extracting stack trace locations for Rule 2")
                        # Extract only stack trace locations (fast), not full code snippets (slow)
                        stack_trace = data.get('stack_trace', '')
                        if stack_trace:
                            code_locations = parse_stack_trace(stack_trace)
                            data['stack_trace_locations'] = code_locations
                # Note: patch_diff is now required, so this else block should not be reached
                # But keep it for safety
                if not patch_diff:
                    print(f"  [*] No patch_diff found after all attempts")
            except Exception as e:
                print(f"[-] Error extracting data: {e}")
                # Try with minimal data only
                db_report = get_report_from_db(local_id)
                if not db_report:
                    return None
                # Construct minimal data
                data = {
                    'localId': local_id,
                    'osssfuzz_report': {
                        'project_name': db_report.get('project', ''),
                        'bug_type': db_report.get('crash_type', ''),
                        'severity': db_report.get('severity', ''),
                    },
                    'stack_trace': db_report.get('crash_output', ''),
                    'patch_info': {
                        'fix_commit': db_report.get('fix_commit', ''),
                        'submodule_bug': bool(db_report.get('submodule_bug', False)),
                        'patch_diff': '',
                    },
                    'srcmap': {
                        'vulnerable_version': {
                            'dependencies': []
                        }
                    }
                }
            
            if not data:
                return None
        
        # Initialize
        self.rules_satisfied = []
        self.confidence_score = 0
        
        # Prepare data
        project_name = data.get('osssfuzz_report', {}).get('project_name', '')
        stack_trace = data.get('stack_trace', '')
        patch_info = data.get('patch_info', {})
        fix_commit = patch_info.get('fix_commit', '')
        
        # IMPROVED: If stack_trace is empty, try to get from database
        if not stack_trace:
            db_report = get_report_from_db(local_id)
            if db_report:
                stack_trace = db_report.get('crash_output', '')
                if stack_trace:
                    data['stack_trace'] = stack_trace
                    print(f"  [+] Retrieved stack_trace from database (length: {len(stack_trace)})")
                else:
                    print(f"  [!] Database report found but crash_output is empty")
            else:
                print(f"  [!] Database report not found for localId {local_id}")
        
        # IMPROVED: Extract stack_trace_locations EARLY for patch_crash_distance calculation
        # This MUST be done before patch_crash_distance calculation
        # Calculate stack_trace_key_locations here (before it's needed)
        stack_trace_locations = []
        if stack_trace:
            # Compute from stack_trace
            lines = stack_trace.split('\n')
            for line in lines[:20]:  # Check first 20 lines
                # Pattern: /src/project/file.c:line
                match = re.search(r'(/src/[^\s:]+):(\d+)', line)
                if match:
                    file_path = match.group(1)
                    line_num = int(match.group(2))
                    stack_trace_locations.append({
                        'file_path': file_path,
                        'line': line_num
                    })
                    if len(stack_trace_locations) >= 5:  # Top 5 frames
                        break
            if stack_trace_locations:
                print(f"  [+] Computed {len(stack_trace_locations)} stack_trace_locations from stack_trace")
            else:
                print(f"  [!] Failed to extract stack_trace_locations from stack_trace")
        
        # If still empty, try to get from data (if already computed in previous run)
        if not stack_trace_locations:
            stack_trace_locations = data.get('stack_trace_locations', [])
            if stack_trace_locations:
                print(f"  [+] Retrieved {len(stack_trace_locations)} stack_trace_locations from data")
        if not stack_trace_locations:
            stack_trace_locations = data.get('stack_trace_key_locations', [])
            if stack_trace_locations:
                print(f"  [+] Retrieved {len(stack_trace_locations)} stack_trace_key_locations from data")
        
        # Store in data for later use (both for patch_crash_distance and GT output)
        if stack_trace_locations:
            data['stack_trace_locations'] = stack_trace_locations
            data['stack_trace_key_locations'] = stack_trace_locations  # Also store as key_locations
        else:
            print(f"  [!] No stack_trace_locations available for patch_crash_distance calculation")
        submodule_bug = patch_info.get('submodule_bug', False)
        patch_diff = patch_info.get('patch_diff', '')
        patched_files = patch_info.get('patched_files', [])  # Extract patched_files early
        code_snippets = data.get('code_snippets', {})  # Extract code_snippets from data
        repo_addr = None  # Need to fetch from database
        
        # Get additional information from database
        db_report = get_report_from_db(local_id)
        if db_report:
            repo_addr = db_report.get('repo_addr', '')
            if not fix_commit:
                fix_commit = db_report.get('fix_commit', '')
            if not submodule_bug:
                submodule_bug = bool(db_report.get('submodule_bug', False))
        
        # srcmap dependencies
        srcmap_data = data.get('srcmap', {})
        vulnerable_deps = srcmap_data.get('vulnerable_version', {}).get('dependencies', [])
        
        # Filter: must have dependencies other than the project itself
        # Find main project path
        main_project_path = f"/src/{project_name}"
        other_dependencies = []
        
        for dep in vulnerable_deps:
            dep_path = dep.get('path', '')
            dep_name = dep.get('name', '')
            
            # Only filter dependencies that are not the main project
            if dep_path != main_project_path and dep_name.lower() != project_name.lower():
                other_dependencies.append(dep)
        
        # Case 1: 의존성 2개 미만 -> 메인온리로 분류
        # Cannot build Ground Truth if no dependencies (or only 1 dependency)
        # Exception: submodule_bug cases - the bug is in a submodule, which may not be in srcmap dependencies
        # For submodule bug cases, we can still proceed using Rule 2 (stack trace) even without dependencies in srcmap
        if len(other_dependencies) < 2:
            if submodule_bug:
                print(f"[*] Submodule bug case: No dependencies in srcmap, but will proceed with Rule 2 (stack trace)")
                # For submodule bug, add the submodule as a dependency based on repo_addr
                if repo_addr:
                    # Extract submodule name from repo_addr
                    submodule_name = repo_addr.split('/')[-1].replace('.git', '')
                    other_dependencies = [{
                        'name': submodule_name,
                        'path': f"/src/{submodule_name}",  # Estimated path
                        'url': repo_addr,
                        'type': 'submodule',
                        'commit_sha': fix_commit.split('\n')[0].strip() if fix_commit else ''
                    }]
                    vulnerable_deps.append(other_dependencies[0])
                    print(f"[*] Added submodule '{submodule_name}' as dependency for analysis")
                    # After adding submodule, check again
                    if len(other_dependencies) < 2:
                        # Still less than 2 dependencies -> main_only
                        print(f"[*] Main Only: localId {local_id} - Dependencies < 2 (only {len(other_dependencies)} dependency found)")
                        return {
                            'localId': local_id,
                            'main_only': True,
                            'reason': f'Dependencies < 2 (only {len(other_dependencies)} dependency found)',
                            'project_name': project_name,
                            'dependencies_count': len(other_dependencies),
                            'Heuristically_Root_Cause_Type': 'Main_Project_Specific',
                            'Heuristically_Root_Cause_Dependency': None
                        }
            else:
                print(f"[*] Main Only: localId {local_id} - Dependencies < 2 (only {len(other_dependencies)} dependency found, main project '{project_name}')")
                return {
                    'localId': local_id,
                    'main_only': True,
                    'reason': f'Dependencies < 2 (only {len(other_dependencies)} dependency found)',
                    'project_name': project_name,
                    'dependencies_count': len(other_dependencies),
                    'Heuristically_Root_Cause_Type': 'Main_Project_Specific',
                    'Heuristically_Root_Cause_Dependency': None
                }
        
        print(f"[+] Found {len(other_dependencies)} dependencies (excluding main project)")
        
        # Keep vulnerable_deps as is (include main project for rule application)
        vulnerable_deps = vulnerable_deps  # Maintain all dependencies (including main project)
        
        # Rule 1: Patch file path mapping with new score system (evaluated first)
        # Score 0: Main project only, Score 1: Mixed, Score 2: Single dependency
        rule1_score = 0
        rule1_dep = None
        rule1_indicates_main_project = False  # Track if Rule 1 indicates Main_Project_Specific
        
        # Analyze patch type to detect version update bypasses
        patch_analysis = None
        if patch_diff or patched_files:
            patch_analysis = self.analyze_patch_type(
                patch_diff if patch_diff else '',
                patched_files if patched_files else []
            )
        
        # Try Rule 1 with patch_diff first
        if patch_diff:
            rule1_score, rule1_dep, rule1_indicates_main_project = self.rule1_patch_file_path_mapping(
                patch_diff, vulnerable_deps, project_name=project_name
            )
            
            # Always add Rule 1 to satisfied rules to track its result
            # Score 0 with indicates_main_project=True means all files in main project
            # Score 0 with indicates_main_project=False means no files found (treat as not satisfied)
            self.rules_satisfied.append({
                'rule': 'Rule 1: Patch File Path Exclusivity',
                'dependency': rule1_dep if rule1_score == 2 else None,
                'score': rule1_score,
                'indicates_main_project': rule1_indicates_main_project
            })
        
        # Rule 2: Stack trace dominance (evaluated after Rule 1, but with highest weight)
        rule2_score = 0
        rule2_dep = None
        rule2_coc = None
        
        # IMPROVED: If stack_trace is empty but stack_trace_locations are available,
        # reconstruct a minimal stack_trace string for Rule 2 evaluation
        rule2_stack_trace = stack_trace
        if not rule2_stack_trace and stack_trace_locations:
            # Reconstruct stack_trace from stack_trace_locations
            rule2_stack_trace = '\n'.join([
                f"#0 0x000000000000 in function {loc.get('file_path', '')}:{loc.get('line', 0)}"
                for loc in stack_trace_locations[:10]  # Use top 10 locations
            ])
            print(f"  [+] Reconstructed stack_trace from {len(stack_trace_locations)} locations for Rule 2")
        
        rule2_satisfied, rule2_dep, rule2_coc = self.rule2_stack_trace_dominance(rule2_stack_trace, vulnerable_deps, project_name)
        
        # Frame Attribution integration (Phase 1 Rule 2 COC enhancement)
        # As described in paper Section 5.3.1.C (740-774 lines)
        llm_attribution_confidence = None
        combined_coc = rule2_coc  # Default to heuristic COC
        
        if self.use_frame_attribution and self.llm_modules and rule2_stack_trace:
            try:
                print(f"  [*] Running Frame Attribution (LLM-based) for Rule 2 COC enhancement...")
                
                # Prepare dependencies summary for Frame Attribution
                dependencies_summary_parts = []
                for dep in vulnerable_deps[:10]:  # Top 10 dependencies
                    dep_name = dep.get('name', 'N/A')
                    dep_path = dep.get('path', 'N/A')
                    dep_version = dep.get('version', 'N/A')
                    dependencies_summary_parts.append(f"- {dep_name} (path: {dep_path}, version: {dep_version})")
                
                dependencies_summary = '\n'.join(dependencies_summary_parts) if dependencies_summary_parts else "No dependencies found"
                
                # Call Frame Attribution
                frame_attribution = self.llm_modules._attribute_stack_trace_frames(
                    rule2_stack_trace, 
                    dependencies_summary,
                    logger=None  # Can add logger if needed
                )
                
                if frame_attribution and frame_attribution.logical_owner:
                    llm_attribution_confidence = frame_attribution.confidence
                    logical_owner = frame_attribution.logical_owner
                    
                    print(f"    [+] Frame Attribution: logical_owner='{logical_owner}', confidence={llm_attribution_confidence:.2f}")
                    
                    # Check if LLM attribution matches Rule 2 dependency
                    if rule2_dep and logical_owner.lower() == rule2_dep.lower():
                        # Match: Use LLM confidence
                        llm_confidence = llm_attribution_confidence
                        print(f"    [+] LLM attribution matches Rule 2 dependency '{rule2_dep}'")
                    else:
                        # Mismatch: Set to 0.0 (as per paper)
                        llm_confidence = 0.0
                        if rule2_dep:
                            print(f"    [!] LLM attribution mismatch: LLM='{logical_owner}' vs Rule2='{rule2_dep}', using 0.0")
                        else:
                            print(f"    [!] Rule 2 dependency not found, using 0.0")
                    
                    # Combined COC using weighted average (60% heuristic + 40% LLM)
                    # As per paper Section 5.3.1.C (756-757 lines)
                    if rule2_coc is not None:
                        heuristic_weight = 0.6
                        llm_weight = 0.4
                        combined_coc = heuristic_weight * rule2_coc + llm_weight * llm_confidence
                        print(f"    [+] Combined COC: {rule2_coc:.3f} (heuristic, 60%) + {llm_confidence:.3f} (LLM, 40%) = {combined_coc:.3f}")
                    else:
                        # Fallback: Use LLM confidence only if heuristic COC not available
                        combined_coc = llm_confidence
                        print(f"    [+] Using LLM COC only (heuristic COC not available): {combined_coc:.3f}")
                else:
                    print(f"    [!] Frame Attribution returned no result, using heuristic COC only")
                    
            except Exception as e:
                print(f"    [!] Frame Attribution failed: {e}, falling back to heuristic COC")
                combined_coc = rule2_coc  # Fallback to heuristic COC
        
        # Use combined COC (or heuristic COC if frame attribution not enabled)
        final_coc = combined_coc if combined_coc is not None else rule2_coc
        
        if rule2_satisfied:
            # Calculate continuous score: score = 2 × COC (using combined COC)
            if final_coc is not None:
                rule2_score = 2.0 * final_coc  # Continuous score (0.0~2.0)
            else:
                rule2_score = 2.0  # Fallback if COC not calculated
            self.rules_satisfied.append({
                'rule': 'Rule 2: Stack Trace Dominance',
                'dependency': rule2_dep,
                'score': rule2_score,
                'coc': final_coc,  # Use combined COC
                'heuristic_coc': rule2_coc,  # Store original heuristic COC for reference
                'llm_attribution_confidence': llm_attribution_confidence,  # Store LLM confidence
                'frame_attribution_enabled': self.use_frame_attribution
            })
        
        # stack_trace_locations already extracted above at the beginning of the function
        # Reuse it here (no need to recalculate)
        
        # Compute patch-crash distance for weight amplification
        # IMPROVED: Use stack_trace_locations if stack_trace is not available
        patch_crash_distance = None
        crash_module = None
        patched_module = None
        
        # Debug: Check what we have
        print(f"  [DEBUG] stack_trace length: {len(stack_trace) if stack_trace else 0}")
        print(f"  [DEBUG] patched_files count: {len(patched_files) if patched_files else 0}")
        print(f"  [DEBUG] stack_trace_locations count: {len(stack_trace_locations) if stack_trace_locations else 0}")
        
        if stack_trace and patched_files:
            print(f"  [DEBUG] Using _compute_patch_crash_distance with stack_trace")
            patch_crash_distance, crash_module, patched_module = self._compute_patch_crash_distance(
                stack_trace, patched_files, project_name
            )
            print(f"  [DEBUG] Result: distance={patch_crash_distance}, crash_module={crash_module}, patched_module={patched_module}")
            
            # IMPROVED: If _compute_patch_crash_distance fails, try fallback
            if patch_crash_distance is None and stack_trace_locations:
                print(f"  [DEBUG] _compute_patch_crash_distance failed, trying fallback")
                # Fall through to fallback logic below
        
        # Fallback: Use stack_trace_locations if stack_trace failed or not available
        if patch_crash_distance is None and patched_files and stack_trace_locations:
            # Fallback: Compute distance from stack_trace_locations
            # This happens when stack_trace is not stored in GT but locations are available
            print(f"  [+] Using fallback: computing distance from stack_trace_locations ({len(stack_trace_locations)} locations)")
            if stack_trace_locations:
                # Extract crash location from first stack trace location
                crash_file_path = None
                if stack_trace_locations:
                    crash_file_path = stack_trace_locations[0].get('file_path', '')
                    if crash_file_path:
                        parts = crash_file_path.split('/')
                        if len(parts) >= 3:
                            crash_module = parts[2]  # /src/{module}/...
                
                # Extract patched module from patched files
                patched_module = None
                main_project_path = f"/src/{project_name}" if project_name else None
                
                for patched_file in patched_files:
                    if patched_file.startswith('/src/'):
                        parts = patched_file.split('/')
                        if len(parts) >= 3:
                            patched_module = parts[2]
                            break
                    else:
                        # Relative path - assume main project
                        patched_module = project_name
                        break
                
                # Calculate distance
                if crash_file_path and patched_module:
                    # Check if same file
                    same_file = False
                    for patched_file in patched_files:
                        if crash_file_path == patched_file or crash_file_path.endswith(patched_file) or patched_file.endswith(crash_file_path):
                            patch_crash_distance = 0
                            same_file = True
                            break
                    
                    if not same_file:
                        # Check if same module
                        if crash_module and patched_module and crash_module.lower() == patched_module.lower():
                            patch_crash_distance = 1
                        else:
                            # Different modules
                            if main_project_path:
                                crash_in_main = crash_file_path.startswith(main_project_path)
                                patched_in_main = any(pf.startswith(main_project_path) if pf.startswith('/src/') else True for pf in patched_files)
                                
                                if crash_in_main != patched_in_main:
                                    patch_crash_distance = 2
                                else:
                                    patch_crash_distance = 2
                            else:
                                patch_crash_distance = 2
        
        # Rule 3: Dependency update commit
        rule3_score = 0
        rule3_dep = None
        rule3_satisfied, rule3_dep = self.rule3_dependency_update_commit(submodule_bug, repo_addr, project_name)
        if rule3_satisfied:
            rule3_score = 1
            self.rules_satisfied.append({
                'rule': 'Rule 3: Dependency Update Commit',
                'dependency': rule3_dep,
                'score': rule3_score
            })
        
        # Rule 4: External CVE connection
        rule4_score = 0
        rule4_cve = None
        rule4_satisfied, rule4_cve = self.rule4_external_cve_connection(local_id, project_name, fix_commit)
        if rule4_satisfied:
            rule4_score = 1
            self.rules_satisfied.append({
                'rule': 'Rule 4: External CVE/NVD Connection',
                'cve_id': rule4_cve,
                'score': rule4_score
            })
        
        # Calculate final confidence score using ADDITION with weights
        # CONFIDENCE SCORE = Cumulative Evidence Strength for Dependency_Specific hypothesis
        # 
        # Evidence Types:
        # - Positive Evidence: Supports Dependency_Specific (+)
        # - Negative Evidence: Contradicts Dependency_Specific (-)
        # - Strong Evidence: High weight multiplier (Rule 2 amplification)
        # - Weak Evidence: Low weight (Rule 3)
        #
        # Score Interpretation:
        # - Score > 6.0: Strong evidence for Dependency_Specific
        # - Score 3.0-6.0: Moderate evidence
        # - Score < 3.0: Weak or contradictory evidence
        #
        # Rule weights: R1=1, R2=3 (highest, or amplified), R3=1, R4=1
        # This gives Rule 2 (Stack Trace Dominance) the highest importance
        # IMPORTANT: This is addition with weights, not multiplication
        # Example: R1:0 + R2:2×3 + R3:1 + R4:0 = 0 + 6 + 1 + 0 = 7
        
        # Rule 2 weight amplification: Continuous function instead of discrete threshold
        # This removes threshold artifacts (e.g., COC=0.79 vs 0.81)
        def calculate_rule2_amplification(coc: float, patch_crash_distance: int) -> float:
            """
            Calculate continuous amplification factor for Rule 2
            
            Args:
                coc: Crash Ownership Confidence (0.0-1.0)
                patch_crash_distance: Structural distance (0-3)
            
            Returns:
                Amplification factor (1.0-1.333...)
            """
            if coc is None or patch_crash_distance is None:
                return 1.0
            
            # Normalize distance: 0->0.0, 1->0.3, 2->0.7, 3->1.0
            normalized_dist = min(patch_crash_distance / 3.0, 1.0)
            
            # Amplification: α = 0.333... (tunable)
            # This gives: COC=1.0, dist=2 → amplification ≈ 1.333 (weight 3 → 4)
            #             COC=0.8, dist=2 → amplification ≈ 1.267 (weight 3 → 3.8)
            #             COC=0.6, dist=1 → amplification ≈ 1.06 (weight 3 → 3.18)
            alpha = 1.0 / 3.0  # 0.333...
            amplification = 1.0 + alpha * coc * normalized_dist
            
            # Clamp to reasonable range (max 1.333... to match old threshold behavior)
            return min(amplification, 1.3333333333333333)
        
        base_weight = 3.0
        # Use combined COC (heuristic + LLM) if frame attribution is enabled, otherwise use heuristic COC
        coc_for_amplification = final_coc if final_coc is not None else rule2_coc
        if coc_for_amplification is not None and patch_crash_distance is not None:
            amplification = calculate_rule2_amplification(coc_for_amplification, patch_crash_distance)
            rule2_weight = base_weight * amplification
            rule2_amplified = amplification > 1.0
        else:
            rule2_weight = base_weight
            rule2_amplified = False
        
        rule1_weighted = rule1_score * 1  # Weight: 1
        rule2_weighted = rule2_score * rule2_weight  # Weight: 3 (or amplified)
        rule3_weighted = rule3_score * 1  # Weight: 1
        rule4_weighted = rule4_score * 1  # Weight: 1
        
        self.confidence_score = rule1_weighted + rule2_weighted + rule3_weighted + rule4_weighted
        # Note: Rule 1, 2, 3, 4 are all evaluated above regardless of patch_diff
        # Score calculation is done after all rules are evaluated
        
        # Apply conflict penalty: If Rule 1 indicates Main but Rule 2 indicates Dependency,
        # reduce confidence score to reflect uncertainty
        # This preserves information about rule conflicts for ML/LLM training
        rule_conflict_detected = False
        if rule1_indicates_main_project and rule2_satisfied and rule2_dep:
            # CONFLICT: Rule 1 says Main, Rule 2 says Dependency
            # Apply penalty to reflect uncertainty (15% reduction)
            conflict_penalty = 0.85
            self.confidence_score *= conflict_penalty
            rule_conflict_detected = True
        
        # Determine Root Cause
        root_cause_type = None
        root_cause_dependency = None
        
        # IMPROVED: Rule 2 is treated as "strong but conditional signal"
        # Rule 2 Effective when:
        #   1. Workaround pattern detected (distance >= 2 AND module mismatch), OR
        #   2. High reliability (COC >= 0.75 AND distance >= 2)
        # This aligns with design principle: "Crash location > patch location" (workarounds mislead)
        # But only when Rule 2 signal is reliable
        
        # Calculate workaround_detected (needed for Rule 2 effectiveness check)
        workaround_detected = (
            patch_crash_distance is not None and patch_crash_distance >= 2 and
            crash_module is not None and patched_module is not None and
            crash_module != patched_module
        )
        
        # Determine if Rule 2 is effective (strong and reliable signal)
        # Initialize as False (Rule 2 is not effective by default)
        rule2_effective = False
        if rule2_satisfied and rule2_dep:
            # Rule 2 is effective if:
            # 1. Workaround pattern detected (strong evidence for Dependency)
            # 2. OR high reliability (COC >= 0.75 AND distance >= 2)
            if workaround_detected:
                rule2_effective = True
                print(f"  [+] Rule 2 effective: Workaround pattern detected (distance={patch_crash_distance}, module mismatch)")
            elif coc_for_amplification is not None and patch_crash_distance is not None:
                if coc_for_amplification >= 0.75 and patch_crash_distance >= 2:
                    rule2_effective = True
                    print(f"  [+] Rule 2 effective: High reliability (COC={coc_for_amplification:.2f}, distance={patch_crash_distance})")
                else:
                    print(f"  [!] Rule 2 weak: Low reliability (COC={coc_for_amplification:.2f}, distance={patch_crash_distance})")
                    if rule1_indicates_main_project:
                        print(f"  [!] Rule 1 (Main) takes priority due to weak Rule 2 signal")
        
        # Root Cause determination with improved logic
        if rule2_effective:
            # Rule 2 indicates Dependency_Specific with strong/reliable signal
            root_cause_type = "Dependency_Specific"
            root_cause_dependency = rule2_dep
            
            # Check for conflict with Rule 1
            if rule1_indicates_main_project:
                # CONFLICT: Rule 2 says Dependency (effective), Rule 1 says Main
                # Rule 2 takes priority due to strong signal (workaround or high reliability)
                print(f"  [!] Rule conflict: Rule 2=Dependency ({rule2_dep}, effective), Rule 1=Main")
                coc_str = f"{coc_for_amplification:.2f}" if coc_for_amplification else "None"
                print(f"  [!] Rule 2 takes priority (strong signal: workaround={workaround_detected}, COC={coc_str})")
                if patch_analysis and patch_analysis.get('is_version_update_bypass', False):
                    print(f"  [!] Version update bypass detected - confirming Dependency_Specific")
                # root_cause_type remains Dependency_Specific (Rule 2 wins due to strong signal)
        elif rule1_score == 2 and rule1_dep:
            # Rule 1 indicates Dependency_Specific (single dependency)
            root_cause_type = "Dependency_Specific"
            root_cause_dependency = rule1_dep
        elif rule1_indicates_main_project:
            # Rule 1 indicates Main_Project_Specific
            # Rule 2 is either not satisfied, not effective, or indicates same dependency
            # In this case, prioritize Rule 1 (patch location) when Rule 2 is not effective
            root_cause_type = "Main_Project_Specific"
            root_cause_dependency = None
            if rule2_satisfied and rule2_dep and not rule2_effective:
                print(f"  [+] Rule 1 (Main) takes priority: Rule 2 signal is not effective (weak reliability)")
        else:
            # Extract dependencies from satisfied rules
            dependencies_found = []
            for rule_info in self.rules_satisfied:
                if 'dependency' in rule_info and rule_info['dependency']:
                    dependencies_found.append(rule_info['dependency'])
            
            if dependencies_found:
                # Select most frequently mentioned dependency
                dep_counts = defaultdict(int)
                for dep in dependencies_found:
                    dep_counts[dep] += 1
                
                if dep_counts:
                    most_common_dep = max(dep_counts.items(), key=lambda x: x[1])[0]
                    root_cause_dependency = most_common_dep
                    
                    # Compare with main project (enhanced normalization)
                    # Normalize names for comparison (remove common prefixes/suffixes)
                    project_normalized = project_name.lower().strip()
                    dep_normalized = most_common_dep.lower().strip()
                    
                    # Check exact match
                    if dep_normalized == project_normalized:
                        root_cause_type = "Main_Project_Specific"
                        root_cause_dependency = None
                    # Check if dependency name is just project name with common suffixes
                    elif (dep_normalized.startswith(project_normalized) or 
                          project_normalized.startswith(dep_normalized) or
                          dep_normalized == project_normalized + '.c' or
                          dep_normalized == 'lib' + project_normalized or
                          dep_normalized == project_normalized + 'lib'):
                        # Likely the same project (e.g., "file" -> "file", "lcms" -> "lcms")
                        root_cause_type = "Main_Project_Specific"
                        root_cause_dependency = None
                    else:
                        # Additional check: if dependency path matches main project path
                        dep_info_from_srcmap = None
                        for dep in vulnerable_deps:
                            dep_name = dep.get('name', '').lower().strip()
                            dep_path = dep.get('path', '')
                            if dep_name == dep_normalized or dep_path.endswith(f'/{most_common_dep}'):
                                dep_info_from_srcmap = dep
                                break
                        
                        if dep_info_from_srcmap:
                            dep_path = dep_info_from_srcmap.get('path', '')
                            if dep_path == main_project_path:
                                root_cause_type = "Main_Project_Specific"
                                root_cause_dependency = None
                            else:
                                root_cause_type = "Dependency_Specific"
                        else:
                            root_cause_type = "Dependency_Specific"
            else:
                # Cannot determine if dependencies_found is empty
                root_cause_type = "Unknown"
                root_cause_dependency = None
        
        # Final normalization: set dependency to None if Main_Project_Specific
        if root_cause_type == "Main_Project_Specific":
            root_cause_dependency = None
        
        # Construct Root Cause Dependency information
        # Set dependency info to None for Main_Project_Specific (normalization enhancement)
        root_cause_dep_info = None
        if root_cause_dependency and root_cause_type == "Dependency_Specific":
            # Find dependency information from srcmap
            for dep in vulnerable_deps:
                dep_name = dep.get('name', '')
                dep_path = dep.get('path', '')
                
                # Match by name or path
                if (dep_name.lower() == root_cause_dependency.lower() or 
                    dep_path.endswith(f'/{root_cause_dependency}') or
                    root_cause_dependency in dep_path):
                    root_cause_dep_info = {
                        'name': dep.get('name', root_cause_dependency),
                        'commit_sha': dep.get('commit_sha', ''),
                        'url': dep.get('url', ''),
                        'path': dep.get('path', '')
                    }
                    break
            
            if not root_cause_dep_info:
                # If main project
                if root_cause_dependency.lower() == project_name.lower():
                    # Find main project information
                    for dep in vulnerable_deps:
                        if dep.get('path', '') == main_project_path:
                            root_cause_dep_info = {
                                'name': dep.get('name', project_name),
                                'commit_sha': dep.get('commit_sha', ''),
                                'url': dep.get('url', ''),
                                'path': dep.get('path', '')
                            }
                            break
                
                if not root_cause_dep_info:
                    root_cause_dep_info = {
                        'name': root_cause_dependency,
                        'commit_sha': '',
                        'url': '',
                        'path': ''
                    }
        
        # Extract reference information for LLM inference (file paths and line numbers only)
        patch_file_path = None
        patch_info_data = data.get('patch_info', {})
        if patch_info_data.get('patch_diff'):
            # Try to find patch file path (from extract_data, it may be in patch_info)
            # First check if patch_info has the file path
            if 'patch_file_path' in patch_info_data:
                patch_file_path = patch_info_data['patch_file_path']
            else:
                # Try to find patch file path
                patch_diff_file = get_patch_diff(local_id, auto_generate=False)
                if patch_diff_file and patch_diff_file.exists():
                    patch_file_path = str(patch_diff_file)
                    # If it's a temp file, try to find cached version
                    if '/tmp/' in patch_file_path or 'tmp' in patch_file_path:
                        # Check cache paths
                        cache_paths = [
                            Path(f"/data/patches/{local_id}"),
                            Path(f"/root/Git/ARVO/arvo/patches/{local_id}"),
                        ]
                        for cache_path in cache_paths:
                            if cache_path.exists():
                                diff_files = list(cache_path.glob("*.diff"))
                                if diff_files:
                                    patch_file_path = str(diff_files[0])
                                    break
        
        # stack_trace_locations already extracted above for patch_crash_distance calculation
        # Reuse it here for GT output (no need to recalculate)
        
        # patched_files already extracted above, but update from patch_info_data if available
        if not patched_files:
            patched_files = patch_info_data.get('patched_files', [])
        
        # Try Rule 1 again with patch_file_path and patched_files (if Rule 1 score was 0)
        if rule1_score == 0 and not rule1_indicates_main_project:
            if patch_file_path or patched_files:
                rule1_score_retry, rule1_dep_retry, rule1_indicates_main_project_retry = self.rule1_patch_file_path_mapping(
                    patch_diff, vulnerable_deps,
                    patch_file_path=patch_file_path,
                    patched_files=patched_files,
                    project_name=project_name
                )
                
                # Update Rule 1 results if retry found something
                if rule1_score_retry > rule1_score:
                    rule1_score = rule1_score_retry
                    rule1_dep = rule1_dep_retry
                    rule1_indicates_main_project = rule1_indicates_main_project_retry
                    
                    # Check if Rule 1 is already in rules_satisfied
                    rule1_already_added = any('Rule 1' in str(r.get('rule', '')) for r in self.rules_satisfied)
                    if not rule1_already_added:
                        self.rules_satisfied.append({
                            'rule': 'Rule 1: Patch File Path Exclusivity',
                            'dependency': rule1_dep if rule1_score == 2 else None,
                            'score': rule1_score,
                            'indicates_main_project': rule1_indicates_main_project
                        })
                    else:
                        # Update existing Rule 1 entry
                        for rule_info in self.rules_satisfied:
                            if 'Rule 1' in str(rule_info.get('rule', '')):
                                rule_info['score'] = rule1_score
                                rule_info['dependency'] = rule1_dep if rule1_score == 2 else None
                                rule_info['indicates_main_project'] = rule1_indicates_main_project
                                break
                    
                    # Recalculate confidence score using multiplication
                    # CRITICAL: If Rule 1 score is 0 AND indicates_main_project is True, final score is 0
                    if rule1_score == 0 and rule1_indicates_main_project:
                        self.confidence_score = 0
                        # Skip multiplication calculation
                    else:
                        # Recalculate using multiplication
                        # IMPORTANT: This is multiplication, not addition
                        # Example: R1:0(→1) × R2:2 × R3:1 × R4:0(→1) = 1 × 2 × 1 × 1 = 2
                        rule_scores = [
                            rule1_score if rule1_score > 0 else 1,
                            rule2_score if rule2_score > 0 else 1,
                            rule3_score if rule3_score > 0 else 1,
                            rule4_score if rule4_score > 0 else 1
                        ]
                        self.confidence_score = 1
                        for score in rule_scores:
                            self.confidence_score *= score
                        # Ensure multiplication is used (not addition)
                        assert self.confidence_score == (rule_scores[0] * rule_scores[1] * rule_scores[2] * rule_scores[3]), \
                            f"Score calculation error: expected multiplication, got {self.confidence_score}"
        
        # Re-determine Root Cause if Rule 1 score changed in retry
        # CRITICAL: Only override if rule1_indicates_main_project is False
        # If rule1_indicates_main_project is True, it has highest priority and should not be overridden
        if rule1_score == 2 and rule1_dep and not rule1_indicates_main_project:
            # Extract dependencies from satisfied rules (including newly added Rule 1)
            dependencies_found = []
            for rule_info in self.rules_satisfied:
                if 'dependency' in rule_info and rule_info['dependency']:
                    dependencies_found.append(rule_info['dependency'])
            
            if dependencies_found:
                # Select most frequently mentioned dependency
                dep_counts = defaultdict(int)
                for dep in dependencies_found:
                    dep_counts[dep] += 1
                
                if dep_counts:
                    most_common_dep = max(dep_counts.items(), key=lambda x: x[1])[0]
                    root_cause_dependency = most_common_dep
                    
                    # Compare with main project
                    if root_cause_dependency.lower() == project_name.lower():
                        root_cause_type = "Main_Project_Specific"
                        root_cause_dependency = None
                    else:
                        root_cause_type = "Dependency_Specific"
                    
                    # Update root_cause_dep_info if needed
                    if root_cause_dependency and root_cause_type == "Dependency_Specific":
                        for dep in vulnerable_deps:
                            dep_name = dep.get('name', '')
                            dep_path = dep.get('path', '')
                            
                            if (dep_name.lower() == root_cause_dependency.lower() or 
                                dep_path.endswith(f'/{root_cause_dependency}') or
                                root_cause_dependency in dep_path):
                                root_cause_dep_info = {
                                    'name': dep.get('name', root_cause_dependency),
                                    'commit_sha': dep.get('commit_sha', ''),
                                    'url': dep.get('url', ''),
                                    'path': dep.get('path', '')
                                }
                                break
        
        # Construct Ground Truth
        # Store only line references, not full code snippets (to reduce GT file size)
        # Code snippets will be loaded dynamically for ambiguous cases during LLM inference
        bug_type = data.get('osssfuzz_report', {}).get('bug_type', '') if data else ''
        
        # Case 2: 필드 100% 완성 못한 케이스 체크 (2개 이상 의존성인데 root_cause_type이 Unknown)
        # Check if required fields are incomplete (dependencies >= 2 but root_cause_type is Unknown)
        is_incomplete = False
        failure_reason = None
        if len(other_dependencies) >= 2:
            # Required fields for complete GT:
            # 1. root_cause_type must not be Unknown
            # 2. patch_diff should be available (already checked earlier)
            # 3. stack_trace should be available
            # 4. confidence_score should be calculated
            
            # Check patch_diff availability and reason
            patch_diff_reason = None
            if not patch_diff:
                # Determine why patch_diff is missing
                db_report = get_report_from_db(local_id)
                if db_report:
                    if not db_report.get('fix_commit'):
                        patch_diff_reason = "fix_commit not found in database"
                    elif not db_report.get('repo_addr'):
                        patch_diff_reason = "repo_addr not found in database (cannot clone Git repository)"
                    else:
                        patch_diff_reason = f"Git repository clone/checkout failed (repo: {db_report.get('repo_addr', 'unknown')}, commit: {db_report.get('fix_commit', 'unknown')[:20]}...)"
                else:
                    patch_diff_reason = "Cannot determine reason (database report not available)"
            
            # Check stack_trace availability and reason
            stack_trace_reason = None
            if not stack_trace:
                db_report = get_report_from_db(local_id)
                if db_report:
                    if not db_report.get('crash_output'):
                        stack_trace_reason = "crash_output not found in database"
                    else:
                        stack_trace_reason = "stack_trace extraction failed (crash_output exists but extraction failed)"
                else:
                    stack_trace_reason = "Cannot determine reason (database report not available)"
            
            # Determine failure reason
            # Note: root_cause_type == "Unknown"도 성공으로 간주 (유효한 분류)
            if not patch_diff:
                is_incomplete = True
                failure_reason = f"patch_diff is required but not available: {patch_diff_reason}"
            elif not stack_trace:
                is_incomplete = True
                failure_reason = f"stack_trace is required but not available: {stack_trace_reason}"
            # Note: confidence_score == 0이어도 GT 생성 성공으로 간주 (점수 낮아도 포함)
            # No rules satisfied도 GT 생성 성공으로 간주 (Unknown type으로 분류됨)
        
        ground_truth = {
            'localId': local_id,
            'bug_type': bug_type,  # Include bug_type for filtering by bug_type
            'Heuristically_Root_Cause_Type': root_cause_type,
            'Heuristically_Root_Cause_Dependency': root_cause_dep_info,
            # Case 2: 필드 100% 완성 못한 케이스 표시
            'failed': is_incomplete,
            'failure_reason': failure_reason if is_incomplete else None,
            'Heuristic_Confidence_Score': self.confidence_score,
            'Heuristic_Max_Score': self.max_score,
            'Heuristic_Satisfied_Rules': [r['rule'] for r in self.rules_satisfied],
            'Heuristic_Rule_Details': self.rules_satisfied,
            # Evidence Strength interpretation
            'Evidence_Strength_Interpretation': (
                'Strong' if self.confidence_score > 6.0 else
                'Moderate' if self.confidence_score >= 3.0 else
                'Weak'
            ),
            'rule1_indicates_main_project': rule1_indicates_main_project,  # Track Rule 1's main project indication
            'project_name': project_name,
            'submodule_bug': submodule_bug,
            'dependencies_count': len(other_dependencies),
            'total_dependencies_count': len(vulnerable_deps),
            # Include patch_diff for LLM inference (useful for patch intent analysis)
            'patch_diff': patch_diff if patch_diff else None,  # Full patch diff content
            # Store only line references, not full code snippets (loaded dynamically for ambiguous cases)
            # Reference information for LLM inference (can be loaded dynamically when needed)
            'patch_file_path': patch_file_path,  # Path to patch diff file (if available)
            'patched_files': patched_files[:10] if patched_files else [],  # Top 10 patched files
            'stack_trace': stack_trace if stack_trace else None,  # Store stack_trace for Rule 2 evaluation
            'stack_trace_key_locations': stack_trace_locations,  # Top 5 stack trace locations (for dynamic code snippet loading)
            # IMPROVED: Patch analysis for detecting version update bypasses
            'patch_analysis': patch_analysis if patch_analysis else None,  # Patch type analysis
            # Rule conflict information (for LLM to understand ambiguous cases)
            'rule_conflicts': {
                'rule1_main_vs_rule2_dep': (
                    rule1_indicates_main_project and 
                    rule2_satisfied and rule2_dep is not None
                ),
                'rule2_dependency': rule2_dep if rule2_satisfied else None,
                'rule1_dependency': rule1_dep if rule1_score == 2 else None,
                'conflict_penalty_applied': rule_conflict_detected,  # Whether penalty was applied
                'conflict_penalty_value': 0.85 if rule_conflict_detected else None  # Penalty value if applied
            },
            # Patch-crash distance and module information
            'patch_crash_distance': patch_crash_distance,
            'crash_module': crash_module,
            'patched_module': patched_module,
            # Rule 2 COC (Crash Ownership Confidence)
            # Use combined COC (heuristic + LLM) if frame attribution enabled, otherwise heuristic COC
            'rule2_coc': final_coc,  # Combined COC (or heuristic COC if frame attribution disabled)
            'rule2_heuristic_coc': rule2_coc,  # Original heuristic COC for reference
            'rule2_llm_attribution_confidence': llm_attribution_confidence,  # LLM attribution confidence (if available)
            'frame_attribution_enabled': self.use_frame_attribution,  # Flag indicating if frame attribution was used
            # Workaround detection (based on patch-crash distance, module mismatch)
            # Condition: patch_crash_distance >= 2 AND module mismatch
            # IMPROVED: Now used as Rule 2 effectiveness condition (strengthens Rule 2 signal)
            'workaround_detected': (
                patch_crash_distance is not None and patch_crash_distance >= 2 and
                crash_module is not None and patched_module is not None and
                crash_module != patched_module
            ),
            # Rule 2 effectiveness flag (for transparency)
            # Indicates whether Rule 2 signal was strong/reliable enough to take priority
            'rule2_effective': rule2_effective,
            # Flag to help LLM understand if this requires actual dependency fix
            'requires_dependency_fix': (
                root_cause_type == 'Dependency_Specific' and
                not (patch_analysis and patch_analysis.get('is_version_update_bypass', False))
            ) if patch_analysis else (root_cause_type == 'Dependency_Specific')
        }
        
        return ground_truth


def main():
    parser = argparse.ArgumentParser(description='Build Ground Truth for ARVO cases')
    parser.add_argument('--localId', type=int, help='Single localId to process')
    parser.add_argument('--localIds', type=int, nargs='+', help='Multiple localIds to process')
    parser.add_argument('--project', type=str, help='Process all cases from a project')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of cases to process (default: 10)')
    parser.add_argument('--all', action='store_true', help='Process all cases')
    parser.add_argument('-o', '--output', type=str, default='ground_truth.json', help='Output file')
    parser.add_argument('--min-confidence', type=int, default=2, help='Minimum confidence score to include (default: 2)')
    parser.add_argument('--use-frame-attribution', action='store_true', 
                       help='[DEPRECATED] Enable LLM-based Frame Attribution in Phase 1 (NOT recommended: GT should be pure heuristic baseline). Frame Attribution is used in Phase 2 Stage 0 instead.')
    
    args = parser.parse_args()
    
    # Determine localId list
    if args.localId:
        local_ids = [args.localId]
    elif args.localIds:
        local_ids = args.localIds
    elif args.project:
        # Extract by project
        conn = sqlite3.connect(DB_PATH)
        try:
            if args.all or args.num == 0:
                query = "SELECT localId FROM arvo WHERE reproduced = 1 AND project = ? ORDER BY localId DESC"
                params = (args.project,)
            else:
                query = "SELECT localId FROM arvo WHERE reproduced = 1 AND project = ? ORDER BY localId DESC LIMIT ?"
                params = (args.project, args.num)
            
            cursor = conn.execute(query, params)
            local_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    elif args.all:
        # Extract all
        conn = sqlite3.connect(DB_PATH)
        try:
            query = "SELECT localId FROM arvo WHERE reproduced = 1 ORDER BY localId DESC"
            cursor = conn.execute(query)
            local_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    else:
        print("[-] Please specify --localId, --localIds, --project, or --all")
        return
    
    if not local_ids:
        print("[-] No localIds found")
        return
    
    print(f"[+] Processing {len(local_ids)} localIds...")
    
    # Frame Attribution should NOT be used in Phase 1 (GT must be pure heuristic baseline)
    # Frame Attribution is used in Phase 2 Stage 0 instead
    use_frame_attribution = args.use_frame_attribution  # Default: False (disabled)
    llm_modules = None
    if use_frame_attribution:
        print("[!] WARNING: Frame Attribution in Phase 1 is deprecated.")
        print("[!] GT should be pure heuristic baseline for fair comparison with LLM.")
        print("[!] Frame Attribution is used in Phase 2 Stage 0 instead.")
        try:
            # Import LLM modules
            llm_modules_path = Path(__file__).parent.parent / "llm_inference_modules.py"
            if llm_modules_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("llm_inference_modules", llm_modules_path)
                llm_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(llm_module)
                LLMInferenceModules = llm_module.LLMInferenceModules
                
                # Check for API key
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("[!] Warning: OPENAI_API_KEY not set. Frame attribution will be disabled.")
                    llm_modules = None
                else:
                    llm_modules = LLMInferenceModules(llm_api_key=api_key, llm_model="gpt-4o-mini")
                    print("[+] Frame Attribution enabled: LLM modules initialized")
            else:
                print("[!] Warning: llm_inference_modules.py not found. Frame attribution will be disabled.")
                llm_modules = None
        except Exception as e:
            print(f"[!] Warning: Failed to initialize LLM modules: {e}. Frame attribution will be disabled.")
            llm_modules = None
    
    # Initialize Ground Truth builder
    builder = GroundTruthBuilder(use_frame_attribution=use_frame_attribution, llm_modules=llm_modules)
    
    # Process each localId
    results = []
    failed = []
    # 별도 카운터: main_only와 failed 케이스는 results에 추가하지 않으므로 별도 추적
    main_only_list = []  # main_only 케이스 리스트 (의존성 < 2)
    incomplete_list = []  # incomplete 케이스 리스트 (필드 100% 완성 못함)
    
    # Load existing results if output file exists (for incremental save)
    existing_results = []
    existing_failed = []
    if Path(args.output).exists():
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_results = existing_data.get('ground_truth', [])
                existing_failed = existing_data.get('summary', {}).get('failed_localIds', [])
                print(f"[*] Found existing GT file with {len(existing_results)} entries, will append/update")
        except Exception as e:
            print(f"[!] Warning: Could not load existing GT file: {e}")
    
    # Track already processed localIds to avoid duplicates
    processed_localIds = {gt['localId'] for gt in existing_results}
    processed_localIds.update(existing_failed)
    
    # Start with existing results
    results = existing_results.copy()
    failed = existing_failed.copy()
    
    for idx, local_id in enumerate(local_ids, 1):
        # Skip if already processed
        if local_id in processed_localIds:
            print(f"[*] Skipping localId {local_id} (already processed)")
            # Still save to ensure processed_localIds is updated
            processed_localIds.add(local_id)
            # Save progress even for skipped cases
            try:
                # Count main_only and incomplete cases from separate lists
                main_only_count = len(main_only_list)
                incomplete_count = len(incomplete_list)
                
                output_data = {
                    'summary': {
                        'total_processed': len(set(local_ids) | processed_localIds),
                        'success': len(results),
                        'failed': len(failed),
                        'failed_localIds': [f.get('localId') if isinstance(f, dict) else f for f in failed],  # Backward compatibility
                        'failed_details': failed,  # Detailed failure information
                        'min_confidence': args.min_confidence,
                        # Case 1: Main Only 케이스 카운트 (의존성 2개 미만)
                        'main_only_count': main_only_count,
                        'main_only_localIds': [gt.get('localId') for gt in main_only_list],
                        # Case 2: Failed 케이스 카운트 (필드 100% 완성 못한 케이스)
                        'incomplete_count': incomplete_count,
                        'incomplete_localIds': [gt.get('localId') for gt in incomplete_list]
                    },
                    'ground_truth': results
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[!] Warning: Could not save incremental results: {e}")
            continue
        
        should_save = False
        
        try:
            print(f"\n[*] Processing localId {local_id} ({idx}/{len(local_ids)})...")
            gt = builder.build_ground_truth(local_id)
            
            if gt:
                # GT 생성 성공 케이스: 모든 케이스를 results에 포함
                # (의존성 0개, 점수 낮음, main_only 모두 포함)
                
                # 실제 실패 케이스만 failed로 분류
                if gt.get('failed', False):
                    # 필수 필드를 완성할 수 없는 경우 (patch_diff 없음, stack_trace 없음 등)
                    failure_reason = gt.get('failure_reason', 'Unknown failure reason')
                    print(f"[-] Failed: localId {local_id} - {failure_reason}")
                    failed_detail = {
                        'localId': local_id,
                        'reason': failure_reason,
                        'type': 'incomplete_fields'
                    }
                    if local_id not in [f.get('localId') if isinstance(f, dict) else f for f in failed]:
                        failed.append(failed_detail)
                    processed_localIds.add(local_id)
                    should_save = True
                elif gt.get('skipped', False):
                    # skipped는 실제로 GT 생성 실패 (patch_diff 생성 실패 등)
                    skip_reason = gt.get('reason', 'Unknown reason')
                    print(f"[-] Failed: localId {local_id} - {skip_reason}")
                    failed_detail = {
                        'localId': local_id,
                        'reason': skip_reason,
                        'type': 'gt_generation_failed'
                    }
                    if local_id not in [f.get('localId') if isinstance(f, dict) else f for f in failed]:
                        failed.append(failed_detail)
                    processed_localIds.add(local_id)
                    should_save = True
                else:
                    # GT 생성 성공: 모든 케이스를 results에 포함
                    # (의존성 0개든, 점수 낮든, main_only든 모두 포함)
                    results = [r for r in results if r.get('localId') != local_id]
                    results.append(gt)
                    processed_localIds.add(local_id)
                    
                    score = gt.get('Heuristic_Confidence_Score', 0)
                    max_score = gt.get('Heuristic_Max_Score', 0)
                    root_type = gt.get('Heuristically_Root_Cause_Type', 'Unknown')
                    deps_count = gt.get('dependencies_count', 0)
                    
                    # 통계를 위해 main_only 리스트에도 추가 (통계용)
                    if gt.get('main_only', False):
                        if local_id not in [m.get('localId') for m in main_only_list]:
                            main_only_list.append(gt)
                    
                    # 정보 출력
                    if gt.get('main_only', False):
                        print(f"[+] Success (main only): localId {local_id}, Score: {score}/{max_score}, Type: {root_type}, Dependencies: {deps_count}")
                    elif score < args.min_confidence:
                        print(f"[+] Success (low score): localId {local_id}, Score: {score}/{max_score}, Type: {root_type}, Dependencies: {deps_count}")
                    else:
                        print(f"[+] Success: localId {local_id}, Score: {score}/{max_score}, Type: {root_type}, Dependencies: {deps_count}")
                    should_save = True
            else:
                # build_ground_truth가 None 반환: 데이터 추출 실패
                failed_detail = {
                    'localId': local_id,
                    'reason': 'build_ground_truth returned None - data extraction failed (possibly database access error or missing report)',
                    'type': 'data_extraction_failed'
                }
                if local_id not in [f.get('localId') if isinstance(f, dict) else f for f in failed]:
                    failed.append(failed_detail)
                processed_localIds.add(local_id)
                print(f"[-] Failed: localId {local_id} - build_ground_truth returned None (data extraction failed)")
                should_save = True
        except Exception as e:
            # Exception 발생: 네트워크 오류, 파일 시스템 오류 등
            exception_type = type(e).__name__
            exception_msg = str(e)
            
            # 더 명확한 실패 사유 분류
            failure_type = 'exception'
            if 'git' in exception_msg.lower() or 'clone' in exception_msg.lower() or 'checkout' in exception_msg.lower():
                failure_type = 'git_operation_failed'
                reason = f"Git operation failed: {exception_msg}"
            elif 'network' in exception_msg.lower() or 'connection' in exception_msg.lower() or 'timeout' in exception_msg.lower():
                failure_type = 'network_error'
                reason = f"Network error: {exception_msg}"
            elif 'file' in exception_msg.lower() or 'permission' in exception_msg.lower() or 'not found' in exception_msg.lower():
                failure_type = 'file_system_error'
                reason = f"File system error: {exception_msg}"
            else:
                reason = f"Exception ({exception_type}): {exception_msg}"
            
            failed_detail = {
                'localId': local_id,
                'reason': reason,
                'type': failure_type,
                'exception_type': exception_type
            }
            if local_id not in [f.get('localId') if isinstance(f, dict) else f for f in failed]:
                failed.append(failed_detail)
            processed_localIds.add(local_id)
            print(f"[-] Failed: localId {local_id} - {reason}")
            import traceback
            traceback.print_exc()
            should_save = True
        
        # Incremental save after each processing (ALWAYS save, even if should_save is False)
        # This ensures progress is saved even if there's an unexpected state
        try:
            # Count main_only and incomplete cases from separate lists
            main_only_count = len(main_only_list)
            incomplete_count = len(incomplete_list)
            
            # Extract localIds from failed list (for backward compatibility)
            failed_localIds = [f.get('localId') if isinstance(f, dict) else f for f in failed]
            
            output_data = {
                'summary': {
                    'total_processed': len(set(local_ids) | processed_localIds),
                    'success': len(results),
                    'failed': len(failed),
                    'failed_localIds': failed_localIds,  # Backward compatibility
                    'failed_details': failed,  # Detailed failure information
                    'min_confidence': args.min_confidence,
                    # Case 1: Main Only 케이스 카운트 (의존성 2개 미만)
                    'main_only_count': main_only_count,
                    # Case 2: Failed 케이스 카운트 (필드 100% 완성 못한 케이스)
                    'incomplete_count': incomplete_count
                },
                'ground_truth': results
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            if should_save or idx % 10 == 0:  # Print every 10th or when should_save is True
                print(f"[*] Saved progress: {len(results)} successful, {len(failed)} failed (processed {idx}/{len(local_ids)})")
        except Exception as e:
            print(f"[!] ERROR: Could not save incremental results: {e}")
            import traceback
            traceback.print_exc()
            # Don't continue if save fails - this is critical
            print(f"[!] FATAL: Save failed, stopping to prevent data loss")
            raise
    
    # Final save (ensure latest state is saved)
    # Count main_only and incomplete cases from separate lists
    main_only_count = len(main_only_list)
    incomplete_count = len(incomplete_list)
    
    # Extract localIds from failed list (for backward compatibility)
    failed_localIds = [f.get('localId') if isinstance(f, dict) else f for f in failed]
    
    output_data = {
        'summary': {
            'total_processed': len(set(local_ids) | processed_localIds),
            'success': len(results),
            'failed': len(failed),
            'failed_localIds': failed_localIds,  # Backward compatibility
            'failed_details': failed,  # Detailed failure information
            'min_confidence': args.min_confidence,
            # Case 1: Main Only 케이스 카운트 (의존성 2개 미만)
            'main_only_count': main_only_count,
            'main_only_localIds': [gt.get('localId') for gt in main_only_list],
            # Case 2: Failed 케이스 카운트 (필드 100% 완성 못한 케이스)
            'incomplete_count': incomplete_count,
            'incomplete_localIds': [gt.get('localId') for gt in incomplete_list]
        },
        'ground_truth': results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] Ground Truth building complete!")
    print(f"[+] Success: {len(results)}")
    print(f"[+] Failed: {len(failed)}")
    print(f"[+] Main Only (의존성 < 2): {main_only_count}")
    print(f"[+] Incomplete (필드 100% 완성 못함): {incomplete_count}")
    print(f"[+] Output saved to: {args.output}")
    
    # Print statistics
    if results:
        score_distribution = defaultdict(int)
        type_distribution = defaultdict(int)
        for gt in results:
            score = gt.get('Heuristic_Confidence_Score', 0)
            score_distribution[score] = score_distribution.get(score, 0) + 1
            root_cause_type = gt.get('Heuristically_Root_Cause_Type', 'Unknown')
            type_distribution[root_cause_type] = type_distribution.get(root_cause_type, 0) + 1
        
        print(f"\n[+] Score distribution:")
        for score in sorted(score_distribution.keys()):
            print(f"    Score {score}: {score_distribution[score]} cases")
        
        print(f"\n[+] Type distribution:")
        for type_name in sorted(type_distribution.keys()):
            print(f"    {type_name}: {type_distribution[type_name]} cases")
    
    # Print paper metrics summary
    print_paper_metrics_summary(output_data['summary'], len(results), len(failed), main_only_count)


def print_paper_metrics_summary(summary: Dict, success_count: int, failed_count: int, main_only_count: int):
    """Print paper metrics summary in a readable format"""
    total_processed = summary.get('total_processed', 0)
    success = summary.get('success', success_count)
    failed = summary.get('failed', failed_count)
    main_only = summary.get('main_only_count', main_only_count)
    
    if total_processed == 0:
        return
    
    print("\n" + "=" * 80)
    print("📊 PAPER METRICS SUMMARY (Phase 1 - Heuristic Ground Truth)")
    print("=" * 80)
    
    # Phase 1 Statistics
    success_rate = (success / total_processed * 100) if total_processed > 0 else 0
    failure_rate = (failed / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  • Total vulnerabilities processed: {total_processed:,}")
    print(f"  • Phase 1 Success: {success:,} ({success_rate:.1f}%)")
    print(f"  • Phase 1 Failed: {failed:,} ({failure_rate:.1f}%)")
    print(f"  • Main Only (dependencies < 2): {main_only:,}")
    
    print(f"\n📝 Paper Values:")
    print(f"  • **{total_processed:,}** - ARVO database total vulnerabilities")
    print(f"  • **{success:,}** - Phase 1 successfully attributed cases")
    print(f"  • **{success_rate:.1f}%** - Phase 1 success rate (baseline)")
    print(f"  • **{failed:,}** - Phase 1 failed cases (Phase 2 candidates)")
    print(f"  • **{failure_rate:.1f}%** - Phase 1 failure rate")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

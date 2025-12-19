#!/usr/bin/env python3
"""
Data Extraction Script for ARVO Database

Extracts vulnerability information from ARVO database:
1. OSS-Fuzz report: vulnerability description, bug type, project_name
2. Stack trace: full stack trace at crash time
3. ARVO patch info: patch diff from fixed_commit, patched file paths
4. srcmap.json: list of all dependency libraries used in the build
5. Code snippets: code snippets from main project and dependency libraries

Usage:
    python3 01_extract_from_db.py --localId 40096184
    python3 01_extract_from_db.py --localIds 40096184 40096185
    python3 01_extract_from_db.py --project skia -n 5
    python3 01_extract_from_db.py --localId 40096184 --code-snippets
"""

import sys
import json
import sqlite3
import re
import subprocess
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import os

DB_PATH = os.environ.get("ARVO_DB_PATH") or str((Path(__file__).resolve().parents[1] / "arvo.db"))


def get_report_from_db(local_id: int) -> Optional[Dict]:
    """Retrieve report information directly from database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT * FROM arvo WHERE localId = ?", (local_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_srcmap_files(local_id: int, auto_download: bool = False) -> List[Path]:
    """Find srcmap files (with optional auto-download)
    
    Args:
        local_id: Local ID to find srcmap for
        auto_download: Whether to auto-download if not found
    
    Returns:
        List of srcmap file paths
    """
    possible_paths = [
        Path(f"/data/oss-out/{local_id}"),
        Path(f"/data/oss-work/{local_id}"),
        Path(f"/data/issues/{local_id}"),
    ]
    
    srcmaps = []
    for base_path in possible_paths:
        if base_path.exists():
            for srcmap_file in base_path.rglob("*.srcmap.json"):
                srcmaps.append(srcmap_file)
            if srcmaps:
                break
    
    # Check ARVO issues directory
    if not srcmaps:
        arvo_issues = Path("/data/issues")
        if arvo_issues.exists():
            issue_dir = arvo_issues / str(local_id)
            if issue_dir.exists():
                for srcmap_file in issue_dir.glob("*.srcmap.json"):
                    srcmaps.append(srcmap_file)
    
    # Check ARVO NewTracker directory
    if not srcmaps:
        newtracker_issues = Path("/root/Git/ARVO/arvo/NewTracker/Issues")
        if newtracker_issues.exists():
            issue_dir = newtracker_issues / f"{local_id}_files"
            if issue_dir.exists():
                for srcmap_file in issue_dir.glob("*.srcmap.json"):
                    srcmaps.append(srcmap_file)
    
    # Auto-download if enabled and not found
    if not srcmaps and auto_download:
        print(f"  [*] Attempting to download srcmap files...")
        try:
            from arvo.utils_meta import download_build_artifacts, getMeta
            from arvo.utils import getIssue
            
            issue = getIssue(local_id)
            if issue:
                metadata = None
                try:
                    from arvo.utils_init import MetaDataFile
                    
                    if Path(MetaDataFile).exists():
                        with open(MetaDataFile, 'r') as f:
                            for line in f:
                                try:
                                    meta_line = json.loads(line.strip())
                                    if meta_line.get('localId') == local_id:
                                        metadata = meta_line
                                        break
                                except:
                                    continue
                except Exception as e:
                    print(f"  [-] Failed to read metadata: {e}")
                
                if metadata:
                    download_dir = Path(f"/data/issues/{local_id}_files")
                    download_dir.mkdir(parents=True, exist_ok=True)
                    
                    if 'build_artifacts_url' in metadata and metadata['build_artifacts_url']:
                        try:
                            downloaded = download_build_artifacts(
                                metadata, metadata['build_artifacts_url'], download_dir
                            )
                            if downloaded:
                                for srcmap_file in download_dir.glob("*.srcmap.json"):
                                    srcmaps.append(srcmap_file)
                                print(f"  [+] Successfully downloaded {len(srcmaps)} srcmap file(s)")
                        except Exception as e:
                            print(f"  [-] Error downloading srcmap: {e}")
                else:
                    print(f"  [-] Cannot download srcmap: metadata not found")
        except Exception as e:
            print(f"  [-] Failed to download srcmap: {e}")
    
    return sorted(srcmaps) if srcmaps else []


def detect_repo_type(repo_url: str) -> Optional[str]:
    """Auto-detect repository type from URL"""
    if not repo_url:
        return None
    
    repo_url_lower = repo_url.lower()
    
    if 'hg.' in repo_url_lower or 'hg.nginx.org' in repo_url_lower:
        return 'hg'
    
    if 'svn.' in repo_url_lower or 'svn.code.sf.net' in repo_url_lower:
        return 'svn'
    
    if ('github.com' in repo_url_lower or 'gitlab.com' in repo_url_lower or
        'bitbucket.org' in repo_url_lower or '.git' in repo_url_lower or
        'googlesource.com' in repo_url_lower):
        return 'git'
    
    return 'git'  # Default


def get_patch_diff(local_id: int, auto_generate: bool = False, report_data: Optional[Dict] = None) -> Optional[Path]:
    """Find patch diff file (with optional auto-generation)
    
    Args:
        local_id: Local ID to find patch for
        auto_generate: Whether to auto-generate from Git if not found
        report_data: Report data dict (if None, fetched from DB)
    
    Returns:
        Path to diff file or None
    """
    possible_paths = [
        Path(f"/data/patches/{local_id}"),
        Path(f"/root/Git/ARVO/arvo/patches/{local_id}"),
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            diff_files = sorted(base_path.glob("*.diff"))
            if diff_files:
                return diff_files[0]
    
    if auto_generate:
        print(f"  [*] Attempting to generate patch diff from Git...")
        try:
            if not report_data:
                report_data = get_report_from_db(local_id)
            
            if not report_data or not isinstance(report_data, dict):
                print(f"  [-] Cannot generate patch diff: report data not found")
                return None
            
            if not report_data.get('fix_commit'):
                print(f"  [-] Cannot generate patch diff: fix_commit not found")
                return None
            
            # Note: submodule_bug cases can still have patch_diff
            # - Submodule bugs may be fixed by updating submodule version in main project
            # - Or the fix_commit may point to the submodule repository
            # - We should attempt to generate patch_diff regardless of submodule_bug flag
            
            try:
                from arvo.utils_git import GitTool
                from arvo.utils_init import PATCHES
                import shutil
            except ModuleNotFoundError as e:
                error_msg = str(e)
                missing_module = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                if missing_module in ['sqlalchemy', 'requests', 'tiktoken']:
                    print(f"  [-] Cannot generate patch diff: {missing_module} module not installed.")
                    print(f"      Please activate virtual environment: source /root/Git/ARVO/arvo-run/bin/activate")
                    print(f"      Or install manually: pip install {missing_module}")
                    return None
                else:
                    raise
            
            fix_commit = report_data['fix_commit']
            if isinstance(fix_commit, str):
                fix_commits = [c.strip() for c in fix_commit.split('\n') if c.strip()]
            else:
                fix_commits = [str(fix_commit)]
            
            repo_addr = report_data.get('repo_addr', '')
            if not repo_addr:
                project = report_data.get('project', '')
                if project:
                    repo_addr = f"https://github.com/{project}/{project}.git"
                    if project in ['skia']:
                        repo_addr = f"https://github.com/google/{project}.git"
            
            if not repo_addr:
                print(f"  [-] Cannot generate patch diff: repo_addr not found")
                return None
            
            cache_dir = PATCHES / str(local_id)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cached_diffs = sorted(cache_dir.glob("*.diff"))
            if cached_diffs and len(cached_diffs) == len(fix_commits):
                tmp_dir = Path("/tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                combined_diff = tmp_dir / f"arvo_diff_{local_id}_{hash(str(fix_commits))}.diff"
                with open(combined_diff, 'wb') as output:
                    for diff_file in cached_diffs:
                        with open(diff_file, 'rb') as f:
                            output.write(f.read())
                print(f"  [+] Generated patch diff file (from cache): {combined_diff}")
                return combined_diff
            
            repo_type = detect_repo_type(repo_addr)
            if not repo_type:
                print(f"  [-] Failed to detect repository type: {repo_addr}")
                return None
            
            gt = None
            github_mirror_used = False
            
            try:
                if repo_type == 'hg':
                    try:
                        gt = GitTool(repo_addr, repo_type)
                        if gt and gt.repo:
                            print(f"  [+] Successfully cloned Mercurial repository")
                    except:
                        pass
                
                if (not gt or not gt.repo) and repo_type == 'hg' and 'hg.nginx.org' in repo_addr:
                    github_mirror = repo_addr.replace('http://hg.nginx.org/', 'https://github.com/nginx/')
                    if github_mirror != repo_addr:
                        print(f"  [*] Mercurial failed, trying GitHub mirror: {github_mirror}")
                        try:
                            gt = GitTool(github_mirror, 'git')
                            if gt and gt.repo:
                                print(f"  [+] Successfully cloned from GitHub mirror")
                                github_mirror_used = True
                                repo_type = 'git'
                        except:
                            pass
                
                if not gt or not gt.repo:
                    gt = GitTool(repo_addr, repo_type)
                
                if not gt or not gt.repo:
                    print(f"  [-] Failed to clone {repo_type} repository: {repo_addr}")
                    if repo_type == 'hg':
                        print(f"  [*] Note: Mercurial clone failed (network error or server issue)")
                        print(f"  [*] Continuing without patch diff")
                    return None
            except SystemExit:
                print(f"  [-] Failed to clone {repo_type} repository: {repo_addr}")
                if repo_type == 'hg':
                    print(f"  [*] Note: Mercurial clone failed (network error or server issue)")
                    print(f"  [*] Continuing without patch diff")
                return None
            except Exception as e:
                print(f"  [-] Error cloning repository: {e}")
                if repo_type == 'hg':
                    print(f"  [*] Note: Mercurial clone failed (network error or server issue)")
                    print(f"  [*] Continuing without patch diff")
                return None
            
            success_count = 0
            for commit in fix_commits:
                commit = commit.strip()
                if not commit:
                    continue
                
                try:
                    diff_file = gt.showCommit(commit)
                    if diff_file and Path(diff_file).exists():
                        shutil.move(diff_file, cache_dir / f"{commit}.diff")
                        success_count += 1
                except Exception as e:
                    print(f"  [-] Failed to generate diff for commit {commit}: {e}")
                    continue
            
            if success_count == 0:
                if gt and hasattr(gt, 'repo') and gt.repo and gt.repo.exists():
                    try:
                        shutil.rmtree(gt.repo.parent, ignore_errors=True)
                    except:
                        pass
                print(f"  [-] Failed to generate diff for all commits")
                print(f"  [*] Note: Mercurial commit hashes may differ from Git")
                print(f"  [*] Continuing without patch diff")
                return None
            
            tmp_dir = Path("/tmp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            combined_diff = tmp_dir / f"arvo_diff_{local_id}_{hash(str(fix_commits))}.diff"
            with open(combined_diff, 'wb') as output:
                for diff_file in sorted(cache_dir.glob("*.diff")):
                    with open(diff_file, 'rb') as f:
                        output.write(f.read())
            
            if gt and hasattr(gt, 'repo') and gt.repo and gt.repo.exists():
                try:
                    shutil.rmtree(gt.repo.parent, ignore_errors=True)
                except:
                    pass
            print(f"  [+] Successfully generated patch diff file: {combined_diff}")
            return combined_diff
            
        except Exception as e:
            print(f"  [-] Failed to generate patch diff: {e}")
            import traceback
            traceback.print_exc()
    
    return None


def extract_dependencies_from_srcmap(srcmap_path: Path) -> List[Dict]:
    """Extract dependencies from srcmap file"""
    if not srcmap_path or not srcmap_path.exists():
        return []
    
    dependencies = []
    try:
        with open(srcmap_path, 'r') as f:
            srcmap_data = json.load(f)
        
        for path, info in srcmap_data.items():
            if isinstance(info, dict) and 'url' in info:
                dep_info = {
                    'path': path,
                    'name': Path(path).name if path != '/' else 'root',
                    'url': info.get('url', ''),
                    'type': info.get('type', 'unknown'),
                    'commit_sha': info.get('rev', '')
                }
                dependencies.append(dep_info)
    except Exception as e:
        print(f"  [-] Error reading srcmap {srcmap_path}: {e}")
    
    return dependencies


def extract_patched_files(diff_file: Path) -> List[str]:
    """Extract patched file paths from diff file"""
    if not diff_file or not diff_file.exists():
        return []
    
    patched_files = []
    try:
        with open(diff_file, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            for line in lines:
                if line.startswith('diff --git'):
                    parts = line.split()
                    if len(parts) >= 4:
                        file_path = parts[2].replace('a/', '', 1)
                        if file_path not in patched_files:
                            patched_files.append(file_path)
    except:
        pass
    
    return patched_files


def parse_stack_trace(stack_trace: str) -> List[Dict]:
    """Extract file paths and line numbers from stack trace"""
    code_locations = []
    patterns = [
        r'(/src/[^\s:]+):(\d+)(?::(\d+))?',
        r'\.\./\.\./src/([^\s:]+):(\d+)(?::(\d+))?',
        r'\.\./\.\./\.\./src/([^\s:]+):(\d+)(?::(\d+))?',
    ]
    
    seen_locations = set()
    
    for line in stack_trace.split('\n'):
        for pattern_idx, pattern in enumerate(patterns):
            matches = re.finditer(pattern, line)
            for match in matches:
                if pattern_idx == 0:
                    file_path = match.group(1)
                    line_num = int(match.group(2))
                    col_num = int(match.group(3)) if match.group(3) else None
                    
                    parts = file_path.split('/')
                    if len(parts) >= 3 and parts[1] == 'src':
                        project_name = parts[2]
                        relative_path = '/'.join(parts[3:])
                    else:
                        continue
                else:
                    relative_path = match.group(1)
                    line_num = int(match.group(2))
                    col_num = int(match.group(3)) if match.group(3) else None
                    project_name = None
                    file_path = f"/src/{relative_path}"
                
                location_key = (file_path, line_num)
                if location_key in seen_locations:
                    continue
                seen_locations.add(location_key)
                
                code_locations.append({
                    'file_path': file_path,
                    'relative_path': relative_path if 'relative_path' in locals() else file_path.replace('/src/', ''),
                    'project': project_name,
                    'line': line_num,
                    'column': col_num,
                    'context': line.strip()
                })
    
    return code_locations


def get_code_snippet_from_local(local_path: Path, line_num: int, context_lines: int = 10, include_code: bool = False) -> Optional[Dict]:
    """Get code snippet from local file
    
    Args:
        local_path: File path
        line_num: Target line number
        context_lines: Number of context lines
        include_code: Whether to include code content (default: False, only line numbers)
    """
    try:
        file_path = Path(local_path)
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        start_line = max(0, line_num - context_lines - 1)
        end_line = min(len(lines), line_num + context_lines)
        
        result = {
            'file_path': str(file_path),
            'start_line': start_line + 1,
            'end_line': end_line,
            'target_line': line_num,
        }
        
        if include_code:
            snippet_lines = lines[start_line:end_line]
            result['code'] = ''.join(snippet_lines)
            result['lines'] = [{'line_num': start_line + 1 + i, 'content': line.rstrip()} 
                             for i, line in enumerate(snippet_lines)]
        
        return result
    except Exception:
        return None


def get_code_snippet_from_git(repo_url: str, commit_sha: str, file_path: str, line_num: int, context_lines: int = 10) -> Optional[Dict]:
    """Get code snippet from Git repository"""
    try:
        import tempfile
        import shutil
        import os
        
        normalized_path = os.path.normpath(file_path)
        
        temp_dir = Path(tempfile.mkdtemp())
        repo_dir = temp_dir / "repo"
        
        try:
            if commit_sha:
                clone_result = subprocess.run(
                    ['git', 'clone', '--depth', '100', repo_url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                    text=True
                )
                checkout_result = subprocess.run(
                    ['git', 'checkout', commit_sha],
                    cwd=repo_dir,
                    capture_output=True,
                    timeout=30,
                    text=True
                )
                if checkout_result.returncode != 0:
                    fetch_result = subprocess.run(
                        ['git', 'fetch', '--depth', '1000', 'origin', commit_sha],
                        cwd=repo_dir,
                        capture_output=True,
                        timeout=60,
                        text=True
                    )
                    checkout_result = subprocess.run(
                        ['git', 'checkout', commit_sha],
                        cwd=repo_dir,
                        check=True,
                        capture_output=True,
                        timeout=30,
                        text=True
                    )
            else:
                clone_result = subprocess.run(
                    ['git', 'clone', '--depth', '1', repo_url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                    timeout=60,
                    text=True
                )
            
            relative_path = normalized_path
            if normalized_path.startswith('/src/'):
                parts = normalized_path.split('/')
                if len(parts) >= 3:
                    relative_path = '/'.join(parts[3:])
            
            relative_path = relative_path.lstrip('/')
            relative_path = os.path.normpath(relative_path)
            
            file_full_path = repo_dir / relative_path
            
            if not file_full_path.exists():
                if 'out/Fuzz' in normalized_path:
                    simplified = normalized_path.replace('/out/Fuzz/../../', '/')
                    simplified = os.path.normpath(simplified)
                    if simplified.startswith('/src/'):
                        parts = simplified.split('/')
                        if len(parts) >= 3:
                            simplified = '/'.join(parts[3:])
                    simplified = simplified.lstrip('/')
                    file_full_path = repo_dir / simplified
            
            if not file_full_path.exists():
                return None
            
            snippet = get_code_snippet_from_local(file_full_path, line_num, context_lines, include_code=False)
            if snippet:
                snippet['repository_url'] = repo_url
                snippet['commit_sha'] = commit_sha or 'N/A'
                if '/tmp/' in snippet['file_path'] or 'repo' in snippet['file_path']:
                    snippet['file_path'] = relative_path
            return snippet
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None


def extract_code_snippets_from_stack_trace(stack_trace: str, project_name: str, srcmap_data: Optional[Dict] = None, context_lines: int = 15) -> Dict:
    """Extract code snippets from stack trace"""
    code_locations = parse_stack_trace(stack_trace)
    
    if not code_locations:
        return {
            'stack_trace_locations': [],
            'code_snippets': {
                'main_project': {},
                'dependencies': {}
            }
        }
    
    if not project_name and code_locations:
        project_counts = {}
        for loc in code_locations:
            if loc.get('project'):
                project_counts[loc['project']] = project_counts.get(loc['project'], 0) + 1
        if project_counts:
            project_name = max(project_counts.items(), key=lambda x: x[1])[0]
    
    normalized_srcmap = {}
    if srcmap_data:
        if isinstance(srcmap_data, dict):
            if '/src/' in str(srcmap_data.keys()):
                normalized_srcmap = srcmap_data
            elif 'vulnerable_version' in srcmap_data:
                vul_deps = srcmap_data.get('vulnerable_version', {}).get('dependencies', [])
                if isinstance(vul_deps, list):
                    for dep in vul_deps:
                        if isinstance(dep, dict) and 'path' in dep:
                            normalized_srcmap[dep['path']] = {
                                'url': dep.get('url', ''),
                                'rev': dep.get('commit_sha', ''),
                                'type': dep.get('type', '')
                            }
    
    main_project_url = ''
    main_project_commit = ''
    
    if normalized_srcmap:
        if project_name:
            project_path = f"/src/{project_name}"
            if project_path in normalized_srcmap:
                main_project_info = normalized_srcmap[project_path]
                main_project_url = main_project_info.get('url', '')
                main_project_commit = main_project_info.get('rev', '')
        
        if not main_project_url and normalized_srcmap:
            for key in normalized_srcmap.keys():
                if key.startswith('/src/') and key != '/src/llvm-project':
                    parts = key.split('/')
                    if len(parts) >= 3:
                        inferred_project = parts[2]
                        main_project_info = normalized_srcmap[key]
                        main_project_url = main_project_info.get('url', '')
                        main_project_commit = main_project_info.get('rev', '')
                        if not project_name:
                            project_name = inferred_project
                        break
    
    if not main_project_url and project_name:
        common_repos = {
            'skia': 'https://github.com/google/skia.git',
            'arrow': 'https://github.com/apache/arrow.git',
            'ffmpeg': 'https://github.com/FFmpeg/FFmpeg.git',
            'harfbuzz': 'https://github.com/harfbuzz/harfbuzz.git',
        }
        if project_name.lower() in common_repos:
            main_project_url = common_repos[project_name.lower()]
    
    main_project_snippets = []
    if project_name and main_project_url:
        for loc in code_locations:
            if not loc.get('project') or loc['project'] == project_name:
                snippet = get_code_snippet_from_git(
                    main_project_url,
                    main_project_commit,
                    loc['file_path'],
                    loc['line'],
                    context_lines
                )
                
                if snippet:
                    snippet['location'] = loc
                    main_project_snippets.append(snippet)
    
    dependency_snippets = {}
    dependency_projects = set()
    for loc in code_locations:
        if loc.get('project') and loc['project'] != project_name:
            dependency_projects.add(loc['project'])
    
    for dep_project in dependency_projects:
        if not dep_project:
            continue
        
        dep_info = None
        if normalized_srcmap:
            dep_path = f"/src/{dep_project}"
            if dep_path in normalized_srcmap:
                dep_info = normalized_srcmap[dep_path]
        
        dep_url = dep_info.get('url', '') if dep_info else ''
        dep_commit = dep_info.get('rev', '') if dep_info else ''
        
        dep_snippets = []
        for loc in code_locations:
            if loc.get('project') == dep_project:
                if dep_url:
                    snippet = get_code_snippet_from_git(
                        dep_url,
                        dep_commit,
                        loc['file_path'],
                        loc['line'],
                        context_lines
                    )
                else:
                    snippet = get_code_snippet_from_local(
                        loc['file_path'],
                        loc['line'],
                        context_lines
                    )
                
                if snippet:
                    snippet['location'] = loc
                    dep_snippets.append(snippet)
        
        if dep_snippets:
            dependency_snippets[dep_project] = {
                'project_name': dep_project,
                'repository_url': dep_url or 'N/A',
                'commit_sha': dep_commit or 'N/A',
                'snippets': dep_snippets
            }
    
    return {
        'stack_trace_locations': code_locations,
        'code_snippets': {
            'main_project': {
                'project_name': project_name or 'N/A',
                'repository_url': main_project_url or 'N/A',
                'commit_sha': main_project_commit or 'N/A',
                'snippets': main_project_snippets
            } if project_name else {},
            'dependencies': dependency_snippets
        }
    }


def extract_data(local_id: int, include_code_snippets: bool = False, code_context_lines: int = 15, auto_fetch: bool = True) -> Optional[Dict]:
    """Extract all data for a specific localId
    
    Args:
        local_id: Local ID to extract
        include_code_snippets: Whether to include code snippets
        code_context_lines: Number of context lines for code snippets
        auto_fetch: Whether to auto-download srcmap and generate patch_diff if missing
    """
    print(f"\n[+] Extracting data for localId: {local_id}")
    
    report = get_report_from_db(local_id)
    if not report:
        print(f"[-] No report found in database for localId: {local_id}")
        return None
    
    project_name = report.get('project', '')
    stack_trace = report.get('crash_output', '')
    
    result = {
        'localId': local_id,
        'osssfuzz_report': {
            'project_name': project_name,
            'bug_type': report.get('crash_type', ''),
            'severity': report.get('severity', ''),
            'fuzzer': report.get('fuzz_engine', ''),
            'sanitizer': report.get('sanitizer', ''),
            'fuzz_target': report.get('fuzz_target', ''),
            'language': report.get('language', ''),
            'report_url': report.get('report', f'https://issues.oss-fuzz.com/issues/{local_id}')
        },
        'stack_trace': stack_trace,
        'patch_info': {
            'fix_commit': report.get('fix_commit', ''),
            'patch_url': report.get('patch_url', ''),
            'submodule_bug': bool(report.get('submodule_bug', 0)),
            'patch_located': bool(report.get('patch_located', 0)),
            'patch_diff': '',
            'patched_files': []
        },
        'srcmap': {}
    }
    
    print(f"  [*] Looking for srcmap files...")
    srcmaps = get_srcmap_files(local_id, auto_download=auto_fetch)
    srcmap_data = {}
    
    if srcmaps:
        print(f"  [+] Found {len(srcmaps)} srcmap file(s)")
        if len(srcmaps) >= 1:
            deps = extract_dependencies_from_srcmap(srcmaps[0])
            result['srcmap']['vulnerable_version'] = {
                'file': str(srcmaps[0]),
                'dependencies': deps
            }
            print(f"  [+] Extracted {len(deps)} dependencies from vulnerable version")
            
            try:
                with open(srcmaps[0], 'r') as f:
                    srcmap_data = json.load(f)
            except:
                pass
        
        if len(srcmaps) >= 2:
            deps = extract_dependencies_from_srcmap(srcmaps[1])
            result['srcmap']['fixed_version'] = {
                'file': str(srcmaps[1]),
                'dependencies': deps
            }
            print(f"  [+] Extracted {len(deps)} dependencies from fixed version")
    else:
        print(f"  [-] No srcmap files found")
    
    print(f"  [*] Looking for patch diff...")
    diff_file = get_patch_diff(local_id, auto_generate=auto_fetch, report_data=report)
    if diff_file:
        print(f"  [+] Found patch diff: {diff_file}")
        try:
            with open(diff_file, 'rb') as f:
                result['patch_info']['patch_diff'] = f.read().decode('utf-8', errors='ignore')
            result['patch_info']['patched_files'] = extract_patched_files(diff_file)
            # Store patch file path for reference (prefer cached path over temp path)
            patch_file_path = str(diff_file)
            # If temp file, try to find cached version
            if '/tmp/' in patch_file_path or 'tmp' in patch_file_path:
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
            result['patch_info']['patch_file_path'] = patch_file_path
            print(f"  [+] Extracted patch diff ({len(result['patch_info']['patch_diff'])} bytes)")
            print(f"  [+] Found {len(result['patch_info']['patched_files'])} patched files")
        except Exception as e:
            print(f"  [-] Error reading patch diff: {e}")
    else:
        print(f"  [-] No patch diff file found")
    
    if include_code_snippets:
        print(f"  [*] Extracting code snippets...")
        code_data = extract_code_snippets_from_stack_trace(
            stack_trace,
            project_name,
            srcmap_data,
            code_context_lines
        )
        result['stack_trace_locations'] = code_data['stack_trace_locations']
        result['code_snippets'] = code_data['code_snippets']
        print(f"  [+] Extracted {len(code_data['stack_trace_locations'])} code locations")
        print(f"  [+] Main project snippets: {len(code_data['code_snippets']['main_project'].get('snippets', []))}")
        print(f"  [+] Dependency snippets: {len(code_data['code_snippets']['dependencies'])}")
    
    return result


def get_localIds_from_db(project: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[int]:
    """Get list of localIds from database"""
    conn = sqlite3.connect(DB_PATH)
    try:
        if project:
            query = "SELECT localId FROM arvo WHERE reproduced = 1 AND project = ? ORDER BY localId DESC"
            params = (project,)
            if limit:
                query += " LIMIT ? OFFSET ?"
                params = (project, limit, offset)
            elif offset > 0:
                query += " OFFSET ?"
                params = (project, offset)
        else:
            query = "SELECT localId FROM arvo WHERE reproduced = 1 ORDER BY localId DESC"
            params = ()
            if limit:
                query += " LIMIT ? OFFSET ?"
                params = (limit, offset)
            elif offset > 0:
                query += " OFFSET ?"
                params = (offset,)
        
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Extract ARVO data from database')
    parser.add_argument('--localId', type=int, help='Single localId to extract')
    parser.add_argument('--localIds', type=int, nargs='+', help='Multiple localIds to extract')
    parser.add_argument('--localIds-file', type=str, help='File containing localIds (one per line)')
    parser.add_argument('--project', type=str, help='Extract from specific project')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number to extract (default: 10, use 0 or --all for all records)')
    parser.add_argument('--all', action='store_true', help='Extract all records from database')
    parser.add_argument('-o', '--output', type=str, default='extracted_data.json', help='Output file')
    parser.add_argument('--code-snippets', action='store_true', help='Include code snippets (requires network)')
    parser.add_argument('--code-context', type=int, default=15, help='Context lines for code snippets (default: 15)')
    parser.add_argument('--auto-fetch', action='store_true', default=True, help='Auto download srcmap and generate patch_diff if missing (default: True)')
    parser.add_argument('--no-auto-fetch', dest='auto_fetch', action='store_false', help='Do not auto download srcmap or generate patch_diff')
    parser.add_argument('--skip-existing', type=str, help='Skip localIds that already exist in this JSON file')
    parser.add_argument('--offset', type=int, default=0, help='Offset for pagination (skip first N records)')
    
    args = parser.parse_args()
    
    if args.localId:
        local_ids = [args.localId]
    elif args.localIds_file:
        local_ids_file = Path(args.localIds_file)
        if not local_ids_file.exists():
            print(f"[-] File not found: {args.localIds_file}")
            sys.exit(1)
        with open(local_ids_file, 'r') as f:
            local_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"[+] Read {len(local_ids)} localIds from file: {args.localIds_file}")
    elif args.localIds:
        local_ids = args.localIds
    elif args.project:
        if args.all or args.num == 0:
            local_ids = get_localIds_from_db(project=args.project, limit=None, offset=args.offset)
        else:
            local_ids = get_localIds_from_db(project=args.project, limit=args.num, offset=args.offset)
    else:
        if args.all or args.num == 0:
            local_ids = get_localIds_from_db(limit=None, offset=args.offset)
        else:
            local_ids = get_localIds_from_db(limit=args.num, offset=args.offset)
    
    if args.skip_existing:
        skip_file = Path(args.skip_existing)
        if skip_file.exists():
            print(f"[*] Checking existing localIds in {args.skip_existing}...")
            with open(skip_file, 'r') as f:
                existing_data = json.load(f)
            existing_ids = {item['localId'] for item in existing_data.get('data', [])}
            before_count = len(local_ids)
            local_ids = [lid for lid in local_ids if lid not in existing_ids]
            skipped_count = before_count - len(local_ids)
            if skipped_count > 0:
                print(f"[+] Skipped {skipped_count} localIds that already exist in {args.skip_existing}")
                print(f"[+] Remaining: {len(local_ids)} localIds to extract")
            else:
                print(f"[+] No existing localIds found in {args.skip_existing}")
        else:
            print(f"[-] Skip file not found: {args.skip_existing}")
    
    if not local_ids:
        print("[-] No localIds found")
        return
    
    print(f"[+] Found {len(local_ids)} localIds to extract")
    print(f"[+] LocalIds: {local_ids[:10]}{'...' if len(local_ids) > 10 else ''}")
    
    results = []
    failed = []
    
    for local_id in local_ids:
        try:
            data = extract_data(local_id, include_code_snippets=args.code_snippets, code_context_lines=args.code_context, auto_fetch=args.auto_fetch)
            if data:
                results.append(data)
                print(f"[+] Successfully extracted {local_id}")
            else:
                failed.append(local_id)
        except Exception as e:
            print(f"[-] Error extracting {local_id}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(local_id)
    
    output_file = Path(args.output)
    
    existing_data = None
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"[*] Found existing file: {args.output}")
            print(f"[*] Existing records: {len(existing_data.get('data', []))}")
            
            existing_ids = {item['localId'] for item in existing_data.get('data', [])}
            new_ids = {item['localId'] for item in results}
            
            duplicates = existing_ids & new_ids
            if duplicates:
                print(f"[!] Warning: {len(duplicates)} localIds already exist")
                print(f"[!] Duplicate localIds: {list(duplicates)[:10]}{'...' if len(duplicates) > 10 else ''}")
                print(f"[!] Existing records will be kept, only new records will be added")
            
            existing_items = {item['localId']: item for item in existing_data.get('data', [])}
            for item in results:
                existing_items[item['localId']] = item
            
            merged_data = list(existing_items.values())
            existing_summary = existing_data.get('summary', {})
            
            output = {
                'summary': {
                    'total': existing_summary.get('total', 0) + len(local_ids),
                    'success': len(merged_data),
                    'failed': existing_summary.get('failed', 0) + len(failed),
                    'failed_localIds': list(set(existing_summary.get('failed_localIds', []) + failed))
                },
                'data': merged_data
            }
            print(f"[+] Merged with existing data: {len(merged_data)} total records")
        except Exception as e:
            print(f"[-] Failed to read existing file: {e}")
            print(f"[-] Overwriting with new file")
            output = {
                'summary': {
                    'total': len(local_ids),
                    'success': len(results),
                    'failed': len(failed),
                    'failed_localIds': failed
                },
                'data': results
            }
    else:
        output = {
            'summary': {
                'total': len(local_ids),
                'success': len(results),
                'failed': len(failed),
                'failed_localIds': failed
            },
            'data': results
        }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] Extraction complete!")
    print(f"[+] Success: {len(results)}")
    print(f"[+] Failed: {len(failed)}")
    print(f"[+] Output saved to: {args.output}")
    
    # Print paper metrics summary
    print_paper_metrics_summary(output.get('summary', {}), results)


def print_paper_metrics_summary(summary: Dict, results: List[Dict]):
    """Print paper metrics summary in a readable format"""
    total = summary.get('total', len(results))
    success = summary.get('success', len(results))
    
    if total == 0:
        return
    
    # Count unique projects
    projects = set()
    for item in results:
        project = item.get('osssfuzz_report', {}).get('project_name', '')
        if project:
            projects.add(project)
    
    print("\n" + "=" * 80)
    print("📊 PAPER METRICS SUMMARY (Data Extraction)")
    print("=" * 80)
    
    print(f"\n📈 Extraction Statistics:")
    print(f"  • Total vulnerabilities processed: {total:,}")
    print(f"  • Successfully extracted: {success:,}")
    print(f"  • Failed: {summary.get('failed', 0):,}")
    print(f"  • Unique projects: {len(projects)}")
    
    print(f"\n📝 Paper Values:")
    print(f"  • **{total:,}** - ARVO database total vulnerabilities")
    print(f"  • **{len(projects)}+** - Projects spanned")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

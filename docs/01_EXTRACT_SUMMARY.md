# 01 Extract from Database - 데이터 추출 로직 정리

## 개요

`01_extract_from_db.py`는 ARVO 데이터베이스(`arvo.db`)에서 취약점 정보를 추출하는 스크립트입니다. OSS-Fuzz 리포트, 스택 트레이스, 패치 정보, 의존성 정보 등을 종합적으로 추출합니다.

---

## 데이터베이스 구조

### 테이블: `arvo`

주요 컬럼:
- `localId`: 고유 식별자 (OSS-Fuzz 이슈 ID)
- `project`: 프로젝트 이름
- `crash_type`: 크래시 타입 (bug type)
- `crash_output`: 스택 트레이스 (전체 크래시 출력)
- `fix_commit`: 수정 커밋 SHA (여러 개일 수 있음, 줄바꿈으로 구분)
- `repo_addr`: 저장소 주소
- `submodule_bug`: 서브모듈 버그 여부 (0/1)
- `patch_located`: 패치 위치 확인 여부 (0/1)
- `patch_url`: 패치 URL
- `severity`: 심각도
- `fuzz_engine`: 퍼저 엔진
- `sanitizer`: 사용된 sanitizer
- `fuzz_target`: 퍼징 타겟
- `language`: 프로그래밍 언어
- `reproduced`: 재현 성공 여부 (0/1)

---

## 추출 데이터 구조

### 1. OSS-Fuzz Report 정보
```python
'osssfuzz_report': {
    'project_name': str,      # 프로젝트 이름
    'bug_type': str,          # 크래시 타입
    'severity': str,          # 심각도
    'fuzzer': str,            # 퍼저 엔진
    'sanitizer': str,         # Sanitizer
    'fuzz_target': str,       # 퍼징 타겟
    'language': str,          # 언어
    'report_url': str         # 리포트 URL
}
```

### 2. Stack Trace
- 원본 크래시 출력 전체 (`crash_output` 필드)
- 파싱된 스택 트레이스 위치 정보 (선택적)

### 3. Patch Info
```python
'patch_info': {
    'fix_commit': str,        # 수정 커밋 SHA
    'patch_url': str,         # 패치 URL
    'submodule_bug': bool,    # 서브모듈 버그 여부
    'patch_located': bool,    # 패치 위치 확인 여부
    'patch_diff': str,        # 패치 diff 내용
    'patched_files': List[str], # 패치된 파일 경로 리스트
    'patch_file_path': str    # 패치 파일 경로
}
```

### 4. Srcmap (의존성 정보)
```python
'srcmap': {
    'vulnerable_version': {
        'file': str,          # srcmap 파일 경로
        'dependencies': [     # 의존성 리스트
            {
                'path': str,      # 경로 (예: /src/libjxl)
                'name': str,      # 이름 (예: libjxl)
                'url': str,       # 저장소 URL
                'type': str,      # 타입 (git/hg/svn)
                'commit_sha': str # 커밋 SHA
            },
            ...
        ]
    },
    'fixed_version': {        # (선택적) 수정 버전의 의존성
        'file': str,
        'dependencies': [...]
    }
}
```

### 5. Code Snippets (선택적)
```python
'stack_trace_locations': [    # 스택 트레이스 위치 정보
    {
        'file_path': str,
        'line': int,
        'column': int,
        'project': str,
        'context': str
    },
    ...
],
'code_snippets': {
    'main_project': {
        'snippets': [...]
    },
    'dependencies': [
        {
            'dependency': str,
            'snippets': [...]
        },
        ...
    ]
}
```

---

## 주요 함수

### 1. `get_report_from_db(local_id: int) -> Optional[Dict]`
- **목적**: 데이터베이스에서 리포트 정보 조회
- **반환**: 리포트 딕셔너리 (모든 컬럼 포함)

### 2. `get_srcmap_files(local_id: int, auto_download: bool = False) -> List[Path]`
- **목적**: srcmap 파일 찾기
- **검색 경로**:
  1. `/data/oss-out/{local_id}`
  2. `/data/oss-work/{local_id}`
  3. `/data/issues/{local_id}`
  4. `/root/Git/ARVO/arvo/NewTracker/Issues/{local_id}_files`
- **자동 다운로드**: `auto_download=True`일 때 메타데이터에서 다운로드 시도

### 3. `get_patch_diff(local_id: int, auto_generate: bool = False, report_data: Optional[Dict] = None) -> Optional[Path]`
- **목적**: 패치 diff 파일 찾기 또는 생성
- **검색 경로**:
  1. `/data/patches/{local_id}`
  2. `/root/Git/ARVO/arvo/patches/{local_id}`
- **자동 생성**: `auto_generate=True`일 때 Git에서 diff 생성
  - 저장소 타입 자동 감지 (git/hg/svn)
  - Mercurial 실패 시 GitHub mirror 시도
  - 여러 커밋이 있으면 각각 diff 생성 후 병합

### 4. `extract_dependencies_from_srcmap(srcmap_path: Path) -> List[Dict]`
- **목적**: srcmap.json 파일에서 의존성 정보 추출
- **srcmap 형식**: `{"/src/project": {"url": "...", "rev": "...", "type": "..."}, ...}`
- **반환**: 의존성 리스트 (path, name, url, type, commit_sha)

### 5. `extract_patched_files(diff_file: Path) -> List[str]`
- **목적**: diff 파일에서 패치된 파일 경로 추출
- **패턴**: `diff --git a/path/to/file b/path/to/file` 라인 파싱

### 6. `parse_stack_trace(stack_trace: str) -> List[Dict]`
- **목적**: 스택 트레이스에서 파일 경로와 라인 번호 추출
- **패턴**:
  - `/src/[path]:[line]:[column]`
  - `../../src/[path]:[line]:[column]`
  - `../../../src/[path]:[line]:[column]`
- **반환**: 코드 위치 리스트 (file_path, line, column, project, context)

### 7. `extract_code_snippets_from_stack_trace(...) -> Dict`
- **목적**: 스택 트레이스 위치에서 코드 스니펫 추출
- **소스**: 로컬 파일 또는 Git 저장소
- **컨텍스트**: 지정된 라인 수만큼 앞뒤 코드 포함

### 8. `extract_data(local_id: int, include_code_snippets: bool = False, code_context_lines: int = 15, auto_fetch: bool = True) -> Optional[Dict]`
- **목적**: 모든 데이터를 종합적으로 추출하는 메인 함수
- **프로세스**:
  1. 데이터베이스에서 리포트 조회
  2. srcmap 파일 찾기/다운로드
  3. 패치 diff 찾기/생성
  4. (선택적) 코드 스니펫 추출
- **반환**: 통합된 데이터 딕셔너리

---

## 사용 예시

### 기본 사용
```bash
# 단일 localId 추출
python3 01_extract_from_db.py --localId 40096184

# 여러 localId 추출
python3 01_extract_from_db.py --localIds 40096184 40096185

# 프로젝트별 추출
python3 01_extract_from_db.py --project skia -n 5

# 코드 스니펫 포함
python3 01_extract_from_db.py --localId 40096184 --code-snippets
```

### 출력 형식
- 기본: `extracted_data.json` (JSON 형식)
- 각 localId별로 추출된 데이터 저장

---

## 데이터 흐름

```
arvo.db (SQLite)
    ↓
get_report_from_db()
    ↓
[Report Data]
    ├─→ OSS-Fuzz Report 정보
    ├─→ Stack Trace (crash_output)
    └─→ Patch Info (fix_commit, repo_addr)
         ↓
    get_patch_diff() → Patch Diff 파일
         ↓
    extract_patched_files() → 패치된 파일 리스트

get_srcmap_files()
    ↓
[srcmap.json 파일들]
    ↓
extract_dependencies_from_srcmap()
    ↓
[의존성 리스트]

(선택적) parse_stack_trace() + extract_code_snippets_from_stack_trace()
    ↓
[코드 스니펫]

    ↓
extract_data() → 통합 데이터 구조
    ↓
extracted_data.json
```

---

## 주요 특징

### 1. 자동 다운로드/생성
- `auto_fetch=True`일 때:
  - srcmap 파일이 없으면 자동 다운로드 시도
  - 패치 diff가 없으면 Git에서 자동 생성

### 2. 저장소 타입 자동 감지
- Git, Mercurial, SVN 자동 감지
- Mercurial 실패 시 GitHub mirror 시도

### 3. 다중 커밋 지원
- `fix_commit`에 여러 커밋이 있으면 각각 diff 생성 후 병합

### 4. 에러 처리
- 파일이 없어도 계속 진행 (다른 데이터는 추출)
- 각 단계별 에러 메시지 출력

---

## 의존성

- `sqlite3`: 데이터베이스 접근
- `arvo.utils_meta`: 메타데이터 및 다운로드 유틸리티
- `arvo.utils_git`: Git 작업 유틸리티
- `arvo.utils`: 이슈 정보 유틸리티

---

## 출력 파일

### `extracted_data.json`
```json
{
  "localId": 40096184,
  "osssfuzz_report": {...},
  "stack_trace": "...",
  "patch_info": {...},
  "srcmap": {...},
  "stack_trace_locations": [...],  // 선택적
  "code_snippets": {...}            // 선택적
}
```

---

## 참고사항

1. **데이터베이스 경로**: `/root/Git/arvo.db` (하드코딩)
2. **srcmap 파일**: 빌드 아티팩트에서 추출, `.srcmap.json` 확장자
3. **패치 diff**: Git 커밋에서 생성하거나 미리 저장된 파일 사용
4. **코드 스니펫**: 선택적 기능, 로컬 파일 또는 Git에서 추출
5. **에러 복구**: 각 단계가 실패해도 다음 단계 계속 진행














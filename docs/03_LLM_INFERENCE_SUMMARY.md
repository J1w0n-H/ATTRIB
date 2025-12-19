# 03 LLM Inference Modules - 계층적 LLM 기반 추론 파이프라인

## 개요

`03_llm_inference_modules.py`는 Ground Truth 데이터에 LLM을 적용하여 취약점의 근본 원인을 추론하는 계층적 파이프라인입니다. Module 0 (전처리)과 3개의 추론 모듈로 구성되어 있으며, 각 모듈은 서로 다른 추상화 수준에서 분석을 수행합니다. 특히 **Module 3은 새로운 분류기가 아니라 Module 1 결과를 교차 프로젝트 증거로 검증·정제하는 validation/consolidation 레이어**입니다.

---

## 파이프라인 구조

### 전체 흐름

```
Ground Truth (02_build_ground_truth.py)
    ↓
[Module 3.1: Feature Extraction (전처리)]
    ├─→ Stack Trace 요약 (LLM, 필수)
    ├─→ Patch 요약 (LLM, 선택적: 실험 모드에서 생략 가능)
    ├─→ Dependencies 요약 (LLM, 선택적: 실험 모드에서 생략 가능)
    └─→ 구조적 특성 계산 (코드 기반, 필수): patch-crash distance, patch semantic type, crash_module, patched_module
         ↓
[Module 1: Bug Type Grouping + Individual Inference]
    ├─→ Bug Type별 그룹화 (코드 기반)
    └─→ 개별 취약점 Root Cause 추론 (LLM)
         ↓
[Module 2: Dependency-Based Sub-Grouping]
    ├─→ Module 1 결과 기반 의존성 그룹화 (코드 기반)
    └─→ Sub-Group 생성
         ↓
[Module 3: Cross-Project Pattern Validation]
    ├─→ 개별 추론 + 그룹 패턴 종합 (LLM)
    ├─→ Cross-project 패턴 검증
    └─→ Confidence Score 조정
         ↓
LLM Inference Results
```

---

## 모듈별 상세 설명

### Module 3.1: Vulnerability Feature Extraction (전처리)

**목적**: 취약점 특성 추출 및 요약 (LLM 추론을 위한 전처리 단계)

**주요 기능**:
1. **Stack Trace 요약** (LLM, 필수): LLM을 사용하여 스택 트레이스 요약
2. **Patch 요약** (LLM, 선택적): 패치 diff 분석 및 요약
   - `--no-patch-summary`로 비활성화 가능 (실험 모드에서 빠른 실행)
3. **Dependencies 요약** (LLM, 선택적): 의존성 라이브러리 정보 요약
   - `--no-dependency-description`로 비활성화 가능 (실험 모드에서 빠른 실행)
4. **Code Snippets 요약**: 코드 스니펫 요약 (메인 프로젝트 vs 의존성 분리)
5. **LLM Reasoning Summary** (LLM, 선택적): Chain-of-Thought 추론 요약
   - `--no-reasoning-summary`로 비활성화 가능 (가장 긴 단계, 실험 모드에서 생략)

**구조적 특성 계산** (코드 기반, 필수):
- `patch_crash_distance`: 패치-크래시 구조적 거리 (0-3)
- `patch_semantic_type`: 패치 시맨틱 타입 (VALIDATION_ONLY, ALGORITHM_CHANGE, etc.)
- `crash_module`: 크래시 발생 모듈
- `patched_module`: 패치 적용 모듈
- `control_flow_only`: 제어 흐름만 추가하는 패치 여부
- `workaround_detected`: GT에서 로드하거나 계산 (patch_crash_distance >= 2 AND module mismatch)

**참고**: 코드에서는 "Module 3.1"로 명명되지만, 논문에서는 "Module 0"으로도 언급될 수 있습니다.

**선택적 의미**:
- **실험 모드**: 빠른 실행을 위해 LLM 요약 생략 가능 (`--no-patch-summary`, `--no-dependency-description`, `--no-reasoning-summary`)
  - 비활성화 시 최소한의 텍스트로 대체 (예: "Patch diff available (N chars)")
- **논문 모드**: 모든 LLM 요약 활성화 (`--paper-mode`)
- **필수 항목**: Stack Trace 요약과 구조적 특성 계산은 항상 수행
- **주의사항**: 
  - 비활성화 시 LLM 프롬프트에 포함되는 정보가 줄어들어 추론 결과에 영향을 줄 수 있음
  - 구조적 특성 계산(`patch-crash distance`, `patch semantic type`)은 `patch_diff`와 `patch_file_path`를 우선 사용하므로 큰 영향 없음
  - 하지만 LLM이 받는 컨텍스트가 줄어들면 추론 정확도가 떨어질 수 있음

**최적화 옵션**:
- `--no-patch-summary`: 패치 요약 비활성화 (실험 모드, 빠름)
- `--no-dependency-description`: 의존성 설명 비활성화 (실험 모드, 빠름)
- `--no-reasoning-summary`: 추론 요약 비활성화 (실험 모드, 빠름)
- `--paper-mode`: 모든 LLM 요약 활성화 (논문/리포트용)

---

### Module 1: Bug Type Grouping + Individual Root Cause Inference

**목적**: Bug Type별 그룹화 및 개별 취약점 Root Cause 추론

**처리 과정**:

1. **Bug Type 그룹화** (코드 기반):
   - `crash_type` (bug_type) 기준으로 그룹화
   - LLM 사용 없음 (효율적)

2. **개별 Root Cause 추론** (LLM):
   - 각 취약점에 대해 LLM으로 Root Cause 추론
   - **GT와 독립적으로 수행** (GT 정보가 프롬프트에 포함되지 않음)
   - Workaround 패치 감지에 집중
   - Deterministic workaround detection 사용 (명확한 경우)
   - **중요**: GT 검증은 Module 3에서 수행됨

**출력**:
- `BugTypeGroupInfo`: Bug Type별 그룹 정보
  - `bug_type`: 버그 타입
  - `localIds`: 그룹 내 localId 리스트
  - `common_dependencies_in_group`: 그룹 내 공통 의존성
  - `individual_root_causes`: 개별 추론 결과 딕셔너리

**개별 추론 결과** (`IndividualRootCause`):
- `root_cause_type`: Main_Project_Specific 또는 Dependency_Specific
- `root_cause_dependency`: 의존성 이름 (Dependency_Specific인 경우)
- `patch_intent`: ACTUAL_FIX, WORKAROUND, DEFENSIVE
- `is_workaround_patch`: boolean (true if workaround, false if real fix, None if unknown)
- `patch_semantic_llm_opinion`: LLM의 패치 시맨틱 타입 의견
- `main_project_score`: Main_Project_Specific 점수 (0.0-1.0)
- `dependency_score`: Dependency_Specific 점수 (0.0-1.0)
- `confidence`: 전체 신뢰도 (0.0-1.0)
- `reasoning`: 추론 과정
- `evidence`: 증거

**LLM 프롬프트 핵심 요소**:
- Root Cause 정의 (Dependency_Specific vs Main_Project_Specific)
- Workaround 패치 설명
- Decision Tree 가이드
- Few-Shot 예시
- 의존성 명명 요구사항
- **구조적 특성 활용**: patch-crash distance, patch semantic type, control-flow only 등 정량적 증거 제공
- **submodule_bug 정보**: DB에서 조회하여 프롬프트에 포함 (경로만 보고 판단하는 것을 방지)
- **workaround_detected**: GT에서 로드하거나 구조적 특성으로 계산
- **주의**: GT 정보는 프롬프트에 포함되지 않음 (GT와 독립적인 추론)

---

### Module 2: Fine-Grained Semantic Sub-Grouping

**목적**: Module 1의 개별 추론 결과를 기반으로 의미론적 유사성을 판단하여 세부 Sub-Group 생성

**처리 과정**:

1. **Step A-0: Dependency 기반 사전 그룹화** (코드 기반):
   - Module 1의 `individual_root_causes`에서 dependency 이름 추출
   - 동일 dependency를 가진 케이스들을 사전 그룹화
   - 효율적인 전처리 단계로 LLM 호출 최소화

2. **Step A: Deterministic Structural Grouping** (코드 기반):
   - Step A-0의 dependency 그룹 내에서 구조적 패턴 매칭 수행
   - 함수 이름, 파일 경로, CWE ID 등으로 그룹화
   - dependency 없는 케이스도 별도로 처리
   - 효율적인 코드 기반 그룹화

3. **Step B: LLM 기반 의미론적 그룹화**:
   - Step A에서 그룹화되지 않은 케이스들에 대해 LLM 클러스터링
   - Seed-based incremental clustering 방식 사용
   - LLM이 의미론적 유사성을 판단하여 세부 그룹 형성
   - 패턴 기반 그룹화 (패치 패턴, 크래시 패턴, 코드 패턴 등)
   - 그룹화 이유 설명 생성 (CoT reasoning)

3. **Sub-Group 생성**:
   - 각 Sub-Group은 공통 Root Cause를 가진 취약점들의 집합
   - Cross-project 패턴 식별 가능

**출력**:
- `SubGroupInfo`: Sub-Group 정보
  - `sub_group_id`: Sub-Group ID
  - `bug_type_group`: Bug Type 그룹 이름
  - `localIds`: Sub-Group 내 localId 리스트
  - `pattern_description`: 취약점 패턴 설명
  - `grouping_reasoning`: 그룹화 이유 설명
  - `inferred_root_cause_type`: 추론된 Root Cause Type
  - `inferred_root_cause_dependency`: 추론된 의존성 이름
  - `common_dependency_versions`: 공통 의존성 버전 리스트
  - `confidence_score`: 신뢰도 점수

---

### Module 3: Cross-Project Pattern Validation

**목적**: 개별 추론과 그룹 패턴을 종합하여 최종 Root Cause 추론 및 검증

**처리 과정**:

1. **개별 추론 종합**:
   - Module 1의 개별 추론 결과 수집
   - Sub-Group 내 취약점들의 추론 결과 분석

2. **의존성 매칭 비율 계산**:
   - Sub-Group 내 취약점들이 동일 의존성을 공유하는 비율
   - `dependency_matching_ratio`: 0.0-1.0

3. **Cross-Project 패턴 분석** (LLM):
   - 여러 프로젝트에서 동일 의존성 문제가 발생하는지 확인
   - Cross-project propagation insight 생성

4. **최종 추론 및 GT 검증** (LLM):
   - 개별 추론 (Module 1) + 그룹 패턴 (Module 2) + Cross-project 패턴 종합
   - Confidence Score 조정
   - **Ground Truth와의 비교 및 불일치 분석** (discrepancy analysis)
   - **중요**: GT 검증은 Module 3에서만 수행됨

**최적화**:
- 단일 케이스 Sub-Group은 Module 3 스킵 (Module 1 결과 직접 사용)

**출력**:
- `RootCauseInference`: Sub-Group별 최종 추론 결과
  - `sub_group_id`: Sub-Group ID
  - `bug_type_group`: Bug Type 그룹 이름
  - `localIds`: Sub-Group 내 모든 localId
  - `group_level_root_cause_type`: 그룹 레벨 Root Cause Type
  - `group_level_root_cause_dependency`: 그룹 레벨 의존성 이름
  - `group_pattern_justification`: 그룹 패턴 기반 정당화
  - `dependency_matching_ratio`: 의존성 매칭 비율
  - `dependency_matching_count`: 의존성 매칭 개수
  - `cross_project_propagation_insight`: Cross-project 전파 분석
  - `llm_reasoning_process`: 전체 LLM 추론 과정
  - `confidence_score`: 전체 신뢰도 (0.0-1.0)
  - `main_project_score`: Main_Project_Specific 점수
  - `dependency_score`: Dependency_Specific 점수
  - `module1_confidence`: Module 1 신뢰도
  - `module2_confidence`: Module 2 신뢰도
  - `module3_confidence`: Module 3 신뢰도
  - `discrepancy_analysis`: Ground Truth와의 불일치 분석
  - `discrepancy_type`: 불일치 타입 (heuristic_error, llm_error, borderline_case)
  - `corrective_reasoning`: GT와 불일치 시 반박 추론
  - `per_localId_discrepancies`: localId별 불일치 상세 정보

---

## 주요 데이터 구조

### VulnerabilityFeatures
```python
@dataclass
class VulnerabilityFeatures:
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
    # 구조적 특성
    patch_crash_distance: Optional[int] = None
    patch_semantic_type: Optional[str] = None
    patched_module: Optional[str] = None
    crash_module: Optional[str] = None
    control_flow_only: Optional[bool] = None
    workaround_detected: Optional[bool] = None  # from Phase 1 GT or computed
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

### IndividualRootCause
```python
@dataclass
class IndividualRootCause:
    localId: int
    root_cause_type: str  # Main_Project_Specific or Dependency_Specific
    root_cause_dependency: Optional[str] = None
    patch_intent: Optional[str] = None  # ACTUAL_FIX, WORKAROUND, DEFENSIVE
    is_workaround_patch: Optional[bool] = None  # True if workaround, False if real fix, None if unknown
    patch_semantic_llm_opinion: Optional[str] = None
    main_project_score: float = 0.0
    dependency_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    evidence: str = ""
```

### RootCauseInference
```python
@dataclass
class RootCauseInference:
    sub_group_id: int
    bug_type_group: str
    localIds: List[int]
    group_level_root_cause_type: str
    group_level_root_cause_dependency: Optional[str] = None
    group_pattern_justification: str
    dependency_matching_ratio: float = 0.0
    dependency_matching_count: int = 0
    cross_project_propagation_insight: Optional[str] = None
    llm_reasoning_process: str = ""
    confidence_score: float = 0.0
    main_project_score: float = 0.0
    dependency_score: float = 0.0
    module1_confidence: Optional[float] = None
    module2_confidence: Optional[float] = None
    module3_confidence: Optional[float] = None
    discrepancy_analysis: Optional[str] = None
    discrepancy_type: Optional[str] = None
    corrective_reasoning: Optional[str] = None
    per_localId_discrepancies: List[Dict] = None
```

---

## 핵심 설계 원칙

### 1. 계층적 추론
- **Module 1**: 개별 취약점 분석 (세부적)
- **Module 2**: 의존성 기반 그룹화 (중간)
- **Module 3**: Cross-project 패턴 검증 (전역)

### 2. 효율성 최적화
- Bug Type 그룹화: 코드 기반 (LLM 없음)
- Sub-Group 생성: 하이브리드 접근
  - Step A-0, A: 코드 기반 사전 그룹화 (LLM 없음)
  - Step B: LLM 기반 의미론적 그룹화 (ungrouped 케이스만)
- LLM 호출 최소화: 
  - 대부분의 케이스는 코드 기반 그룹화로 처리
  - LLM은 개별 추론(Module 1), 의미론적 그룹화(Module 2 Step B), 최종 검증(Module 3)에만 사용

### 3. Deterministic Workaround Detection
- 명확한 workaround 패턴은 LLM 없이 감지
- `use_deterministic_workaround_detection=True` (기본값)
- 조건: `patch_crash_distance >= 2` AND `module_mismatch` AND `control_flow_only`

### 4. Cross-Project 패턴 활용
- 여러 프로젝트에서 동일 의존성 문제 발생 시 강한 증거
- Dependency_Specific 판단의 신뢰도 향상

### 5. Discrepancy Analysis
- Ground Truth와의 불일치 자동 분석
- 불일치 타입 분류: heuristic_error, llm_error, borderline_case
- 반박 추론 (corrective reasoning) 생성

---

## 사용 예시

### 기본 사용
```bash
# 전체 파이프라인 실행
python3 03_llm_inference_modules.py --gt-file ground_truth.json -n 50

# 특정 프로젝트만 처리
python3 03_llm_inference_modules.py --gt-file ground_truth.json --project imagemagick

# 특정 Bug Type만 처리
python3 03_llm_inference_modules.py --gt-file ground_truth.json --bug-type "heap-buffer-overflow"

# 특정 모듈만 실행
python3 03_llm_inference_modules.py --gt-file ground_truth.json --module 1
python3 03_llm_inference_modules.py --gt-file ground_truth.json --module 2
python3 03_llm_inference_modules.py --gt-file ground_truth.json --module 3

# 특정 모듈까지 실행
python3 03_llm_inference_modules.py --gt-file ground_truth.json --stop-after-module 2
```

### 최적화 옵션
```bash
# 실험 모드 (빠름, LLM 요약 비활성화)
python3 03_llm_inference_modules.py --gt-file ground_truth.json \
    --no-patch-summary --no-dependency-description --no-reasoning-summary

# 논문 모드 (모든 LLM 요약 활성화)
python3 03_llm_inference_modules.py --gt-file ground_truth.json --paper-mode
```

### LLM 모델 선택
```bash
# 기본: o4-mini
python3 03_llm_inference_modules.py --gt-file ground_truth.json --llm-model o4-mini

# 다른 모델 사용
python3 03_llm_inference_modules.py --gt-file ground_truth.json --llm-model gpt-4
```

### 체크포인트 및 재개
```bash
# 체크포인트 저장 간격 설정
python3 03_llm_inference_modules.py --gt-file ground_truth.json --checkpoint-interval 20

# 체크포인트에서 재개
python3 03_llm_inference_modules.py --gt-file ground_truth.json --resume-from checkpoint.json
```

---

## 출력 파일

### 기본 출력
- `llm_inference_results.json`: 전체 결과 (기본)
- `llm_inference_results_core.json`: 핵심 결과만 (추론 필드)
- `llm_inference_results_explain.json`: 설명 필드만 (LLM 요약)

### 출력 구조
```json
{
  "summary": {
    "total_processed": int,
    "module1_completed": int,
    "module2_completed": int,
    "module3_completed": int,
    "failed": int,
    "failed_localIds": [...]
  },
  "features": [...],  // Module 0 결과
  "bug_type_groups": [...],  // Module 1 결과
  "sub_groups": [...],  // Module 2 결과
  "root_cause_inferences": [...]  // Module 3 결과
}
```

---

## LLM 프롬프트 핵심 요소

### Root Cause 정의
- **Dependency_Specific**: 근본 원인이 외부 라이브러리/의존성에 있음
- **Main_Project_Specific**: 근본 원인이 메인 프로젝트 고유 코드에 있음
- **Workaround 패치**: 메인 프로젝트에서 의존성 버그를 우회하는 패치

### Decision Tree 가이드
1. 스택 트레이스에서 의존성 코드 경로 분석
2. 특정 의존성 이름 식별 (단일, 구체적)
3. 패치 위치 및 타입 분석
4. 의존성 매칭 비율 확인
5. Cross-project 검증
6. 최종 결정

### Few-Shot 예시
- Dependency_Specific 예시 (libavcodec, libjxl)
- Main_Project_Specific 예시
- Workaround 패치 예시

---

## 주요 특징

### 1. 계층적 분석
- 개별 → 그룹 → Cross-project 순서로 분석
- 각 단계에서 신뢰도 향상

### 2. 효율성
- 코드 기반 그룹화로 LLM 호출 최소화
- Deterministic workaround detection으로 명확한 경우 LLM 스킵

### 3. Cross-Project 패턴 활용
- 여러 프로젝트에서 동일 의존성 문제 발생 시 강한 증거
- Dependency_Specific 판단의 신뢰도 향상

### 4. Discrepancy Analysis
- Ground Truth와의 불일치 자동 분석
- 불일치 타입 분류 및 반박 추론 생성

### 5. 모듈별 독립 실행
- 각 모듈을 독립적으로 실행 가능
- 중간 결과 저장 및 재개 지원

---

## 데이터 처리 범위

- **전체 Ground Truth 케이스 처리**: `ground_truth.json`의 모든 케이스 처리
- **의존성 0개 케이스 포함**: `main_only=True` 케이스도 처리됨
- **필터링 없음**: 현재 버전은 의존성 개수에 관계없이 모든 케이스 처리

**참고**: 의존성 0개 케이스는 Phase 1에서 자동으로 Main_Project_Specific으로 분류되지만, LLM 추론도 수행되어 GT와의 일치성을 검증할 수 있습니다.

---

## 참고사항

1. **LLM API Key 필수**: `OPENAI_API_KEY` 환경 변수 또는 `--llm-api-key` 옵션
2. **Ground Truth 파일 필수**: `--gt-file` 옵션으로 Ground Truth JSON 파일 지정
3. **코드 스니펫**: 기본적으로 비활성화 (느림/타임아웃 가능), `--include-code-snippets`로 활성화
4. **체크포인트**: 대량 처리 시 체크포인트 사용 권장
5. **모듈별 실행**: 특정 모듈만 실행하여 디버깅 가능
6. **최적화 모드**: 실험 모드에서 LLM 요약 비활성화로 속도 향상
7. **데이터 필터링**: 현재 버전은 의존성 0개 케이스를 자동으로 제외하지 않음. 필요시 수동 필터링 필요

## 실행 결과 (Use-of-uninitialized-value)

### 실행 기준 및 필터링 조건

**실행 명령어**:
```bash
python3 ./03_llm_inference_modules.py --bug-type "Use-of-uninitialized-value" --num 100
```

**필터링 프로세스**:

1. **데이터베이스 필터링**:
   - 데이터베이스에서 `crash_type = "Use-of-uninitialized-value"`인 케이스 조회
   - `reproduced = 1` 조건 적용 (재현 가능한 케이스만)
   - `ORDER BY localId DESC`로 정렬하여 최신 케이스 우선
   - `--num 100` 옵션으로 최대 100개 제한

2. **Ground Truth 필터링**:
   - 데이터베이스에서 조회된 `localId`가 Ground Truth 파일에 존재하는지 확인
   - Ground Truth의 `bug_type` 필드가 `"Use-of-uninitialized-value"`와 정확히 일치하는 케이스만 포함
   - 이중 필터링으로 데이터 일관성 보장

3. **최종 처리 대상**:
   - **총 52개 케이스** 처리 완료
   - 모두 `Use-of-uninitialized-value` 버그 타입
   - 모두 `Medium` severity

**처리된 케이스 특성**:

- **프로젝트 분포** (총 59개 프로젝트):
  - 상위 프로젝트(Top 10): `imagemagick`(137), `skia`(65), `ffmpeg`(27), `matio`(26), `wolfssl`(19), `kimageformats`(19), `poppler`(18), `mruby`(17), `leptonica`(16), `gnutls`(14)

- **Workaround 감지(heuristic flag)**:
  - Workaround 감지됨: 144개 케이스 (27.9%)
  - Workaround 미감지: 373개 케이스 (72.1%)

- **의존성 필터링**:
  - 의존성 0개 케이스도 포함 (필터링 없음)
  - 모든 케이스에 대해 의존성 정보 분석 수행

**처리 범위**:
- 모든 케이스에 대해 Module 0 (전처리) → Module 1 → Module 2 → Module 3 파이프라인 완료
- 전체 파이프라인 성공적으로 완료 (실패 케이스 없음)

### 최근 실행 결과 요약

```
================================================================================
📊 PAPER METRICS SUMMARY (Phase 2 - LLM Inference)
================================================================================

📈 Dataset Statistics:
  • Total cases processed: 517
  • Bug type groups: 1
  • Sub-groups formed: 125
  • Root cause inferences: 125

📊 Root Cause Type Distribution:
  • Main_Project_Specific: 435 cases (84.14%)
  • Dependency_Specific: 82 cases (15.86%)

📦 Top Dependencies (Dependency_Specific cases, Top 10):
  • libjpeg-turbo: 14 cases
  • libde265: 10 cases
  • libraw: 9 cases
  • zlib: 6 cases
  • aom: 6 cases
  • Image-codec libraries (libtiff, openjpeg, libjpeg-turbo): 5 cases
  • libxml2: 5 cases
  • libheif: 4 cases
  • freetype2: 4 cases
  • libxml2, HDF5: 4 cases

🔗 Sub-Group Statistics:
  • Average sub-group size: 4.14 cases
  • Largest sub-group: 11 cases
  • Smallest sub-group: 1 cases
  • Sub-groups with ≥2 cases: 117

🏗️  Project Distribution:
  • Total projects: 59
  • imagemagick: 137 cases
  • skia: 65 cases
  • ffmpeg: 27 cases
  • matio: 26 cases
  • wolfssl: 19 cases

📝 Paper Values:
  • **517** - Evaluation cases (Stage 1)
  • **125** - Distinct sub-groups formed
  • **15.86%** - Dependency_Specific prediction rate
  • **84.14%** - Main_Project_Specific prediction rate
  • **59+** - Projects spanned
```

### 주요 결과 해석

- **데이터셋 규모**: 517개 케이스 처리 완료 (UUV 단일 버그타입 슬라이스)
- **Root Cause 분포**: Main_Project_Specific (84.14%)가 다수이며, Dependency_Specific는 15.86%로 소수 클래스
- **의존성 분포**: `libjpeg-turbo`, `libde265`, `libraw` 등 이미지/미디어 관련 라이브러리가 상위에 위치
- **Sub-Group 형성**: 125개의 세부 그룹으로 분류, 평균 4.14개 케이스/그룹
- **프로젝트 다양성**: 총 59개 프로젝트에 걸쳐 분석 수행 (상위 2개 프로젝트가 다수를 차지하지만, long tail 존재)

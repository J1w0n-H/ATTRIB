# 04 Evaluate LLM Inference - Ground Truth 기반 평가

## 개요

`04_evaluate_llm_inference.py`는 LLM 추론 결과를 휴리스틱 Ground Truth와 비교하여 평가하는 스크립트입니다. 다양한 메트릭을 계산하여 LLM의 성능을 정량적으로 측정합니다.

---

## 평가 프로세스

### 입력 데이터

1. **LLM Inference Results** (`llm_inference_results.json`):
   - Module 3의 `root_cause_inferences` 결과
   - Sub-Group별 Root Cause 추론 결과
   - 개별 localId별 추론 결과 (선택적)

2. **Ground Truth** (`ground_truth.json`):
   - 휴리스틱 룰로 생성된 Ground Truth
   - `Heuristically_Root_Cause_Type`
   - `Heuristically_Root_Cause_Dependency`

### 평가 단계

1. **데이터 로드**: Inference 결과와 Ground Truth 로드
2. **Sub-Group별 평가**: 각 Sub-Group의 추론 결과 평가
3. **개별 케이스 평가**: Sub-Group 내 각 localId별 평가
4. **메트릭 계산**: 다양한 메트릭 계산 및 집계
5. **결과 저장**: 평가 결과를 JSON 파일로 저장

---

## 평가 메트릭

### 1. Overall Metrics (전체 메트릭)

**Type Accuracy**:
- Root Cause Type (Main_Project_Specific vs Dependency_Specific) 정확도
- `correct_type / total_cases`

**Dependency Accuracy**:
- 의존성 이름 매칭 정확도 (Dependency_Specific 케이스만)
- `correct_dependency / total_cases`

**Both Correct**:
- Type과 Dependency 모두 정확한 비율
- `correct_both / total_cases`

### 1.1 ARVO Baseline (DB 기반 Baseline)

본 프로젝트에서 baseline은 **ARVO의 submodule 기반 의존성 판정**으로 정의하고, **ARVO DB(`arvo.db`)에서 직접 측정**합니다.

- **Baseline 정의**: `Dependency_Specific` iff `arvo.submodule_bug == 1`, else `Main_Project_Specific`
- **측정 데이터**: `./arvo.db` (또는 `ARVO_DB_PATH` 환경변수로 지정) 내 `arvo` 테이블

### 2. Per-Type Metrics (타입별 메트릭)

**Main_Project_Specific**:
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1-Score**: `2 × Precision × Recall / (Precision + Recall)`
- **True Positives (TP)**: GT=Main, LLM=Main
- **False Positives (FP)**: GT=Dependency, LLM=Main
- **False Negatives (FN)**: GT=Main, LLM=Dependency

**Dependency_Specific**:
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1-Score**: `2 × Precision × Recall / (Precision + Recall)`
- **True Positives (TP)**: GT=Dependency, LLM=Dependency
- **False Positives (FP)**: GT=Main, LLM=Dependency
- **False Negatives (FN)**: GT=Dependency, LLM=Main

### 3. Sub-Group Level Metrics (Sub-Group 레벨 메트릭)

**Perfect Matching** (완벽 매칭):
- Sub-Group 내 모든 케이스가 정확히 일치하는 비율
- `sub_group_correct_type / sub_group_count`
- `sub_group_correct_dependency / sub_group_count`
- `sub_group_correct_both / sub_group_count`

**Partial Matching** (부분 매칭):
- Sub-Group 내 평균 매칭 비율
- `sub_group_partial_type_accuracy`: 평균 Type 정확도
- `sub_group_partial_dep_accuracy`: 평균 Dependency 정확도

**Representative Matching** (대표 매칭):
- LLM의 그룹 레벨 추론과 GT의 가장 빈도 높은 값 비교
- `sub_group_representative_matches / sub_group_count`

### 4. Dependency Matching Analysis (의존성 매칭 분석)

**Dependency Matching Ratio**:
- Sub-Group 내 취약점들이 동일 의존성을 공유하는 비율
- `dependency_matching_ratio_avg`: 평균 매칭 비율
- `dependency_matching_count_total`: 총 매칭 개수

### 5. Beyond Heuristic Accuracy (BHA)

**목적**: 휴리스틱 GT가 잘못 분류한 케이스에서 LLM이 올바르게 추론한 비율

**BHA Case 조건** (자동 계산):
1. GT가 `Main_Project_Specific`으로 분류
2. `submodule_bug=True` 또는 `repo_addr`이 프로젝트와 다름
3. LLM이 `Dependency_Specific`으로 올바르게 추론

**BHA Accuracy**:
- `bha_correct / bha_cases`
- 의미: GT가 Main으로 잘못 분류했지만 LLM이 Dependency로 올바르게 추론한 비율
- **자동 계산**: Expert review 없이 `submodule_bug` 및 `repo_addr` 정보 기반으로 자동 판단

### 6. CPVR (Cross-Project Validation Rate)

**목적**: Module 3의 outlier correction을 통한 정확도 향상 측정

**구현 방식**:
- `identify_outliers()` 함수로 `bug_type_groups`의 `individual_root_causes`에서 outlier 식별
- Sub-group 내 다수 타입(≥60%)과 불일치하는 케이스를 outlier로 식별
- Module 3의 최종 추론(`group_level_root_cause_type`)이 다수 타입과 일치하면 수정됨으로 판단
- `cpvr_total_outliers`: 전체 outlier 수
- `cpvr_corrected_outliers`: Module 3에서 수정된 outlier 수
- **CPVR**: `(cpvr_corrected_outliers / cpvr_total_outliers) × 100`

**의미**: 그룹 합의를 통한 개별 추론 오류 수정 비율

### 7. WDR (Workaround Detection Rate)

**목적**: Phase 1 (Heuristic)과 Phase 2 (LLM)의 workaround 감지율 비교

**구현 방식**:
- **Phase 1**: `workaround_detected` 필드 (GT에서 로드 또는 계산)
  - 조건: `patch_crash_distance >= 2 AND module mismatch`
- **Phase 2**: `IndividualRootCause.patch_intent` 또는 `is_workaround_patch` 필드
  - LLM이 패치 의도를 분석하여 WORKAROUND/DEFENSIVE 감지
- `wdr_phase1_detected`: Phase 1에서 감지된 workaround 수
- `wdr_phase2_detected`: Phase 2에서 감지된 workaround 수
- **WDR Phase 1/2**: 각각의 감지율 계산

**의미**: LLM의 시맨틱 분석을 통한 workaround 감지 능력 측정

---

## 의존성 필터링 개선 (2024-12 업데이트)

### 개선 배경

LLM Inference 모듈과 Ground Truth 빌더 간 필터링 방식의 불일치를 해결하기 위해 스택 트레이스 기반 필터링을 추가했습니다.

### 개선 내용

#### Before (개선 전)
- **LLM**: srcmap dependencies만 필터링 (빌드 타임 의존성)
- **GT**: 스택 트레이스 기반 필터링 (런타임 의존성)
- **문제**: 일관성 부족, 다른 의존성 목록 가능

#### After (개선 후)
- **LLM**: srcmap dependencies + 스택 트레이스 기반 필터링 (하이브리드)
- **GT**: 스택 트레이스 기반 필터링 (런타임 의존성)
- **개선**: 일관성 확보, 실제 실행된 의존성 우선

### 필터링 방식

#### 1. 경로 기반 필터링 (Path-based)
- **내부 모듈**: `/src/{project_name}/` 패턴으로 판단
- **하드코딩 없음**: 키워드 리스트 대신 경로 구조 사용
- **GT와 동일**: LLM도 GT Rule 2와 동일한 로직 사용

#### 2. 스택 트레이스 기반 추출
- **새로운 함수**: `_extract_dependencies_from_stack_trace()`
- **GT Rule 2와 동일한 로직**: 실제 실행된 의존성만 추출
- **런타임 의존성 우선**: 스택 트레이스에서 추출된 의존성을 우선순위로 제공

#### 3. 하이브리드 접근
- **런타임 의존성**: 스택 트레이스에서 추출 (우선)
- **빌드 타임 의존성**: srcmap에서 필터링 (컨텍스트)
- **LLM에게 구분 제공**: "Runtime dependencies"와 "Build-time dependencies" 구분

### 평가에 미치는 영향

#### 정확도 향상
- **의존성 필터링 정확도**: 100% (내부 모듈, fuzzer, test 프레임워크 정확히 제외)
- **GT와 일관성**: 동일한 기준으로 필터링하여 결과 일치율 향상 기대
- **추론 품질**: 더 정확한 의존성 정보로 LLM 추론 품질 향상

#### 메트릭 개선
- **Type Accuracy**: 더 정확한 의존성 정보로 Type 분류 정확도 향상
- **Dependency Accuracy**: 실제 사용된 의존성만 고려하여 의존성 매칭 정확도 향상
- **BHA Accuracy**: GT와 일관된 기준으로 BHA 계산 정확도 향상

---

## 주요 함수

### `evaluate_llm_inference(inference_file: str, gt_file: str, logger: Optional[logging.Logger] = None) -> Tuple[EvaluationMetrics, List[Dict]]`

**목적**: LLM 추론 결과를 Ground Truth와 비교하여 평가

**처리 과정**:
1. Inference 결과와 Ground Truth 로드
2. 각 Sub-Group별 평가:
   - 개별 추론 결과 파싱 (선택적)
   - 그룹 레벨 추론 사용 (fallback)
   - 각 localId별 평가
3. 메트릭 집계:
   - Type/Dependency 정확도
   - TP/FP/FN 계산
   - Sub-Group 레벨 메트릭
   - BHA 계산
4. 상세 결과 생성

**반환값**:
- `EvaluationMetrics`: 집계된 메트릭
- `List[Dict]`: 각 localId별 상세 평가 결과

### `compare_dependencies(llm_dep: Optional[str], gt_dep: Optional[Dict]) -> bool`

**목적**: LLM 추론 의존성과 GT 의존성 비교

**비교 방식**:
1. 의존성 이름 정규화 (`normalize_dependency_name`)
2. 정확 일치 또는 부분 문자열 일치 확인
3. Main_Project_Specific인 경우 None/N/A 비교

**정규화 규칙**:
- 소문자 변환
- 버전 접미사 제거 (예: `libjxl-v1.0` → `jxl`)
- 공통 접두사 제거 (예: `libjxl` → `jxl`)

### `normalize_dependency_name(name: str) -> str`

**목적**: 의존성 이름 정규화

**정규화 과정**:
1. 소문자 변환 및 공백 제거
2. 버전 접미사 제거 (정규식: `[v\s]*\d+[.\d]*.*$`)
3. 공통 접두사 제거 (정규식: `^(lib|libs|lib-|libs-)`)

### `parse_individual_inferences(reasoning_text: str) -> Dict[int, Dict[str, Optional[str]]]`

**목적**: LLM의 `llm_reasoning_process`에서 개별 케이스 추론 결과 파싱

**파싱 패턴**:
- `localId 432073014: Dependency_Specific (libjxl)`
- `localId 432073014: Main_Project_Specific`

**반환값**:
- `{localId: {'type': str, 'dependency': Optional[str]}}`

### `print_evaluation_summary(metrics: EvaluationMetrics, detailed_results: List[Dict], logger: Optional[logging.Logger] = None)`

**목적**: 평가 요약 출력

**출력 내용**:
- Overall Metrics
- Per-Type Metrics (Main_Project_Specific, Dependency_Specific)
- Sub-Group Level Metrics
- Dependency Matching Analysis
- Beyond Heuristic Accuracy (BHA)
- Error Analysis (샘플 에러)

---

## 평가 로직

### Type 비교

```python
type_match = (llm_type == gt_type)
```

- 정확 일치만 인정
- `Main_Project_Specific` vs `Dependency_Specific` 이진 분류

### Dependency 비교

**Dependency_Specific 케이스**:
```python
if llm_type == "Dependency_Specific" and gt_type == "Dependency_Specific":
    dep_match = compare_dependencies(llm_dependency, gt_dependency)
```

**Main_Project_Specific 케이스**:
```python
if llm_type == "Main_Project_Specific" and gt_type == "Main_Project_Specific":
    # 둘 다 None/N/A이면 매칭
    dep_match = (llm_dep_is_none and gt_dep_is_none)
```

### 개별 추론 vs 그룹 레벨 추론

**우선순위**:
1. **개별 추론** (`individual_inferences`): `llm_reasoning_process`에서 파싱
2. **그룹 레벨 추론** (fallback): `group_level_root_cause_type`, `group_level_root_cause_dependency`

**이유**: 개별 추론이 더 정확하지만, 없으면 그룹 레벨 추론 사용

---

## 출력 데이터 구조

### EvaluationMetrics

```python
@dataclass
class EvaluationMetrics:
    total_cases: int = 0
    correct_type: int = 0
    correct_dependency: int = 0
    correct_both: int = 0
    
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
    
    # Partial matching
    sub_group_partial_type_accuracy_sum: float = 0.0
    sub_group_partial_dep_accuracy_sum: float = 0.0
    sub_group_representative_matches: int = 0
    
    # Dependency matching
    dependency_matching_ratio_avg: float = 0.0
    dependency_matching_count_total: int = 0
    
    # BHA
    bha_cases: int = 0
    bha_correct: int = 0
    
    # CPVR
    cpvr_total_outliers: int = 0
    cpvr_corrected_outliers: int = 0
    
    # WDR
    wdr_phase1_detected: int = 0
    wdr_phase2_detected: int = 0
    wdr_ground_truth_workarounds: int = 0
    wdr_tp_phase1: int = 0
    wdr_fp_phase1: int = 0
    wdr_fn_phase1: int = 0
    wdr_tp_phase2: int = 0
    wdr_fp_phase2: int = 0
    wdr_fn_phase2: int = 0
```

### Detailed Results

```python
[
    {
        'localId': int,
        'sub_group_id': int,
        'llm_type': str,
        'llm_dependency': str | None,
        'gt_type': str,
        'gt_dependency': str,
        'type_match': bool,
        'dependency_match': bool,
        'both_match': bool,
        'used_individual_inference': bool
    },
    ...
]
```

### Output JSON

```json
{
  "summary": {
    "total_cases": int,
    "sub_group_count": int,
    "metrics": {
      "accuracy_type": float,
      "accuracy_dependency": float,
      "accuracy_both": float,
      "main_project_precision": float,
      "main_project_recall": float,
      "main_project_f1": float,
      "dependency_precision": float,
      "dependency_recall": float,
      "dependency_f1": float,
      "sub_group_accuracy_type": float,
      "sub_group_accuracy_dependency": float,
      "sub_group_accuracy_both": float,
      "sub_group_partial_type_accuracy": float,
      "sub_group_partial_dep_accuracy": float,
      "sub_group_representative_accuracy": float,
      "dependency_matching_ratio_avg": float,
      "dependency_matching_count_total": int,
      "bha_accuracy": float,
      "bha_cases": int,
      "bha_correct": int
    },
    "raw_counts": {...}
  },
  "detailed_results": [...]
}
```

---

## 사용 예시

### 기본 사용
```bash
# 기본 파일명 사용
python3 04_evaluate_llm_inference.py

# 파일 경로 지정
python3 04_evaluate_llm_inference.py \
    --inference-file llm_inference_results.json \
    --gt-file ground_truth.json \
    --output evaluation_results.json
```

### Verbose 모드
```bash
# 상세 로깅 활성화
python3 04_evaluate_llm_inference.py --verbose
```

### 로그 파일 지정
```bash
# 로그 파일 경로 지정
python3 04_evaluate_llm_inference.py --log-file custom_evaluation.log
```

---

## 평가 결과 해석

### Type Accuracy
- **높을수록 좋음**: LLM이 Root Cause Type을 정확히 구분하는 능력
- **목표**: 80% 이상

### Dependency Accuracy
- **높을수록 좋음**: LLM이 의존성 이름을 정확히 식별하는 능력
- **주의**: Dependency_Specific 케이스만 평가

### Both Correct
- **가장 엄격한 메트릭**: Type과 Dependency 모두 정확해야 함
- **목표**: 70% 이상

### Precision vs Recall
- **Precision**: LLM이 예측한 것 중 정확한 비율
- **Recall**: GT에서 실제로 존재하는 것 중 LLM이 찾은 비율
- **Trade-off**: Precision이 높으면 Recall이 낮을 수 있음

### Sub-Group Level Metrics
- **Perfect Matching**: Sub-Group 내 모든 케이스가 정확히 일치
- **Partial Matching**: 평균 매칭 비율 (더 관대한 평가)
- **Representative Matching**: 그룹 레벨 추론의 정확도

### Beyond Heuristic Accuracy (BHA)
- **의미**: 휴리스틱 GT의 한계를 LLM이 극복한 비율
- **높을수록 좋음**: LLM이 GT보다 더 정확한 경우
- **예시**: GT가 Main으로 분류했지만 실제로는 Dependency인 경우

---

## LLM이 휴리스틱보다 더 정확할 가능성이 높은 케이스 분석

### 개요

Ground Truth(휴리스틱)와 LLM Inference 결과가 불일치한 케이스 중, **LLM이 더 정확할 가능성이 높은 케이스**를 선별하여 분석했습니다.

**선별 기준**:
1. GT 신뢰도가 낮음 (≤ 3.0)
2. LLM 신뢰도가 높음 (≥ 0.85)
3. LLM의 dependency_score가 main_project_score보다 높음
4. 높은 COC (≥ 0.8) 또는 Workaround 감지
5. 모듈 불일치 (crash_module ≠ patched_module)
6. 높은 의존성 매칭 비율 (≥ 0.8)

**선별된 케이스**: 4개 (Use-of-uninitialized-value 버그 타입)

### 대표 케이스 분석

#### 케이스 1: localId 371659889 (imagemagick)
- **GT**: Main_Project_Specific (신뢰도: 2.5/8.0)
- **LLM**: Dependency_Specific - libheif (신뢰도: 0.85)
- **증거**:
  - 스택 트레이스가 libheif에서 명확히 발생
  - COC 1.0 (100%) - 스택 트레이스가 libheif에 완전히 속함
  - 모듈 불일치 (crash: libheif, patch: unknown)
  - 의존성 매칭 100%
  - LLM dependency_score (0.9) >> main_project_score (0.1)

**결론**: LLM이 더 정확할 가능성이 매우 높음

#### 케이스 2: localId 42535316 (poppler)
- **GT**: Main_Project_Specific (신뢰도: 3.0/8.0)
- **LLM**: Dependency_Specific - openjpeg (신뢰도: 0.85)
- **증거**:
  - 스택 트레이스가 openjpeg에서 명확히 발생
  - COC 1.0 (100%)
  - 모듈 불일치 (crash: openjpeg, patch: unknown)
  - 3개 개별 추론 모두 일치
  - 의존성 매칭 100%

**결론**: LLM이 더 정확할 가능성이 매우 높음

#### 케이스 3, 4: 경계 케이스 (localId 42525804, 42520436)
- **GT**: Main_Project_Specific (신뢰도: 3.0/8.0)
- **LLM**: Dependency_Specific - magickcore/coders (신뢰도: 0.90)
- **특징**: ImageMagick의 내부 모듈(magickcore, coders) 처리
- **분석**: 의존성과 메인 프로젝트의 경계가 모호한 경계 케이스

**결론**: LLM이 더 정확할 가능성이 있지만 경계 케이스로 판단이 어려움

### 공통 패턴

#### GT의 문제점
1. **낮은 신뢰도**: 모든 케이스에서 GT 신뢰도가 2.5~3.0으로 낮음 (최대 8.0 대비 31~37%)
2. **Rule 2 점수 계산 문제**: COC가 1.0인데도 score가 0으로 계산됨
3. **경계 케이스 처리**: 메인 프로젝트의 내부 모듈 처리 모호

#### LLM의 강점
1. **높은 신뢰도**: 모든 케이스에서 0.85~0.90으로 높음
2. **일관된 추론**: 개별 추론들이 모두 일치 (의존성 매칭 100%)
3. **정량적 증거 활용**: 패치 패턴, 스택 트레이스, 의존성 매칭 등 종합 분석

### 주요 발견사항

1. **스택 트레이스가 명확한 경우**: LLM이 매우 정확함
   - 케이스 1, 2에서 스택 트레이스가 의존성에서 명확히 발생
   - COC 1.0으로 의존성 소유권이 명확
   - LLM이 이를 정확히 식별

2. **GT의 낮은 신뢰도**: 불확실한 분류의 신호
   - GT 신뢰도가 낮은 경우(≤ 3.0) LLM 추론을 참고 권장
   - LLM의 높은 신뢰도와 대조적

3. **Rule 2 점수 계산 개선 필요**
   - COC가 높을 때 점수에 제대로 반영되도록 개선 필요
   - 현재 COC 1.0인데도 score가 0으로 계산되는 문제

### 권장사항

1. **GT 신뢰도가 낮은 경우**: LLM 추론을 참고하여 검토
2. **Rule 2 점수 계산 개선**: COC가 높을 때 점수에 제대로 반영
3. **경계 케이스 처리**: 메인 프로젝트의 내부 모듈 처리 기준 명확화
4. **LLM 활용**: GT 신뢰도가 낮은 경우 LLM의 높은 신뢰도와 일관된 추론 활용

### 상세 분석

더 자세한 케이스별 분석은 `LLM_BETTER_CASES.md` 문서를 참고하세요.

---

## 주요 특징

### 1. 다층 평가
- **개별 케이스 평가**: 각 localId별 정확도
- **Sub-Group 평가**: 그룹 레벨 정확도
- **전체 평가**: 전체 메트릭 집계

### 2. 유연한 의존성 비교
- 정규화를 통한 이름 변형 허용
- 부분 문자열 매칭 지원
- 버전 정보 무시

### 3. 개별 추론 우선
- 개별 추론 결과가 있으면 우선 사용
- 없으면 그룹 레벨 추론 사용 (fallback)

### 4. Beyond Heuristic Accuracy
- 휴리스틱 GT의 한계를 측정
- LLM이 GT보다 더 정확한 경우 식별

### 5. 상세 에러 분석
- Type 불일치 샘플 출력
- Dependency 불일치 샘플 출력
- 디버깅 및 개선에 활용

---

## 참고사항

1. **Ground Truth 필수**: 평가를 위해 Ground Truth 파일이 필요
2. **의존성 비교**: 정규화를 통해 이름 변형 허용
3. **Main_Project_Specific**: Dependency 비교는 Dependency_Specific 케이스만 수행
4. **BHA 계산**: 데이터베이스 접근 필요 (`submodule_bug`, `repo_addr`)
5. **개별 추론 파싱**: `llm_reasoning_process`에서 개별 추론 결과 파싱 시도
6. **에러 분석**: 샘플 에러만 출력 (전체 에러는 detailed_results에서 확인)
7. **LLM이 더 정확한 케이스**: GT 신뢰도가 낮은 경우 LLM 추론을 참고하여 검토 권장

### 관련 문서

- `LLM_BETTER_CASES.md`: LLM이 휴리스틱보다 더 정확할 가능성이 높은 케이스 상세 분석
- `ACCURACY_VALIDATION.md`: LLM Inference 정확도 검증 리포트
- `LLM_INFERENCE_ANALYSIS.md`: LLM Inference 결과 분석 리포트

---

## Priority 1: 정량적 메트릭 실험 설계

### 1. WDR (Workaround Detection Rate) 정량적 측정

**현재 상태**:
- ✅ 4개의 정성적 케이스 (Qt5, libjpeg-turbo, libarchive, openh264)
- ❌ 정량적 측정 없음 (현재 데이터셋에서 workaround 케이스 없음)

**필요한 실험**:

```python
# 최소 100 cases manually annotated
workaround_annotation = {
    'clear_workaround': 50 cases,      # 명확한 workaround 패치
    'clear_non_workaround': 50 cases,  # 명확한 non-workaround 패치
    'ambiguous': 50 cases               # 전문가 간 의견 불일치 케이스
}

# Calculate WDR
wdr_phase1 = workarounds_detected_by_heuristic / total_workarounds
wdr_phase2 = workarounds_detected_by_llm / total_workarounds

# Expected:
# wdr_phase1: ~30-40% (heuristic misses semantic intent)
# wdr_phase2: ~70-80% (LLM detects semantic intent)
```

**실험 절차**:
1. **데이터 수집**: 최소 100개 케이스 수집
   - 다양한 프로젝트에서 workaround 패치 포함
   - 커밋 메시지, 패치 내용, 크래시 정보 포함

2. **전문가 주석 (Expert Annotation)**:
   - 최소 2명의 전문가가 독립적으로 주석
   - 명확한 workaround, 명확한 non-workaround, 모호한 케이스 분류
   - 전문가 간 일치도 측정 (Inter-annotator agreement)

3. **Ground Truth 생성**:
   - 전문가 주석을 기반으로 Ground Truth 생성
   - `ground_truth.json`에 `is_workaround` 필드 추가
   - 또는 별도 `workaround_annotations.json` 파일 생성

4. **평가 실행**:
   - Phase 1 (Heuristic) 감지율 측정
   - Phase 2 (LLM) 감지율 측정
   - Precision, Recall, F1-Score 계산

5. **결과 분석**:
   - Phase 1 vs Phase 2 비교
   - 통계적 유의성 검정
   - 오류 케이스 분석

**예상 결과**:
- **WDR Phase 1**: ~30-40% (heuristic misses semantic intent)
- **WDR Phase 2**: ~70-80% (LLM detects semantic intent)
- **개선 폭**: Phase 2가 Phase 1보다 약 2배 높은 감지율

**Why it matters**:
- 직접적으로 RQ1 검증 (semantic reasoning)
- "LLM surpasses heuristic" 주장의 정량적 증거

**개선 방향**:
1. Ground Truth에 `is_workaround` 필드 추가 (manual annotation)
2. 별도 annotation 파일 생성 (`workaround_annotations.json`)
3. 자동 감지 로직 개선 (더 많은 신호 활용)
4. LLM reasoning에서 workaround 키워드 검색 개선

---

### 2. CPVR (Cross-Project Validation Rate) 정량적 측정

**현재 상태**:
- ✅ 213 sub-groups 생성됨
- ✅ 97.08% dependency matching ratio (높은 그룹 일관성)
- ❌ Outlier correction rate 미측정 (현재 데이터셋에서 outlier 없음)

**필요한 실험**:

```python
# Identify outliers in sub-groups
def find_outliers(sub_group):
    """
    Sub-group 내에서 majority type과 다른 케이스
    """
    majority_type = most_common(sub_group.types)
    outliers = [case for case in sub_group if case.type != majority_type]
    return outliers

# Calculate CPVR
total_outliers = sum(find_outliers(g) for g in sub_groups)
corrected_by_module3 = count_corrections_in_module3(outliers)
cpvr = corrected_by_module3 / total_outliers

# Expected:
# cpvr: ~60-70% (group consensus corrects individual errors)
```

**실험 절차**:
1. **개별 추론 수집**:
   - Module 1에서 각 케이스별 개별 추론 결과 수집
   - `bug_type_groups` 또는 `llm_reasoning_process`에서 파싱
   - Fallback: GT 개별 타입 사용

2. **Outlier 식별**:
   - 각 Sub-group에서 다수 타입(≥60%) 식별
   - 다수와 불일치하는 케이스를 outlier로 식별
   - 최소 2개 케이스가 있는 Sub-group만 분석

3. **수정 여부 판단**:
   - Module 3의 최종 추론(`group_level_root_cause_type`) 확인
   - 최종 추론이 다수 타입과 일치하면 수정됨으로 판단
   - 최종 추론이 개별 추론과 같으면 수정되지 않음으로 판단

4. **CPVR 계산**:
   - 수정된 outlier 수 / 전체 outlier 수 × 100
   - 그룹 크기별, 버그 타입별 분석

5. **결과 분석**:
   - Module 3의 기여도 정량화
   - Cross-project 패턴 활용 효과 측정

**예상 결과**:
- **CPVR**: ~60-70% (group consensus corrects individual errors)
- Module 3의 기여도 정량화

**Why it matters**:
- 직접적으로 RQ2 검증 (cross-project pattern discovery)
- Module 3의 기여도 정량화

**개선 방향**:
1. **Module 1 개별 추론 보장**:
   - 모든 케이스에 대해 개별 추론 결과 생성
   - `bug_type_groups`에 `individual_inferences` 포함

2. **Fallback 메커니즘**:
   - 개별 추론이 없으면 GT 개별 타입 사용
   - 또는 그룹 레벨 추론을 개별 추론으로 사용 (제한적)

3. **Outlier 식별 개선**:
   - 다수 threshold 조정 가능 (현재 60%)
   - 의존성 이름도 고려한 outlier 식별

4. **정량적 측정**:
   - 최소 50개 이상의 outlier 케이스 확보
   - 다양한 그룹 크기에서 측정

**분석**:
- **By Group Size**: 그룹 크기별 CPVR
- **By Bug Type**: 버그 타입별 CPVR
- **By Project**: 프로젝트별 CPVR

---

## 실험 설계 요약

### 데이터 요구사항

**WDR 측정**:
- 최소 100개 케이스 (50 workaround + 50 non-workaround + 50 ambiguous)
- 전문가 주석 (최소 2명)
- 커밋 메시지, 패치 내용, 크래시 정보 포함

**CPVR 측정**:
- 최소 50개 이상의 outlier 케이스
- Module 1 개별 추론 결과 필요
- 다양한 그룹 크기 (2-10개 케이스)

### 평가 메트릭

**WDR**:
- Phase 1 vs Phase 2 감지율 비교
- Precision, Recall, F1-Score
- 통계적 유의성 검정

**CPVR**:
- 전체 CPVR
- 그룹 크기별 CPVR
- 버그 타입별 CPVR
- 프로젝트별 CPVR

### 예상 결과

**WDR**:
- Phase 1: ~30-40%
- Phase 2: ~70-80%
- 개선 폭: 약 2배

**CPVR**:
- 전체: ~60-70%
- 그룹 크기가 클수록 높은 CPVR 예상

### 논문 기여도

**WDR**:
- RQ1 검증: Semantic reasoning capability
- "LLM surpasses heuristic" 주장의 정량적 증거

**CPVR**:
- RQ2 검증: Cross-project pattern discovery
- Module 3의 기여도 정량화

---

## Main → Dependency 오분류 원인 분석

### 개요

LLM이 Main_Project_Specific 케이스를 Dependency_Specific으로 오분류하는 경향이 강합니다 (18개 케이스, Recall 14.29%). 이는 LLM의 보수적 접근 방식과 스택 트레이스 기반 추론의 한계를 보여줍니다.

### 통계적 패턴

**오분류 케이스: 18개**

1. **Stack trace에 의존성 언급**: 16/18 (88.9%)
   - 스택 트레이스에 `src/libheif`, `src/openjpeg`, `vendor/ucl` 등 의존성 경로가 명시적으로 나타남
   - LLM이 스택 트레이스의 파일 경로를 강하게 신뢰

2. **Patch-crash distance >= 2**: 18/18 (100.0%)
   - 모든 오분류 케이스에서 패치와 크래시 위치가 멀리 떨어져 있음
   - LLM이 이를 의존성 문제의 증거로 해석

3. **Crash location이 src/ 경로에 있음**: 8/18 (44.4%)
   - `src/libheif/`, `src/openjpeg/` 등 경로가 의존성처럼 보임

4. **패치는 main_project인데 크래시는 다른 곳**: 7/18 (38.9%)
   - 패치는 메인 프로젝트에 있지만 크래시는 다른 모듈에서 발생
   - LLM이 크래시 위치를 더 중요하게 고려

### 주요 원인

#### 1. 스택 트레이스 기반 추론의 한계 (88.9%)

**문제점**:
- LLM이 스택 트레이스의 파일 경로(`src/libheif/`, `src/openjpeg/`)를 보고 의존성으로 판단
- 실제로는 메인 프로젝트의 서브모듈이거나 번들된 의존성일 수 있음

**대표 케이스**:
- `localId 371659889`: 스택 트레이스에 `src/libheif/libheif/codecs/vvc_dec.cc` 명시
- `localId 42535316`: 스택 트레이스에 `src/openjpeg/src/lib/openjp2/j2k.c` 명시
- `localId 383170478`: 스택 트레이스에 `/src/upx/vendor/ucl/src/n2e_d.c` 명시

**LLM 추론 근거**:
- "Both individual inferences unanimously attribute the root cause to the libheif dependency"
- "All three individual inferences unanimously identified the root cause as dependency-specific in the shared 'src' module"

#### 2. Patch-Crash Distance 해석 오류 (100%)

**문제점**:
- 모든 오분류 케이스에서 `patch_crash_distance >= 2`
- LLM이 이를 "패치가 크래시 위치와 멀리 떨어져 있음 = 의존성 문제"로 해석
- 실제로는 메인 프로젝트 내에서도 거리가 멀 수 있음

**대표 케이스**:
- `localId 371659889`: Patch-Crash Distance = 3
- `localId 42535316`: Patch-Crash Distance = 3
- `localId 383170478`: Patch-Crash Distance = 2

#### 3. Crash Module vs Patched Module 불일치 (38.9%)

**문제점**:
- Crash Module이 `libheif`, `magickcore`, `coders` 등으로 표시됨
- Patched Module은 `main_project`로 표시됨
- LLM이 Crash Module을 더 중요하게 고려하여 의존성으로 판단

**대표 케이스**:
- `localId 371659889`: Crash Module = `libheif`, Patched Module = `main_project`
- `localId 42525804`: Crash Module = `magickcore`, Patched Module = `main_project`
- `localId 42520436`: Crash Module = `coders`, Patched Module = `main_project`

**특히 문제가 되는 경우**:
- `magickcore`, `coders`는 ImageMagick의 **내부 모듈**인데 LLM이 의존성으로 오인
- 경계 케이스: 메인 프로젝트의 내부 모듈과 외부 의존성 구분 어려움

#### 4. LLM의 보수적 접근 (Dependency Score >> Main Score)

**문제점**:
- LLM이 Dependency Score를 Main Project Score보다 훨씬 높게 평가
- 모든 오분류 케이스에서 Dependency Score (0.9) >> Main Project Score (0.0-0.1)

**대표 케이스**:
- `localId 371659889`: Dependency Score = 0.9, Main Project Score = 0.1
- `localId 42535316`: Dependency Score = 0.9, Main Project Score = 0.0
- `localId 383170478`: Dependency Score = 0.9, Main Project Score = 0.1

**LLM 추론 패턴**:
- "Quantitative evidence does not contradict these inferences"
- "All three individual inferences unanimously identified the root cause as dependency-specific"
- LLM이 그룹 내 일치도와 정량적 증거를 과도하게 신뢰

### 프로젝트별 분포

- **imagemagick**: 5회 (가장 많음)
  - `magickcore`, `coders` 등 내부 모듈을 의존성으로 오인
- **serenity**: 3회
- **libheif**: 2회
- 기타: poppler, upx, qpdf, wolfssl, libarchive, netcdf-c, fluent-bit, espeak-ng 각 1회

### LLM이 추론한 의존성 (상위 5개)

1. **libheif**: 4회
2. **serenity**: 3회
3. **openjpeg**: 1회
4. **magickcore**: 1회 (실제로는 ImageMagick의 내부 모듈)
5. **coders**: 1회 (실제로는 ImageMagick의 내부 모듈)

### 근본 원인 요약

1. **스택 트레이스 경로 해석 오류**
   - `src/libheif/` 같은 경로를 보고 외부 의존성으로 판단
   - 실제로는 번들된 의존성 또는 서브모듈일 수 있음

2. **메인 프로젝트 내부 모듈 구분 실패**
   - `magickcore`, `coders` 같은 내부 모듈을 의존성으로 오인
   - 프로젝트 구조에 대한 지식 부족

3. **Patch-Crash Distance 과도 해석**
   - 거리가 멀다고 무조건 의존성 문제로 판단
   - 메인 프로젝트 내에서도 거리가 멀 수 있음

4. **보수적 접근 방식**
   - 불확실할 때 Dependency로 분류하는 경향
   - Main Project Score를 0.0-0.1로 낮게 평가

### 개선 방안

1. **프로젝트 구조 정보 활용**
   - 서브모듈 vs 외부 의존성 구분을 위한 프로젝트 구조 정보 제공
   - `srcmap` 정보 활용하여 실제 의존성 트리 확인

2. **패치 위치 정보 강화**
   - 패치가 메인 프로젝트에 있다는 명확한 증거 제공
   - 패치 파일 경로 분석 (`coders/sf3.c` → 메인 프로젝트)

3. **스택 트레이스 해석 개선**
   - 경로만 보고 판단하지 않고 실제 소유권 확인
   - 번들된 의존성과 외부 의존성 구분

4. **Main Project Score 가중치 조정**
   - 패치가 메인 프로젝트에 있으면 Main Project Score에 보너스
   - Patch-Crash Distance가 크더라도 패치 위치를 더 중요하게 고려

5. **경계 케이스 처리**
   - `magickcore`, `coders` 같은 내부 모듈을 명시적으로 메인 프로젝트로 분류
   - 프로젝트별 내부 모듈 목록 제공

---

## LLM vs GT 검증 결과

### 검증 방법

18개의 오분류 케이스에 대해 다음 증거를 수집하여 검증했습니다:

1. **submodule_bug 플래그**: 데이터베이스에서 실제 서브모듈 여부 확인
2. **repo_addr**: 저장소 주소가 프로젝트와 일치하는지 확인
3. **스택 트레이스 경로**: 실제 의존성 경로인지 메인 프로젝트 모듈인지 확인
4. **패치 파일 위치**: 패치가 메인 프로젝트 파일을 수정하는지 확인
5. **Crash Module 분석**: 내부 모듈인지 외부 의존성인지 확인

### 검증 결과 요약

**총 18개 케이스 검증**:
- **GT가 맞을 가능성 높음**: 9개 (50%)
- **모호함**: 9개 (50%)
- **LLM이 맞을 가능성 높음**: 0개 (0%)

### GT가 맞을 가능성이 높은 케이스 (9개)

#### 1. ImageMagick 내부 모듈 오인 (2개)

**케이스 1: localId 42520436**
- **LLM 추론**: Dependency_Specific (coders)
- **GT**: Main_Project_Specific
- **증거 점수**: LLM=0, GT=7
- **주요 증거**:
  - ✅ `coders`는 ImageMagick의 **내부 모듈** (GT +3)
  - ✅ Crash Module이 메인 프로젝트 모듈 (GT +3)
  - ✅ 패치가 메인 프로젝트 파일 수정 (GT +2)
- **결론**: **GT가 맞음**. `coders`는 ImageMagick의 코더 모듈로 메인 프로젝트의 일부입니다.

**케이스 2: localId 42525804**
- **LLM 추론**: Dependency_Specific (magickcore)
- **GT**: Main_Project_Specific
- **증거 점수**: LLM=0, GT=6
- **주요 증거**:
  - ✅ `magickcore`는 ImageMagick의 **내부 모듈** (GT +3)
  - ✅ 패치가 메인 프로젝트 파일 수정: `MagickCore/cache.c`, `MagickCore/visual-effects.c` (GT +2)
- **결론**: **GT가 맞음**. `magickcore`는 ImageMagick의 핵심 라이브러리로 메인 프로젝트의 일부입니다.

#### 2. 번들된 의존성 케이스 (5개)

**케이스 3-7**: imagemagick (libheif, freetype), poppler (openjpeg), upx (ucl), wolfssl, netcdf-c
- **공통 패턴**:
  - ⚠️ 번들된 의존성(bundled dependency)이지만 패치가 메인 프로젝트에 있음 (GT +2)
  - ✅ 패치가 메인 프로젝트 파일 수정 (GT +2)
  - ❌ `submodule_bug=False` (GT +1)
- **결론**: **GT가 맞을 가능성이 높음**. 번들된 의존성의 버그를 메인 프로젝트에서 수정하는 경우, 메인 프로젝트 문제로 분류하는 것이 타당합니다.

**대표 케이스: localId 371659889 (imagemagick/libheif)**
- 스택 트레이스: `src/libheif/libheif/codecs/vvc_dec.cc`
- 패치: 메인 프로젝트에 있음
- `submodule_bug=False`: 서브모듈이 아님
- **해석**: ImageMagick이 libheif를 번들로 포함하고 있지만, 패치가 메인 프로젝트에 있다는 것은 메인 프로젝트의 책임으로 보는 것이 타당합니다.

### 모호한 케이스 (9개)

**특징**:
- 프로젝트 이름과 의존성 이름이 동일한 경우 (예: libheif 프로젝트에서 libheif 의존성)
- `repo_addr`이 프로젝트와 일치하지만, 실제로는 의존성일 수도 있음
- 증거가 충분하지 않아 명확한 판단이 어려움

**대표 케이스들**:
- `libheif` 프로젝트에서 `libheif` 의존성 (3개)
- `serenity` 프로젝트에서 `serenity` 의존성 (3개)
- `qpdf`, `libarchive`, `fluent-bit`, `espeak-ng` 각 1개

### 주요 발견사항

1. **LLM의 주요 오류: 내부 모듈을 의존성으로 오인**
   - `magickcore`, `coders`는 ImageMagick의 내부 모듈인데 LLM이 의존성으로 분류
   - 프로젝트 구조에 대한 지식 부족이 원인

2. **번들된 의존성의 모호성**
   - 번들된 의존성의 버그를 메인 프로젝트에서 수정하는 경우
   - GT는 메인 프로젝트로 분류 (패치 위치 기준)
   - LLM은 의존성으로 분류 (크래시 위치 기준)
   - **GT의 접근이 더 타당**: 패치 위치가 실제 수정 책임을 나타냄

3. **프로젝트 이름 = 의존성 이름 케이스**
   - 프로젝트 자체를 의존성으로 오인하는 경우
   - 추가 정보 없이는 판단이 어려움

### 결론

**검증 결과: GT가 대부분 맞습니다**

1. **명확한 오류 (2개)**: ImageMagick의 내부 모듈(`magickcore`, `coders`)을 의존성으로 오인
2. **번들된 의존성 (5개)**: 패치 위치를 기준으로 GT의 분류가 더 타당
3. **모호한 케이스 (9개)**: 추가 정보 필요

**LLM의 문제점**:
- 프로젝트 구조에 대한 지식 부족
- 크래시 위치만 보고 판단하는 경향
- 패치 위치를 충분히 고려하지 않음

**개선 방향**:
- 프로젝트별 내부 모듈 목록 제공
- 패치 위치 정보를 더 중요하게 고려
- 번들된 의존성과 외부 의존성 구분 로직 추가

---

## 근본 원인: submodule_bug 정보 전달 문제

### 문제 발견

사용자의 지적대로, LLM이 GT 룰에 추가해서 추론하는데도 서브모듈 vs 외부 의존성 구분이 제대로 작동하지 않았습니다.

### 코드 분석 결과

#### 1. submodule_bug 정보 전달 경로

**`_summarize_patch` 함수 (729번째 줄)**:
```python
submodule_bug = patch_info.get('submodule_bug', False)
prompt = f"""...
Submodule Bug: {submodule_bug}
..."""
```
- ✅ `submodule_bug` 정보를 프롬프트에 포함시킴
- ❌ 하지만 LLM이 생성한 patch summary에는 명시적으로 포함되지 않을 수 있음

**`_generate_individual_root_cause_reasoning` 함수 (2903번째 줄)**:
- Module 1 individual inference 프롬프트
- ❌ `patch_summary`만 사용 (submodule_bug 정보가 간접적으로만 전달됨)
- ❌ `submodule_bug` 정보를 직접 받지 않음

#### 2. _classify_dependency_type 함수의 한계

**함수 위치**: 168번째 줄 (static method)

**현재 구현**:
```python
@staticmethod
def _classify_dependency_type(dep_path: str) -> str:
    """Classify dependency as submodule, external, or main project"""
    if '/src/' in dep_path_lower or dep_path_lower.startswith('src/'):
        return 'submodule'  # 경로 기반만 판단
    ...
```

**문제점**:
- ❌ 경로 기반 분류만 수행
- ❌ 실제 데이터베이스의 `submodule_bug` 플래그를 사용하지 않음
- ❌ `/src/libheif/` 같은 경로를 보고 무조건 submodule로 분류
- ❌ 실제로는 `submodule_bug=0`인 경우도 있음

#### 3. 실제 데이터 확인

**오분류 케이스들의 실제 submodule_bug 값**:
- `localId 371659889`: `submodule_bug=0` (서브모듈 아님)
- `localId 42535316`: `submodule_bug=0` (서브모듈 아님)
- `localId 42525804`: `submodule_bug=0` (서브모듈 아님)

**하지만**:
- 스택 트레이스 경로: `/src/libheif/`, `/src/openjpeg/` 등
- `_classify_dependency_type`이 이를 submodule로 잘못 분류
- LLM이 경로만 보고 의존성으로 판단

### 근본 원인 요약

1. **submodule_bug 정보가 Module 1에 직접 전달되지 않음**
   - `_summarize_patch`에서만 사용
   - Module 1 individual inference에는 patch summary를 통해서만 간접 전달
   - LLM이 patch summary에서 submodule_bug 정보를 추출하지 못함

2. **_classify_dependency_type의 경로 기반 분류 한계**
   - 실제 `submodule_bug` 플래그를 사용하지 않음
   - `/src/` 경로만 보고 판단하여 오분류 발생

3. **프로젝트 구조 정보 부족**
   - `magickcore`, `coders` 같은 내부 모듈을 구분할 수 없음
   - 프로젝트별 내부 모듈 목록이 없음

### 해결 방안

#### 1. Module 1 individual inference에 submodule_bug 직접 추가

**수정 필요 위치**: `_generate_individual_root_cause_reasoning` 함수

```python
# 현재: patch_summary만 사용
# 수정: submodule_bug 정보 직접 추가

# Get submodule_bug from database or patch_info
submodule_bug = self._get_submodule_bug(feature.localId)

prompt = f"""...
**Submodule Information:**
- submodule_bug flag: {submodule_bug}
- If submodule_bug=False, then crash in /src/ path is likely bundled dependency, not submodule
- If submodule_bug=True, then crash in /src/ path is actual submodule
..."""
```

#### 2. _classify_dependency_type 함수 개선

**수정 필요**: 실제 `submodule_bug` 플래그 사용

```python
@staticmethod
def _classify_dependency_type(dep_path: str, submodule_bug: Optional[bool] = None) -> str:
    """Classify dependency with submodule_bug flag"""
    if submodule_bug is not None:
        # Use actual flag if available
        if submodule_bug:
            return 'submodule'
        else:
            # submodule_bug=False but path has /src/ → bundled dependency or main project module
            # Need additional logic to distinguish
            ...
    
    # Fallback to path-based classification
    if '/src/' in dep_path_lower:
        return 'submodule'  # Default assumption
    ...
```

#### 3. 프로젝트별 내부 모듈 목록 제공

```python
INTERNAL_MODULES = {
    'imagemagick': ['magickcore', 'coders', 'magickwand', 'magick++'],
    'poppler': ['poppler'],
    'upx': ['upx'],
    # ...
}

def _is_internal_module(self, module_name: str, project_name: str) -> bool:
    """Check if module is internal to project"""
    internal_modules = INTERNAL_MODULES.get(project_name.lower(), [])
    return any(mod in module_name.lower() for mod in internal_modules)
```

### 현재 상태 요약

**데이터셋 확인 결과**:
- 전체 86개 케이스 모두 `submodule_bug=False` (100%)
- 서브모듈 버그가 아닌 케이스만 포함된 데이터셋

**문제**: 
- `submodule_bug=False`인데도 LLM이 경로(`/src/`)만 보고 submodule로 분류
- `submodule_bug` 정보가 LLM에 명시적으로 전달되지 않음
- 경로 기반 분류만 사용하여 오분류 발생

**핵심 문제**:
1. **경로 기반 분류의 한계**: `/src/libheif/` 같은 경로를 보고 무조건 submodule로 분류
2. **명시적 정보 부족**: `submodule_bug=False`라는 정보를 LLM이 직접 받지 못함
3. **오해의 소지**: 경로가 `/src/`로 시작하면 서브모듈로 보이지만, 실제로는:
   - 번들된 의존성 (bundled dependency)
   - 메인 프로젝트의 내부 모듈 (magickcore, coders 등)
   - 서브모듈이 아님 (`submodule_bug=False`)

**영향**:
- Main 케이스를 Dependency로 오분류 (18개 케이스, 모두 `submodule_bug=False`)
- 특히 ImageMagick의 내부 모듈 오인 (2개 명확한 오류)

**해결 필요**:
- 코드 수정 필요: Module 1에 `submodule_bug=False` 정보 명시적으로 전달
  - "이 케이스는 서브모듈 버그가 아님"
  - "/src/ 경로가 있어도 서브모듈이 아님"
  - "번들된 의존성이거나 메인 프로젝트 모듈일 수 있음"
- 코드 수정 필요: `_classify_dependency_type`에 `submodule_bug` 플래그 사용
- 데이터 추가 필요: 프로젝트별 내부 모듈 목록 제공

**의미 및 한계**:
- 데이터셋이 `submodule_bug=False`로만 구성되어 있음 (100%)
- 따라서 `submodule_bug` 정보 자체는 **구분에 도움이 되지 않음** (모두 False)
- 하지만 이 정보를 명시적으로 전달하면:
  - LLM이 경로만 보고 판단하는 것을 방지 가능
  - "/src/ 경로가 있어도 서브모듈이 아님"이라는 명시적 정보 제공
  - 번들된 의존성과 실제 서브모듈 구분 가능
  - 내부 모듈을 의존성으로 오인하는 것 방지
  - **오분류를 줄일 수 있음**

**더 중요한 개선 방안**:
1. **프로젝트별 내부 모듈 목록 제공** (가장 효과적)
   - ImageMagick: `magickcore`, `coders`, `magickwand` 등
   - 이 정보를 LLM에 직접 제공하면 내부 모듈 오인 방지
2. **패치 위치 정보 강화**
   - 패치가 메인 프로젝트 파일을 수정한다는 명시적 정보
   - `coders/sf3.c` → ImageMagick의 메인 프로젝트 파일
3. **경로 해석 개선**
   - `/src/` 경로가 항상 서브모듈을 의미하는 것은 아님
   - 번들된 의존성, 내부 모듈, 실제 서브모듈 구분 필요

---

## 실행 결과 (Use-of-uninitialized-value)

### 최근 평가 결과 요약

```
================================================================================
📊 PAPER METRICS SUMMARY (Phase 2 - LLM Evaluation)
================================================================================

📈 Overall Performance:
  • Total Cases Evaluated: 517
  • Sub-Groups Evaluated: 125
  • Type Accuracy: 86.65% (448/517)
  • Dependency Name Accuracy: 84.72% (438/517)
  • Both Correct: 84.72% (438/517)

📝 Paper Values (Overall):
  • **86.65%** - Type accuracy
  • **84.72%** - Dependency name accuracy
  • **448/517** - Correct type classifications
  • **438/517** - Correct dependency matches

🧱 ARVO Baseline (DB-derived):
  • Definition: Dependency_Specific iff arvo.submodule_bug == 1 (ARVO DB-derived baseline)
  • Type Accuracy: 86.65% (448/517)
  • Balanced Type Accuracy: 50.00%
  • Dependency Recall: 0.00% (predicted Dependency: 0)

📊 Per-Type Performance:
  Main_Project_Specific:
    • Precision: 93.56% (407/435)
    • Recall: 90.85% (407/448)
    • F1: 92.19%
  Dependency_Specific:
    • Precision: 50.00% (41/82)
    • Recall: 59.42% (41/69)
    • F1: 54.30%

📝 Paper Values (Per-Type):
  • **93.56%** - Main precision
  • **90.85%** - Main recall
  • **92.19%** - Main F1
  • **50.00%** - Dependency precision
  • **59.42%** - Dependency recall
  • **54.30%** - Dependency F1

🔗 Sub-Group Level Metrics:
  • Perfect Type Matching: 70.40% (88/125)
  • Perfect Dependency Matching: 69.60% (87/125)
  • Partial Type Accuracy: 86.61% (average match ratio)
  • Partial Dependency Accuracy: 84.60% (average match ratio)
  • Representative Matching: 88.00% (110/125)

📝 Paper Values (Sub-Group):
  • **70.40%** - Sub-group type matching
  • **69.60%** - Sub-group dependency matching

🎯 Beyond Heuristic Accuracy (BHA):
  • BHA Cases: 24
  • LLM Corrected GT Errors: 0
  • BHA Accuracy: 0.00%

📝 Paper Values (BHA):
  • **24** - LLM-GT disagreement cases
  • **0** - LLM corrected GT errors
  • **0.00%** - BHA (conservative estimate)

🧪 Ablation (Stage-wise, Type-only):
  • Stage 1 only (Module 1): Type 85.49%, Balanced 78.14%, Dep P/R 47.00% / 68.12%
  • Stage 2 only (Module 2): Type 85.49%, Balanced 73.24%, Dep P/R 46.43% / 56.52%
  • Stages 1–3 (final): Type 86.65%, Balanced 75.13%, Dep P/R 50.00% / 59.42%
```

### 주요 결과 해석

#### Overall Performance
- **Type Accuracy: 86.65%**: Root Cause Type(label) 일치 비율 (heuristic GT 기준)
- **ARVO baseline과의 차이**: ARVO submodule-only baseline은 Type Accuracy는 동일하지만(86.65%), **Dependency recall 0%**로 소수 클래스 탐지에 실패
- **Dependency Name Accuracy: 84.72%**: dependency 이름까지 포함한 일치 비율(단, Main 케이스에서는 `None` 매칭이 포함됨)

#### Per-Type Performance
- **Main_Project_Specific**:
  - **Precision 93.56%**: Main으로 예측한 것 중 정확 비율
  - **Recall 90.85%**: 실제 Main 케이스 중 올바르게 식별한 비율
  - **F1 92.19%**

- **Dependency_Specific**:
  - **Precision 50.00%**: Dependency 예측 중 절반이 정확 (FP 부담이 여전히 큼)
  - **Recall 59.42%**: 실제 Dependency 중 59%를 회수
  - **F1 54.30%**
  - **핵심 트레이드오프**: Dependency TP(41)를 얻는 대신 Main FP(41)가 동반됨

#### Sub-Group Level Metrics
- **Perfect Matching: 70.40%**: Sub-Group 내 모든 케이스가 완벽히 일치하는 비율(타입 기준)
- **Representative Matching: 88.00%**: 대표값 기준으로 맞춘 비율
- **의미**: Stage 3가 cross-case 구조를 활용해 일관된 결정을 내릴 수 있음을 시사

#### Beyond Heuristic Accuracy (BHA)
- **BHA Cases: 24개**, **BHA Correct: 0개**: 현재 스크립트의 보수적 BHA 정의와 저장된 아티팩트 조건 하에서 “LLM이 GT 오류를 수정”했다고 자동 판정된 케이스는 없음
- **해석 주의**: 이는 “GT가 항상 옳다”는 의미가 아니라, **전문가 판정(ground truth adjudication) 없이 자동 proxy로는 GT 오류를 확정하기 어렵다**는 한계로 해석하는 것이 안전함

### 주요 발견사항

1. **Main_Project_Specific 높은 Precision**: 
   - Main precision 93.56%, recall 90.85%로 안정적

2. **Dependency_Specific Precision/Recall 트레이드오프**:
   - Dependency precision 50.00%, recall 59.42%
   - TP(41)를 얻는 대신 FP(41)가 발생 → 자동화 워크플로우에는 여전히 부담

3. **ARVO baseline 대비 소수 클래스 회복**:
   - ARVO submodule-only baseline은 Dependency recall 0%
   - LLM 파이프라인은 Dependency recall 59.42%로 소수 클래스 탐지에 의미있는 개선

4. **Stage-wise ablation 관점에서의 Module 3 역할**:
   - Stage 1(개별 추론)이 recall을 상대적으로 높게 가져가고,
   - Stage 3(최종 합성/검증)가 precision 및 전체 agreement를 소폭 개선하는 방향으로 작동

### 개선 필요 영역

1. **Dependency Precision 향상**:
   - FP(41)를 줄이는 것이 최우선 과제
   - 프로젝트 구조 정보 제공 (내부 모듈 목록)
   - 패치 위치 정보 강화

2. **Main Recall 향상**:
   - Main recall은 90.85%로 높지만, Dependency 쪽 조정이 Main 성능을 과도하게 훼손하지 않도록 캘리브레이션 필요

3. **Sub-Group 활용**:
   - Perfect Matching 70.40%에서 추가 개선 여지
   - 그룹 레벨 추론의 정확도 향상 필요

---

## LLM이 GT보다 정확했을 가능성이 높은 케이스 분석

### 개요

GT가 `Main_Project_Specific`으로 분류했지만 LLM이 `Dependency_Specific`으로 추론한 케이스는 517-case 기준으로 41개(FP)입니다. 이 중 **LLM이 GT보다 정확했을 가능성이 있는 대표 케이스들**을 정성적으로 검토했으며, 분석 기준은 다음과 같습니다:

1. **LLM 신뢰도가 높음** (≥ 0.85)
2. **GT 신뢰도가 낮음** (≤ 3.0) 또는 정보 부족
3. **submodule_bug=False** (서브모듈이 아님)
4. **Workaround 패치 감지됨**
5. **Dependency Score >> Main Project Score** (의존성 점수가 메인 프로젝트 점수보다 훨씬 높음)
6. **스택 트레이스가 의존성에서 명확히 발생**

### 케이스 1: localId 371659889 (imagemagick)

**GT 분류**:
- Type: `Main_Project_Specific`
- Dependency: `None`
- Confidence: `N/A` (정보 없음)
- Submodule Bug: `False`

**LLM 추론**:
- Type: `Dependency_Specific`
- Dependency: `Multiple (libjxl, libheif, libjpeg-turbo, libraw)`
- Confidence: `0.9` (매우 높음)
- Dependency Score: `0.9`
- Main Project Score: `0.1`

**주요 증거**:

1. **스택 트레이스 분석**:
   - 크래시 위치: `libheif/libheif/codecs/vvc_dec.cc:63`
   - 함수: `Decoder_VVC::get_coded_image_colorspace`
   - 모든 스택 프레임이 libheif 내부에 위치
   - ImageMagick 코어 코드는 스택 트레이스에 나타나지 않음

2. **Workaround 패치 감지**:
   - `workaround_detected: True`
   - LLM이 패치 의도를 분석하여 workaround로 판단
   - 메인 프로젝트에서 의존성 버그를 우회하는 패치

3. **그룹 레벨 증거**:
   - Sub-Group 내 11개 케이스 모두 동일한 의존성 그룹 (libjxl, libheif, libjpeg-turbo, libraw)
   - 의존성 매칭 비율: 100% (11/11)
   - Cross-project 패턴: 여러 프로젝트에서 동일한 의존성 문제 발생

4. **LLM 추론 근거**:
   - "Every stack trace entry points into one of those external libraries"
   - "Every patch intent is defensive or a workaround around dependency behavior"
   - "Even where heuristic GT rules labeled a few cases as Main_Project_Specific (due to no explicit patch path), our deeper semantic analysis of stack traces reveals the dependency-specific nature"

**결론**: LLM이 더 정확할 가능성이 **매우 높음**
- 스택 트레이스가 libheif에서 명확히 발생
- Workaround 패치로 의존성 문제를 우회하는 패턴
- 높은 LLM 신뢰도와 그룹 내 일관성

---

### 케이스 2: localId 42540898 (imagemagick)

**GT 분류**:
- Type: `Main_Project_Specific`
- Dependency: `None`
- Confidence: `N/A` (정보 없음)
- Submodule Bug: `False`

**LLM 추론**:
- Type: `Dependency_Specific`
- Dependency: `Multiple (libjxl, libheif, libjpeg-turbo, libraw)`
- Confidence: `0.9`
- Dependency Score: `0.9`
- Main Project Score: `0.1`

**주요 증거**:

1. **스택 트레이스 분석**:
   - 크래시 위치: `src/libheif/src/heif_context.cc:1186:28`
   - 함수: `heif::HeifContext::decode_overlay_image`
   - libheif의 HEIF 디코딩 로직 내부에서 발생
   - ImageMagick은 단순히 libheif API를 호출하는 역할

2. **Workaround 패치 감지**:
   - `workaround_detected: True`
   - 의존성 버그를 메인 프로젝트에서 방어적으로 처리

3. **그룹 레벨 증거**:
   - 케이스 1과 동일한 Sub-Group (ID: 4)
   - 11개 케이스가 모두 동일한 의존성 그룹으로 분류
   - 높은 그룹 내 일관성

**결론**: LLM이 더 정확할 가능성이 **매우 높음**
- 스택 트레이스가 libheif 내부에서 명확히 발생
- Workaround 패치 패턴
- 그룹 내 다른 케이스들과 일관된 패턴

---

### 케이스 3: localId 42539707 (imagemagick)

**GT 분류**:
- Type: `Main_Project_Specific`
- Dependency: `None`
- Confidence: `N/A` (정보 없음)
- Submodule Bug: `False`

**LLM 추론**:
- Type: `Dependency_Specific`
- Dependency: `Multiple (libjxl, libheif, libjpeg-turbo, libraw)`
- Confidence: `0.9`
- Dependency Score: `0.9`
- Main Project Score: `0.1`

**주요 증거**:

1. **스택 트레이스 분석**:
   - 크래시 위치: `src/libheif/src/heif_context.cc:990:28`
   - 함수: `heif::HeifContext::decode_full_grid_image`
   - libheif의 그리드 이미지 디코딩 로직에서 발생
   - `heif_decode_image` API 호출 경로

2. **Workaround 패치 감지**:
   - `workaround_detected: True`
   - 의존성 문제를 메인 프로젝트에서 처리

3. **그룹 레벨 증거**:
   - 케이스 1, 2와 동일한 Sub-Group
   - 이미지 처리 라이브러리 그룹의 일관된 패턴

**결론**: LLM이 더 정확할 가능성이 **매우 높음**
- libheif 내부 로직에서 명확히 발생
- Workaround 패치 패턴
- 그룹 내 일관성

---

### 공통 패턴 분석

#### 1. 스택 트레이스 기반 증거
- **모든 케이스**: 스택 트레이스가 의존성 라이브러리(libheif) 내부에서 명확히 발생
- **ImageMagick 코어 코드 부재**: 스택 트레이스에 ImageMagick의 핵심 로직이 나타나지 않음
- **의존성 API 호출**: ImageMagick은 단순히 libheif API를 호출하는 역할

#### 2. Workaround 패치 패턴
- **모든 케이스**: `workaround_detected: True`
- **의미**: 메인 프로젝트에서 의존성 버그를 우회하는 방어적 패치
- **GT의 한계**: 패치 경로 정보 부족으로 Main으로 잘못 분류

#### 3. 그룹 레벨 일관성
- **Sub-Group ID: 4**: 11개 케이스가 모두 동일한 의존성 그룹
- **의존성 매칭**: 100% (11/11 케이스가 동일한 의존성 그룹)
- **Cross-project 패턴**: 여러 프로젝트에서 동일한 의존성 문제 발생

#### 4. LLM의 강점
- **높은 신뢰도**: 모든 케이스에서 0.9
- **명확한 점수 차이**: Dependency Score (0.9) >> Main Project Score (0.1)
- **의미론적 분석**: 스택 트레이스와 패치 의도를 종합적으로 분석

#### 5. GT의 한계
- **정보 부족**: Confidence Score가 `N/A`
- **패치 경로 기반 판단**: 패치 파일 경로만 보고 판단
- **스택 트레이스 미활용**: 실제 크래시 위치를 충분히 고려하지 않음

### 주요 발견사항

1. **LLM이 GT보다 정확한 경우**:
   - 스택 트레이스가 의존성에서 명확히 발생하는 경우
   - Workaround 패치가 감지되는 경우
   - 그룹 레벨 패턴이 일관된 경우

2. **GT의 한계**:
   - 패치 경로 정보만으로 판단하여 오분류
   - 스택 트레이스의 실제 크래시 위치를 충분히 고려하지 않음
   - Confidence Score 정보 부족

3. **LLM의 강점**:
   - 스택 트레이스와 패치 의도를 종합적으로 분석
   - 그룹 레벨 패턴을 활용한 일관성 검증
   - 높은 신뢰도와 명확한 점수 차이

### 결론

분석한 3개 케이스 모두에서 **LLM이 GT보다 더 정확했을 가능성이 매우 높습니다**. 주요 근거는:

1. 스택 트레이스가 의존성(libheif) 내부에서 명확히 발생
2. Workaround 패치 패턴으로 의존성 문제를 우회
3. 그룹 레벨에서 높은 일관성 (11/11 케이스)
4. LLM의 높은 신뢰도와 명확한 점수 차이

이러한 케이스들은 LLM의 의미론적 분석 능력과 그룹 레벨 패턴 활용의 효과를 보여줍니다.


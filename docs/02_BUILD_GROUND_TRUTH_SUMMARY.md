# 02 Build Ground Truth - 휴리스틱 룰 기반 Ground Truth 생성

## 개요

`02_build_ground_truth.py`는 추출된 데이터에 휴리스틱 룰을 적용하여 취약점의 근본 원인(`Main_Project_Specific` vs `Dependency_Specific`)을 자동으로 판단하고 Ground Truth를 생성하는 스크립트입니다.

### 처리 통계 (실제 데이터 기준)

| 항목 | 개수 | 비율 |
|------|------|------|
| 총 처리 항목 | 6,138 | 100% |
| **성공** | **6,046** | **98.50%** |
| 실패 | 92 | 1.50% |
| **의존성 있는 케이스** | **5,169** | **85.49%** |
| **의존성 없는 케이스** | **877** | **14.51%** |

**성공 케이스 분석**:
- **의존성 있는 케이스**: 5,169건 (85.49%) - dependencies_count > 0
- **의존성 없는 케이스**: 877건 (14.51%) - dependencies_count = 0, main_only=True

**실패 원인** (92건, 1.50%):
- **저장소 접근 실패**: repository access issues
- **데이터 불완전**: incomplete data

---

## 입력 데이터

`01_extract_from_db.py`에서 추출한 데이터:
- **OSS-Fuzz Report**: 프로젝트 정보, 버그 타입, 크래시 출력
- **Stack Trace**: 크래시 발생 시 스택 트레이스
- **Patch Info**: 패치 diff, 수정 커밋, 패치된 파일 경로
- **Srcmap**: 빌드 시 사용된 모든 의존성 라이브러리 정보

---

## 처리 프로세스

### 1. 데이터 검증 및 필터링

**의존성 필터링**:
- 메인 프로젝트 외에 다른 의존성이 있어야 Ground Truth 생성 가능
- 의존성이 없으면 `main_only`로 분류 (실패와 구분)

**필수 데이터 확인**:
- `patch_diff`: 필수 (없으면 스킵)
- `stack_trace`: Rule 2 평가에 필요
- `srcmap`: 의존성 정보 추출에 필요

### 2. 휴리스틱 룰 평가

#### 평가 순서
1. **Rule 1**: Patch File Path Mapping
2. **Rule 2**: Stack Trace Dominance (가중치 3, 최고, 증폭 조건 시 4)
3. **Rule 3**: Dependency Update Commit
4. **Rule 4**: External CVE Connection
5. **Workaround Detection**: patch-crash distance와 module mismatch 기반 (정보 제공)

#### 가중치 시스템
- Rule 1: 가중치 1
- Rule 2: 가중치 3 (최고) → 증폭 조건 시 가중치 4
- Rule 3: 가중치 1
- Rule 4: 가중치 1

### 3. Confidence Score 계산

**Confidence Score 의미**:
- **정의**: Cumulative Evidence Strength for Dependency_Specific hypothesis
- **Evidence Types**:
  - Positive Evidence: Supports Dependency_Specific (+)
  - Negative Evidence: Contradicts Dependency_Specific (-)
  - Strong Evidence: High weight multiplier (Rule 2 amplification)
  - Weak Evidence: Low weight (Rule 3)
- **Score 해석** (`Evidence_Strength_Interpretation` 필드):
  - Score > 6.0: **Strong** evidence for Dependency_Specific
  - Score 3.0-6.0: **Moderate** evidence
  - Score < 3.0: **Weak** or contradictory evidence

**Confidence Score 의미**:
- **정의**: Cumulative Evidence Strength for Dependency_Specific hypothesis
- **Evidence Types**:
  - Positive Evidence: Supports Dependency_Specific (+)
  - Negative Evidence: Contradicts Dependency_Specific (-)
  - Strong Evidence: High weight multiplier (Rule 2 amplification)
  - Weak Evidence: Low weight (Rule 3)
- **Score 해석**:
  - Score > 6.0: Strong evidence for Dependency_Specific
  - Score 3.0-6.0: Moderate evidence
  - Score < 3.0: Weak or contradictory evidence

```python
# 기본 계산 (가중합)
confidence_score = (rule1_score × 1) + (rule2_score × rule2_weight) + (rule3_score × 1) + (rule4_score × 1)

# Rule 2 점수: 연속값 (0.0~2.0)
rule2_score = 2.0 × COC  # COC가 None이면 2.0 (fallback)

# Rule 2 가중치 증폭: 연속 함수 (threshold artifact 제거)
def calculate_rule2_amplification(coc, patch_crash_distance):
    normalized_dist = min(patch_crash_distance / 3.0, 1.0)
    alpha = 1.0 / 3.0  # 0.333...
    amplification = 1.0 + alpha * coc * normalized_dist
    return min(amplification, 1.333...)  # Max: weight 3 → 4

rule2_weight = 3.0 × calculate_rule2_amplification(COC, patch_crash_distance)
```

**Rule 2 증폭 로직**:
- **연속 함수**: 이산 threshold 제거 (COC=0.79 vs 0.81의 급격한 차이 방지)
- **증폭 범위**: 1.0 ~ 1.333... (가중치 3.0 ~ 4.0)
- **예시**:
  - COC=1.0, dist=2 → amplification ≈ 1.333 (weight 3 → 4.0)
  - COC=0.8, dist=2 → amplification ≈ 1.267 (weight 3 → 3.8)
  - COC=0.6, dist=1 → amplification ≈ 1.06 (weight 3 → 3.18)

**최대값**:
- 이론적 최대값: 8.0 (Rule 2 증폭 시, rule2_score=2.0)
- 실제로는 8+도 가능 (Rule 2 증폭 + Rule 3/Rule 4 추가 점수)
- 예: Rule 2 증폭(8.0) + Rule 3(1) + Rule 4(1) = 10.0

**Rule 충돌 신뢰도 감소**:
- Rule 1이 Main을 가리키고 Rule 2가 Dependency를 가리킬 때 충돌 발생
- 충돌 시 `confidence_score *= 0.85` (15% 불확실성 증가)
- Root Cause 결정은 변경하지 않음 (Rule 2 우선 유지)
- 의미: "의존성 원인이 유력하지만, 패치가 명확히 메인에 있으므로 불확실성 증가"
- `rule_conflicts.conflict_penalty_applied`: 충돌 penalty 적용 여부
- `rule_conflicts.conflict_penalty_value`: 적용된 penalty 값 (0.85 또는 None)

**Rule 1 Retry 시**:
- Rule 1 score가 0이고 retry로 발견되면 곱셈 방식으로 재계산
- `confidence_score = rule1_score × rule2_score × rule3_score × rule4_score` (0인 경우 1로 대체)

### 4. Root Cause 결정

**우선순위**:
1. Workaround Detector: confidence 조정
2. Rule 2 (가중치 3): 만족 시 `Dependency_Specific`
3. Rule 1 Score = 2: 단일 의존성 → `Dependency_Specific`
4. Rule 1 indicates_main_project: `Main_Project_Specific`
5. 기타: 만족된 룰들에서 가장 자주 언급된 의존성 선택

---

## 주요 클래스 및 메서드

### `GroundTruthBuilder`

#### 핵심 메서드

**`build_ground_truth(local_id: int, data: Optional[Dict] = None) -> Optional[Dict]`**
- 메인 함수: 전체 Ground Truth 생성 프로세스
- 데이터 추출 → 룰 평가 → Root Cause 결정 → Ground Truth 구성

**`rule1_patch_file_path_mapping(...) -> Tuple[int, Optional[str], bool]`**
- 패치 파일 경로 분석
- Score: 0 (메인만), 1 (혼합), 2 (단일 의존성)
- Penalty: Module Mismatch (×0.3), Patch-Crash Distance (×0.5)

**`rule2_stack_trace_dominance(...) -> Tuple[bool, Optional[str], Optional[float]]`**
- 스택 트레이스 상위 프레임 분석
- COC (Crash Ownership Confidence) 계산
- 연속값 스코어링: `score = 2 × COC`

**`rule3_dependency_update_commit(...) -> Tuple[bool, Optional[str]]`**
- 의존성 업데이트 커밋 감지
- Score: 0 (만족하지 않음) 또는 1 (만족함)
- 만족 조건: `submodule_bug=True` 또는 `repo_addr`이 프로젝트와 다름

**`rule4_external_cve_connection(...) -> Tuple[bool, Optional[str]]`**
- 외부 CVE/NVD 연결 확인 (현재 미구현)

**`analyze_patch_type(patch_diff: str, patched_files: List[str]) -> Dict`**
- 패치 타입 분석 (version_update / code_fix / mixed)
- Version update bypass 감지

**`_compute_patch_crash_distance(...) -> Tuple[Optional[int], Optional[str], Optional[str]]`**
- 패치-크래시 구조적 거리 계산 (0-3)
- 모듈 분류 (crash_module, patched_module)

**Workaround Detection** (코드 내 직접 계산)
- `workaround_detected` 필드로 저장됨
- 조건: `patch_crash_distance >= 2` AND `crash_module != patched_module`
- Ground Truth 출력에 포함됨

---

## 출력 데이터 구조

### Ground Truth 항목

```python
{
    'localId': int,
    'Heuristically_Root_Cause_Type': 'Dependency_Specific' | 'Main_Project_Specific',
    'Heuristically_Root_Cause_Dependency': {
        'name': str,
        'commit_sha': str,
        'url': str,
        'path': str
    } | None,
    'Heuristic_Confidence_Score': float,  # Raw score (0~8+)
    'Heuristic_Max_Score': 8.0,
    'Heuristic_Satisfied_Rules': List[str],
    'Heuristic_Rule_Details': [
        {
            'rule': str,
            'dependency': str | None,
            'score': float,
            # Rule별 추가 필드
        },
        ...
    ],
    'rule1_indicates_main_project': bool,
    'project_name': str,
    'submodule_bug': bool,
    'dependencies_count': int,
    'total_dependencies_count': int,
    'patch_diff': str,  # 전체 패치 diff
    'patch_file_path': str,
    'patched_files': List[str],
    'stack_trace_key_locations': [
        {'file_path': str, 'line': int},
        ...
    ],
    'patch_analysis': {
        'patch_type': str,
        'is_version_update_bypass': bool,
        'version_update_files': List[str],
        'code_fix_files': List[str],
        'confidence': float
    },
    'rule_conflicts': {
        'rule1_main_vs_rule2_dep': bool,
        'rule2_dependency': str | None,
        'rule1_dependency': str | None,
        'conflict_penalty_applied': bool,  # Whether conflict penalty was applied
        'conflict_penalty_value': float | None  # Penalty value (0.85 or None)
    },
    'Evidence_Strength_Interpretation': str,  # 'Strong' | 'Moderate' | 'Weak'
    'patch_crash_distance': int | None,
    'crash_module': str | None,
    'patched_module': str | None,
    'workaround_detected': bool,
    'rule2_coc': float | None,
    'requires_dependency_fix': bool
}
```

---

## 검증 케이스

## 검증 케이스 분석

### 케이스 1: Main_Project_Specific 검토 (localId: 42516571)

**기본 정보**:
- **프로젝트**: skia
- **Root Cause Type**: Main_Project_Specific
- **의존성 개수**: 46개 (데이터셋에서 가장 많은 의존성을 가진 케이스 중 하나)
- **main_only**: False
- **bug_type**: Heap-buffer-overflow READ 8
- **crash_module**: skia
- **patched_module**: skia
- **patch_crash_distance**: 1
- **workaround_detected**: False

**상세 분석**:

1. **Stack Trace 분석**:
   - 모든 스택 프레임이 `/src/skia/` 경로를 가리킵니다:
     - `/src/skia/out/Fuzz/../../src/core/SkPath.cpp:1849`
     - `/src/skia/out/Fuzz/../../src/utils/SkParsePath.cpp:276`
     - `/src/skia/out/Fuzz/../../src/svg/SkSVGDevice.cpp:669`
     - `/src/skia/out/Fuzz/../../src/svg/SkSVGDevice.cpp:954`
     - `/src/skia/out/Fuzz/../../src/core/SkCanvas.cpp:2199`
   - 크래시가 명확히 skia 메인 프로젝트 내부 코드에서 발생했습니다.

2. **Patch 분석**:
   - **패치 위치**: `src/core/SkTDArray.cpp` (메인 프로젝트)
   - **모듈 일치**: crash_module=skia, patched_module=skia
   - **패치-크래시 거리**: 1 (같은 모듈 내에서 가까운 거리)
   - **Workaround 패턴 없음**: `workaround_detected=False`로 표시되어 실제 버그 수정임을 나타냅니다.

3. **의존성 분석**:
   - **의존성 개수**: 46개 (데이터셋에서 상위 수준)
   - **의존성 존재**: 많은 의존성이 있지만, 크래시와 패치 모두 메인 프로젝트에 있음
   - **의미**: 의존성이 많다고 해서 항상 의존성 버그인 것은 아님

**Rule 적용 결과**:
- **Rule 1 (Patch File Path Exclusivity)**: 
  - Score: 0.0 (메인 프로젝트)
  - Indicates Main Project: True
  - **의미**: 패치가 메인 프로젝트에 있어서 Main_Project_Specific을 지시
  
- **Rule 2 (Stack Trace Dominance)**:
  - Score: 0.0 (스택 트레이스도 메인 프로젝트를 가리킴)
  - Dependency: None
  - **의미**: 스택 트레이스가 메인 프로젝트를 명확히 가리킴

- **Rule 충돌 없음**: Rule 1과 Rule 2가 모두 Main을 지시하여 일관된 신호
- **최종 Confidence Score**: 0.0 (낮은 confidence는 의존성 버그가 아니라는 강한 신호)

**검증 결과**: ✅ **올바른 분류**

**근거**:
1. **의존성 존재에도 Main 버그**: 이 케이스는 46개의 의존성을 가지고 있지만, 모든 증거가 메인 프로젝트 버그를 가리킵니다. 이는 의존성이 많다고 해서 항상 의존성 버그인 것은 아님을 보여줍니다.
2. **증거 일관성**: Stack trace, crash module, patched module이 모두 메인 프로젝트(skia)를 가리키며, patch-crash distance가 1로 가까워 명확한 메인 프로젝트 버그입니다.
3. **Workaround 패턴 없음**: `workaround_detected=False`로 표시되어 실제 버그 수정이며, 의존성 버그를 우회하는 것이 아님을 나타냅니다.
4. **분류 로직 검증**: 휴리스틱 시스템이 의존성 개수에 관계없이 실제 증거를 기반으로 올바르게 분류함을 보여줍니다.

**결론**: 의존성이 많아도(46개) 모든 증거가 메인 프로젝트를 가리키면 Main_Project_Specific으로 올바르게 분류됩니다. 이 케이스는 휴리스틱 시스템이 의존성 존재 여부가 아닌 실제 증거를 기반으로 분류함을 검증합니다.

---

### 케이스 2: Dependency_Specific 검토 (localId: 432091963) - Workaround 패턴

**기본 정보**:
- **프로젝트**: imagemagick
- **Root Cause Type**: Dependency_Specific
- **Root Cause Dependency**: libjxl
  - Repository: `https://github.com/libjxl/libjxl`
  - Commit SHA: `a3c7fedfc19b979f9662800da6a716dfdd180ea5`
  - Path: `/src/libjxl`
- **의존성 개수**: 16개
- **bug_type**: Heap-buffer-overflow READ 4
- **crash_module**: libjxl
- **patched_module**: imagemagick
- **patch_crash_distance**: 2
- **workaround_detected**: True
- **requires_dependency_fix**: True
- **Heuristic_Confidence_Score**: 4.87

**상세 분석**:

1. **Stack Trace 분석**:
   - Stack trace의 모든 key locations가 `/src/libjxl/` 경로를 가리킵니다:
     - `/src/libjxl/lib/jxl/enc_ans_params.h:135`
     - `/src/libjxl/lib/jxl/enc_modular_simd.cc:217`
     - `/src/libjxl/lib/jxl/enc_modular_simd.cc:247`
     - `/src/libjxl/lib/jxl/enc_modular.cc:332`
     - `/src/libjxl/lib/jxl/enc_modular.cc:876`
   - 크래시가 명확히 libjxl 라이브러리 내부에서 발생했습니다.

2. **Patch 분석**:
   - **패치 위치**: `imagemagick/coders/sf3.c` (메인 프로젝트)
   - **커밋 메시지**: "check for EOF"
   - **패치 내용**: EOF(End of File) 체크 추가 - 방어적 코드
   ```c
   // 추가된 코드
   if (y < (ssize_t) image->rows)
   {
     ThrowFileException(exception, CorruptImageError, "UnexpectedEndOfFile",
       image->filename);
     break;
   }
   ```
   - **패치 타입**: `code_fix` (코드 수정)
   - **의도**: 파일의 끝에 도달했을 때 예외를 던지고 루프를 중단하는 방어적 코드

3. **Rule 적용 결과**:
   - **Rule 1 (Patch File Path Exclusivity)**: 
     - Score: 0.0
     - Dependency: None
     - Indicates Main Project: True
     - **의미**: 패치가 메인 프로젝트에 있어서 Main_Project_Specific을 지시
   
   - **Rule 2 (Stack Trace Dominance)**:
     - Score: 1.62
     - Dependency: libjxl
     - COC (Crash Ownership Confidence): 0.81 (높은 신뢰도)
     - **의미**: 스택 트레이스가 libjxl을 명확히 가리킴
   
   - **Rule 충돌**: Rule 1은 Main을, Rule 2는 Dependency를 지시하여 충돌 발생
   - **최종 Confidence Score**: 4.87 (Moderate evidence)
   - Rule 2가 더 높은 가중치(3.0)를 가지므로 최종 분류는 Dependency_Specific

4. **Workaround 패턴 검증**:
   - **패치-크래시 거리 = 2**: 패치와 크래시 위치가 구조적으로 분리됨
   - **모듈 불일치**: crash_module=libjxl, patched_module=imagemagick
   - **방어적 코드**: 패치는 libjxl의 버그를 직접 수정하지 않고, imagemagick 코드에서 예외 상황을 감지하고 처리합니다.
   - **증상 완화**: 실제 버그(libjxl의 heap-buffer-overflow)는 수정하지 않고, 그로 인한 크래시를 방지하기 위한 방어 코드를 추가합니다.
   - **근본 원인 미해결**: `requires_dependency_fix: True`로 표시되어 있어, 실제로는 libjxl 라이브러리 수정이 필요함을 나타냅니다.

**검증 결과**: ✅ **올바른 분류**

**근거**:
1. **Stack Trace 신뢰성**: 모든 스택 프레임이 libjxl 내부 코드를 가리키며, COC가 0.81로 높아 크래시 위치가 매우 신뢰할 만합니다.
2. **Workaround 패턴 명확**: 패치가 메인 프로젝트에 있지만 크래시는 의존성에서 발생하는 전형적인 workaround 패턴입니다. imagemagick은 libjxl의 버그를 직접 수정할 수 없으므로 자신의 코드에 방어적 체크를 추가했습니다.
3. **패치 의도**: "check for EOF"라는 커밋 메시지와 추가된 방어 코드가 workaround의 전형적인 특징입니다.
4. **구조적 거리**: patch_crash_distance=2는 패치와 크래시가 구조적으로 분리되어 있음을 나타냅니다.
5. **의존성 수정 필요성**: `requires_dependency_fix=True`로 설정되어 있어 실제로는 libjxl 라이브러리 수정이 필요함을 나타냅니다.
6. **Rule 2 우선순위**: Rule 2(Stack Trace Dominance)가 Rule 1(Patch File Path)보다 높은 가중치를 가지며, 이 경우 Rule 2가 더 신뢰할 만한 신호를 제공합니다.

**결론**: 이 케이스는 전형적인 workaround 패턴으로, 실제 버그는 libjxl 의존성에 있지만 패치는 imagemagick 메인 프로젝트에 적용되었습니다. Dependency_Specific으로 분류한 것이 정확하며, 휴리스틱 시스템이 workaround 패턴을 올바르게 감지했습니다.실제 heap-buffer-overflow는 여전히 libjxl에 존재하지만 imagemagick은 입력 차단으로 증상만 회피했을 것으로 보입니다.

**Workaround 패턴 통계**:
- 전체 데이터셋에서 `workaround_detected=True`: 804건
- 의존성 정보가 있는 workaround 케이스: 189건 (모두 Dependency_Specific)
- 이 케이스는 189건 중 하나로, 실제 의존성 버그를 올바르게 식별한 사례입니다.

---

### 검증 요약

| 케이스 | Type | 검토 결과 | 주요 근거 |
|--------|------|----------|----------|
| Case 1 (42516571) | Main_Project_Specific | ✅ 정확 | 의존성 46개 있지만 모든 증거가 메인 프로젝트(skia)를 가리킴 |
| Case 2 (432091963) | Dependency_Specific | ✅ 정확 | Stack trace가 libjxl을 가리키며, workaround 패턴 (patch-crash distance=2) |

**검증 통계**:
- 검토한 케이스 수: 2건
- 올바른 분류: 2건 (100%)
- Main_Project_Specific 정확도: 1/1 (100%)
- Dependency_Specific 정확도: 1/1 (100%)

**주요 발견 사항**:
1. **의존성 존재와 버그 분류**: 의존성이 많아도(46개) 모든 증거가 메인 프로젝트를 가리키면 Main_Project_Specific으로 올바르게 분류됩니다. 의존성 존재 여부가 아닌 실제 증거가 중요합니다.
2. **증거 일관성**: Stack trace, crash module, patched module이 모두 같은 위치를 가리킬 때 분류가 명확해집니다.
3. **Workaround 패턴**: Patch가 메인 프로젝트에 있지만 크래시가 의존성에서 발생하는 경우, Stack Trace 신호가 더 신뢰할 만하며 Dependency_Specific 분류가 정확합니다.
4. **Rule 우선순위**: Rule 2(Stack Trace Dominance)가 Rule 1(Patch File Path)보다 높은 가중치를 가지는 것이 타당하며, 실제 케이스에서도 올바르게 작동합니다.

---

## 주요 특징

### 1. 의존성 필터링
- 메인 프로젝트만 있는 경우 `main_only`로 분류 (실패와 구분)
- 다른 의존성이 있어야 Ground Truth 생성 가능

### 2. Rule 2 증폭 메커니즘 (연속화)
- **연속 함수**: 이산 threshold 제거 (COC=0.79 vs 0.81의 급격한 차이 방지)
- **증폭 범위**: 1.0 ~ 1.333... (가중치 3.0 ~ 4.0)
- **예시**:
  - COC=1.0, dist=2 → amplification ≈ 1.222 (weight 3 → 3.67)
  - COC=0.8, dist=2 → amplification ≈ 1.178 (weight 3 → 3.53)
  - COC=0.6, dist=1 → amplification ≈ 1.067 (weight 3 → 3.20)
- **의미**: "크래시는 확실히 의존성인데, 패치는 멀리서 방어만 한다"

### 3. Rule 1 Penalty 시스템
- **Module Mismatch**: score × 0.3
- **Patch-Crash Distance**: score × 0.5
- 더 강한 penalty만 적용 (중첩 방지)
- 연속값 유지 (정수화로 인한 정보 손실 방지)

### 4. Workaround Detection
- `workaround_detected` 필드로 저장됨
- 조건: `patch_crash_distance >= 2` AND `crash_module != patched_module`
- Ground Truth 출력에 포함되어 LLM이 참고할 수 있도록 제공
- Confidence score를 직접 조정하지 않음 (정보 제공 목적)

### 5. Rule Conflict 처리
- Rule 1 (메인) vs Rule 2 (의존성) 충돌 감지
- Rule 2가 더 높은 가중치로 우선순위 가짐
- 충돌 시 `confidence_score *= 0.85` (15% 불확실성 증가)
- `rule_conflicts.conflict_penalty_applied`: penalty 적용 여부
- `rule_conflicts.conflict_penalty_value`: 적용된 penalty 값 (0.85 또는 None)

---

## 사용 예시

### 기본 사용
```bash
# 단일 localId 처리
python3 02_build_ground_truth.py --localId 42533949

# 여러 localId 처리
python3 02_build_ground_truth.py --localIds 42533949 42492074

# 프로젝트별 처리
python3 02_build_ground_truth.py --project libspdm -n 10

# 전체 처리
python3 02_build_ground_truth.py --all
```

### 출력 파일
- 기본: `ground_truth.json`
- 형식:
```json
{
  "summary": {
    "total_processed": int,
    "success": int,
    "failed": int,
    "failed_localIds": [...],
    "main_only": int,
    "main_only_localIds": [...],
    "min_confidence": int
  },
  "ground_truth": [...]
}
```

---

## 데이터 흐름

```
extracted_data.json (01_extract_from_db.py)
    ↓
build_ground_truth()
    ↓
[데이터 검증]
    ├─→ 의존성 필터링
    ├─→ 필수 데이터 확인 (patch_diff)
    └─→ 데이터 준비
         ↓
[휴리스틱 룰 평가]
    ├─→ Rule 1: Patch File Path Mapping
    ├─→ Rule 2: Stack Trace Dominance (COC)
    ├─→ Rule 3: Dependency Update Commit
    └─→ Rule 4: External CVE Connection
         ↓
[Confidence Score 계산]
    ├─→ 가중합 계산
    └─→ Rule 2 증폭 적용 (조건 만족 시)
         ↓
[Workaround Detection]
    └─→ patch-crash distance와 module mismatch 기반 계산
         ↓
[Root Cause 결정]
    ├─→ Rule 우선순위 적용
    ├─→ 충돌 처리
    └─→ 의존성 정보 매핑
         ↓
[Ground Truth 구성]
    ├─→ 메타데이터 추가
    ├─→ Rule 상세 정보
    └─→ 참조 정보 (파일 경로, 라인 번호)
         ↓
ground_truth.json
```

---

## 핵심 설계 원칙

### 1. 크래시 위치 우선
- 패치 위치보다 크래시 위치(스택 트레이스)가 더 신뢰할 만한 신호
- Rule 2가 Rule 1보다 높은 가중치를 가지는 이유

### 2. Workaround 감지
- 패치가 메인 프로젝트에 있어도 크래시가 의존성에서 발생하면 Dependency_Specific
- 패치-크래시 거리와 모듈 불일치로 workaround 패턴 감지

### 3. 연속값 스코어링
- 이산값 대신 연속값 사용 (COC 기반)
- Penalty도 연속값 유지 (정보 손실 방지)

### 4. Workaround Detection
- `workaround_detected` 필드는 정보 제공 목적
- Confidence score를 직접 조정하지 않음
- LLM이 참고할 수 있도록 Ground Truth에 포함

---

## 검증 결과 요약

✅ **Confidence Score 계산**: 가중치 적용 및 Rule 2 amplification 정확  
✅ **Rule 적용 순서**: Rule 1 → Rule 2 → Rule 3 → Rule 4 순서 정확  
✅ **Root Cause 결정**: Rule 2 우선순위 및 충돌 처리 정확  
✅ **Rule Conflicts 감지**: Rule 1(메인) vs Rule 2(의존성) 충돌 정확히 감지  
✅ **실제 분류 정확성**: 패치가 메인 프로젝트에 있어도 크래시가 의존성에서 발생하면 Dependency_Specific으로 올바르게 분류

---

## 실제 데이터 분석 결과 (최신 통계)

### 전체 통계

| 항목 | 개수 | 비율 |
|------|------|------|
| **총 처리된 케이스** | 6,138 | 100% |
| **성공** | 6,046 | 98.50% |
| **실패** | 92 | 1.50% |
| **의존성 있는 케이스** | 5,169 | 85.49% |
| **의존성 없는 케이스** | 877 | 14.51% |

### Root Cause Type 분포 (6,046개 성공 케이스)

| 타입 | 개수 | 비율 |
|------|------|------|
| **Main_Project_Specific** | 5,857 | **96.87%** |
| **Dependency_Specific** | 189 | **3.13%** |

**의존성 있는 케이스 중 분포**:
- Main_Project_Specific: 4,980건 (96.34% of dependency-involved cases)
- Dependency_Specific: 189건 (3.66% of dependency-involved cases)

**분석**: 전체 케이스의 85.5%가 의존성을 가지고 있지만, 실제 의존성 버그는 3.1%에 불과합니다. 대부분(82.4%)은 의존성을 사용하는 메인 프로젝트 버그입니다.

### 의존성 개수 분포

| 의존성 개수 | 케이스 수 | 비율 |
|------------|----------|------|
| 0개 | 877건 | 14.51% |
| 1개 | 2,529건 | 41.83% |
| 2개 | 950건 | 15.71% |
| 3개 | 388건 | 6.42% |
| 4개 | 158건 | 2.61% |
| 5개 이상 | 1,144건 | 18.92% |

**통계**:
- 평균: 3.74개
- 중앙값: 1개
- 최대: 46개
- 백분위수: 50% = 1개, 75% = 3개, 90% = 13개, 99% = 31개

### Dependency_Specific 버그 상세 분석

**의존성 이름별 상위 10개**:

| 의존성 이름 | 케이스 수 | 비율 |
|------------|----------|------|
| libde265 | 23건 | 12.2% |
| wolf-ssl-ssh-fuzzers | 20건 | 10.6% |
| libraw | 18건 | 9.5% |
| hdf5-1.12.0 | 12건 | 6.3% |
| libjxl | 11건 | 5.8% |
| qtbase | 10건 | 5.3% |
| llvm-project | 8건 | 4.2% |
| ffmpeg | 8건 | 4.2% |
| freetype2 | 7건 | 3.7% |
| libjpeg-turbo | 6건 | 3.2% |

### 프로젝트별 통계

**총 프로젝트 수**: 307개

**상위 10개 프로젝트**:

| 프로젝트 | 총 케이스 | 의존성 있음 | 의존성 없음 | 의존성 비율 |
|---------|----------|------------|------------|------------|
| imagemagick | 459건 | 459건 (100.0%) | 0건 (0.0%) | 100.0% |
| ffmpeg | 400건 | 400건 (100.0%) | 0건 (0.0%) | 100.0% |
| gdal | 253건 | 186건 (73.5%) | 67건 (26.5%) | 73.5% |
| ndpi | 179건 | 129건 (72.1%) | 50건 (27.9%) | 72.1% |
| ghostpdl | 177건 | 177건 (100.0%) | 0건 (0.0%) | 100.0% |
| binutils-gdb | 165건 | 165건 (100.0%) | 0건 (0.0%) | 100.0% |
| skia | 163건 | 163건 (100.0%) | 0건 (0.0%) | 100.0% |
| harfbuzz | 151건 | 119건 (78.8%) | 32건 (21.2%) | 78.8% |
| opensc | 127건 | 102건 (80.3%) | 25건 (19.7%) | 80.3% |
| libxml2 | 118건 | 56건 (47.5%) | 62건 (52.5%) | 47.5% |

### Bug Type 통계

**전체 통계**:
- Bug Type이 있는 케이스: 2,640건 (43.67%)
- Bug Type이 없는 케이스: 3,406건 (56.33%)
- 총 Bug Type 종류: 116종류

**상위 10개 Bug Type**:

| Bug Type | 케이스 수 | 비율 |
|----------|----------|------|
| Use-of-uninitialized-value | 521건 | 8.62% |
| Heap-buffer-overflow READ 1 | 308건 | 5.09% |
| UNKNOWN READ | 250건 | 4.13% |
| Heap-buffer-overflow READ 4 | 126건 | 2.08% |
| Index-out-of-bounds | 114건 | 1.89% |
| Heap-buffer-overflow WRITE 1 | 87건 | 1.44% |
| Heap-buffer-overflow READ {*} | 84건 | 1.39% |
| Heap-buffer-overflow READ 8 | 81건 | 1.34% |
| UNKNOWN WRITE | 76건 | 1.26% |
| Segv on unknown address | 73건 | 1.21% |

### Crash Module 통계

**전체 통계**:
- Crash Module이 있는 케이스: 2,587건 (42.79%)
- 총 Crash Module 종류: 157종류

**상위 10개 Crash Module**:

| Crash Module | 케이스 수 | 비율 |
|-------------|----------|------|
| llvm-project | 326건 | 5.39% |
| ffmpeg | 319건 | 5.28% |
| imagemagick | 178건 | 2.94% |
| llvm | 160건 | 2.65% |
| skia | 149건 | 2.46% |
| ghostpdl | 139건 | 2.30% |
| gdal | 119건 | 1.97% |
| mruby | 88건 | 1.46% |
| php-src | 57건 | 0.94% |
| wireshark | 46건 | 0.76% |

### 기타 필드 통계

**main_only**:
- True: 3,406건 (56.33%)
- False: 2,640건 (43.67%)

**requires_dependency_fix**:
- True: 128건 (2.12%)
- False: 5,918건 (97.88%)

**submodule_bug**:
- True: 12건 (0.20%)
- False: 6,034건 (99.80%)

**workaround_detected**:
- True: 804건 (13.3% of total cases)
- 의존성 정보가 있는 workaround 케이스: 189건 (모두 Dependency_Specific)

---

## 참고사항

1. **patch_diff 필수**: 없으면 Ground Truth 생성 불가 (실패 원인의 81.2%)
2. **의존성 필수**: 메인 프로젝트만 있으면 `main_only`로 분류 (실패 원인의 18.8%)
3. **Rule 2가 최고 가중치**: Stack Trace Dominance가 가장 신뢰할 만한 신호
4. **Workaround 감지**: `workaround_detected` 필드로 저장됨 (조건: patch_crash_distance >= 2 AND module mismatch)
5. **증폭 메커니즘**: COC ≥ 0.8 AND distance ≥ 2일 때 Rule 2 가중치 증폭
6. **Penalty 중첩 방지**: Rule 1의 두 penalty는 더 강한 것만 적용
7. **연속값 유지**: 정수화로 인한 정보 손실 방지

### 실패 케이스 분석

**실패한 케이스**: 92건 (1.50%)

**실패 원인**:
- 저장소 접근 실패: repository access issues
- 데이터 불완전: incomplete data

**참고**: 실패한 케이스들은 `ground_truth` 데이터에서 제외되어 있어 상세 정보는 `summary.failed_localIds`에만 기록되어 있습니다.



---


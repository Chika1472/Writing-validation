# Colab 실행 노트북

이 폴더는 기존 `src/` 모듈과 `scripts/` CLI를 Colab에서 재현 가능하게 실행하는
orchestration 계층이다. 구현 코드를 셀에 복제하지 않으므로 Python 모듈과 노트북이 서로
달라지지 않는다. 모든 CLI는 저장소 루트에서 `subprocess.run([...], check=True)`로 실행된다.

## 통합 실행본

`99_all_in_one_colab_pipeline.ipynb`는 아래 8개 노트북의 고유 셀을 한 파일에 원래 순서대로
합친 마스터 노트북이다. 공통 clone·Google Drive·dependency 셀은 맨 앞에 한 번만 두었고,
PART 구분선과 목차로 각 작업 단위를 나눴다. 이 통합본에서는 세션 종료 후 산출물을 보존하기
위해 `USE_GOOGLE_DRIVE=True`가 기본값이며, 테스트와 비용 발생 단계는 `RUN_*=False`다.

## 권장 실행 순서

1. `00_colab_environment_and_data.ipynb`
   - 저장소 clone, 선택적 Google Drive 연결, 의존성, 데이터 감사, 테스트, 5-fold/LOPO
2. `01_zero_shot_baseline.ipynb`
   - 선택적 Qwen3-14B zero-shot 기준선 생성과 파싱/지표 분석
3. `02_cpu_baselines.ipynb`
   - CPU 기준선, nested TF-IDF, 평가, bootstrap, calibration
4. `03_qwen_scorer_training.ipynb`
   - fixed epoch, fold×seed plan/훈련 registry, checkpoint, OOF 조립
5. `04_calibration_ensemble_inference.ipynb`
   - Qwen calibration, 4bit/BF16 비교, Qwen+TF-IDF stacker, anchor/assessment 후보
6. `05_rationale_and_finalization.ipynb`
   - genuine OOF silver, rationale SFT, adapter/fallback 최종화, blind review
7. `06_submission_and_tests.ipynb`
   - 외부 test 입력, 전체 회귀 테스트, create-only runtime config, 제출 엔진, 패키징,
     실제 L40S smoke

`90_optional_final_retraining.ipynb`는 validation label의 최종 학습 사용이 규정상 명시적으로
허용된 경우에만 실행한다. 일반 개발 artifact와 완전히 분리되어 있다.

## 실행 전 필수 설정

- GPU 노트북은 BF16을 지원하는 L4/A100급 런타임이 필요하다. T4는 지원하지 않는다.
- 각 노트북은 변환 시점의 저장소 commit을 기본 `REPO_REF`로 고정한다. 다른 commit을 쓸
  때는 모든 노트북에서 동일한 `WRITING_VALIDATION_REPO_REF`를 지정한다.
- `MODEL_REVISION`에는 `Qwen/Qwen3-14B`의 40자리 commit SHA를 입력한다.
- 새 Colab 세션에서 모델을 최초로 받을 때만 `ALLOW_MODEL_DOWNLOAD=True`를 사용한다. 단,
  scorer registry의 immutable signature에 이 값이 포함되므로 `03`에서 plan한 뒤에는 같은
  experiment를 resume하는 모든 세션에서 값을 그대로 유지한다.
- 긴 훈련은 `USE_GOOGLE_DRIVE=True`로 두고 모든 세션에서 동일한 `DRIVE_ARTIFACT_DIR`과
  `DRIVE_HF_HOME`을 유지한다. 생성 후 artifact를 옮기면 manifest 경로 검증이 실패할 수 있다.
- 각 `RUN_NAMESPACE`/`EXPERIMENT_ID`는 immutable 실행 단위다. 많은 출력은 create-only이므로
  기존 산출물을 삭제하거나 덮어쓰지 말고 새 namespace를 사용한다.
- `06`에서는 학습 artifact를 고르는 `SCORER_RUN_NAMESPACE`와 runtime YAML을 구분하는
  `RUNTIME_CONFIG_TAG`가 분리되어 있다. 같은 scorer로 Colab 기능검증과 L40S package config를
  각각 create-only로 만들 때는 scorer namespace는 유지하고 config tag만 바꾼다.

## 저장소에 포함되지 않은 외부 입력

- 공식 unlabeled `dataset/test.jsonl`
- 제출용 exact `requirements-lock.txt`
- `LICENSE`, `THIRD_PARTY_NOTICES.md`
- 선택적 rationale judge GGUF
- 공식 400건 offline 검증용 NVIDIA L40S 환경

각 노트북의 비싼 단계와 변경 단계는 기본적으로 `RUN_*=False`다. 먼저 출력 command와 경로를
확인한 뒤 필요한 스위치만 켠다. `00`과 `06`의 정적 테스트만 기본 실행된다.

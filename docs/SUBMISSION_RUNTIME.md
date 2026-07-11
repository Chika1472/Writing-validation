# 제출 추론 런타임 계약

이 문서는 아직 공개되지 않은 대회 SDK의 클래스명이나 함수 시그니처를 가정하지 않는다. 실제 SDK가 공개되면 SDK 어댑터는 입력 행을 `EssayInput`으로 변환하고 `SubmissionEngine.predict()`를 한 번 호출한 뒤 `SubmissionResult.rows`를 반환하는 얇은 경계로만 작성한다. 점수 계산, calibration/stacking, rationale 생성, strict JSON 조립은 어댑터로 옮기지 않는다.

## 고정 API

```python
from src.inference.submission import SubmissionEngine

engine = SubmissionEngine.from_config("inference_l40s.yaml")
result = engine.predict(all_test_rows)
final_rows = result.rows
```

`predict()`에는 전체 시험 split을 입력 순서대로 한 번에 전달한다. 설정의 `runtime.expected_rows`와 다르거나 ID가 중복되면 실행을 거부한다. scorer fold들은 한 GPU에서 순차 적재·해제하고, score matrix를 완성한 뒤에만 rationale adapter를 적재한다. rationale 조립은 score float를 반올림하거나 재생성하지 않는다.

## 설정 승격 규칙

`configs/inference_l40s.yaml`은 `submission_inference_v2`의 의도적으로 실행 불가능한
템플릿이다. 다음 항목을 동결한 뒤에만 사용한다.

- `scoring.qwen.model_revision`: Qwen base의 40자리 commit SHA
- `scoring.qwen.checkpoints`: 고정 epoch로 선택한 모든 fold checkpoint
- `scoring.qwen.calibrator`: 정확히 같은 checkpoint ensemble/precision의 OOF calibrator 또는 `null`
- stacker를 쓸 때 승격된 baseline/anchor/assessment artifact, stacker artifact와
  설정한 모든 source alias
- rationale SFT가 승격됐을 때만 `rationale.mode: adapter`와 checkpoint

`qwen_ensemble`은 모든 보조 source와 stacker 경로가 `null`이어야 한다.
`qwen_multisource_stacker`는 Qwen과 하나 이상의 baseline/anchor/assessment source를
요구하며, artifact가 있는 source만 고유 alias를 가져야 한다. Anchor는 deployment의
scorer checkpoint와 fold별 embedding 계약이 같아야 하고, assessment는 패키지에 든
Qwen model/tokenizer revision과 feature contract가 같아야 한다. Stacker의 source order와
모든 scorer signature가 일치하지 않으면 모델 적재 전에 실패한다. 알 수 없는 설정 키도
오류다.

## 패키징

다음 명령은 학습 artifact를 수정하지 않고 새 디렉터리만 만든다. 실제 실행은 고정 dependency와 모델 cache가 준비된 L40S 이관 환경에서 한다.

```powershell
python scripts/package_submission.py `
  --config configs/inference_l40s.yaml `
  --output-dir artifacts/submission/package_v1 `
  --requirements-lock requirements-lock.txt `
  --license-file LICENSE `
  --license-file THIRD_PARTY_NOTICES.md `
  --hf-home C:\path\to\tested_huggingface_cache
```

패키지는 checkpoint closure, 인접 provenance manifest, source code, dependency lock, 라이선스, pinned Hugging Face snapshot을 복사한다. Hugging Face cache symlink는 패키지 내부 일반 파일로 materialize한다. `package.manifest.json`에는 config와 모든 payload 파일의 SHA256이 들어가며, 실행기는 모델 적재 전에 이를 전부 다시 검증한다.

SDK가 서명 검증 전에 import하면서 만들 수 있는 `__pycache__/*.pyc`만 transient runtime
cache로 간주해 payload closure에서 제외한다. 그 밖의 unsigned 파일은 모두 거부하며,
설정 적재 직후 `PYTHONDONTWRITEBYTECODE=1`도 강제한다.

dependency lock의 각 행은 L40S에서 확정한 무조건부 `name==version`이어야 한다. pip의 `--hash=sha256:...`와 줄 연속 표시는 허용하지만 URL, editable install, version range, environment marker는 거부한다.

## 오프라인 smoke test

최종 검증은 패키지 밖의 새 report 경로를 사용한다.

```powershell
python scripts/smoke_test_submission.py `
  --config artifacts/submission/package_v1/inference_l40s.yaml `
  --input path/to/400_rows.jsonl `
  --report artifacts/reports/submission_package_v1_smoke.json `
  --expected-count 400 `
  --offline `
  --strict `
  --verify-determinism `
  --require-package-manifest
```

smoke test는 Hugging Face offline 환경변수와 socket/DNS 차단을 동시에 적용한다. 모든 최종 행을 strict parser로 왕복시키고, 입력 순서와 score 보존을 검사하며, 두 번의 fresh model load에서 score matrix가 bitwise 동일한지 확인한다. pass별 경과 시간과 peak CUDA allocated/reserved memory를 기록하고, 실행 전후 입력·artifact hash가 달라지면 실패한다.

`scripts/run_submission.py`는 동일 엔진을 사용해 ID 포함 JSONL, 내부 ledger, 선택적 bare JSONL, provenance manifest를 새 경로에 원자적으로 기록한다. 공식 SDK의 최종 출력 형식이 확인되면 이 CLI의 바깥 행 wrapper만 맞춘다.

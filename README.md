# 2026 글쓰기 채점 능력 평가 — 구현 저장소

상세 설계는 [`QWEN3_14B_최고점_전략_및_구현_계획.md`](./QWEN3_14B_최고점_전략_및_구현_계획.md)를 따른다. 현재 코드는 다음 범위를 구현한다.

- 엄격한 train/validation JSONL schema와 원문 보존 loader
- 규정 명시 승인 후에만 열리는 immutable train+validation 최종 재학습 데이터 builder
- 표면 feature, deterministic prompt×cohort×score-band 5-fold
- label-free leave-one-prompt-out 강건성 진단 fold
- global mean, prompt mean, surface OLS, TF-IDF Ridge 기준선
- outer-train-only nested-CV TF-IDF Ridge 후보와 배포 fold ensemble
- 영역별 RMSE/MAE/Pearson/Spearman/bias와 prompt·길이 slice
- paired prompt-stratified bootstrap
- strict/repaired zero-shot artifact 재감사
- OOF affine + shrinkage prompt residual calibration
- Qwen3 shared backbone + 세 영역 독립 continuous/native-grid head
- MSE + ordinal distribution + pairwise rank loss
- 동일 문항 multi-item batch 기반 soft-rank ablation
- 정확한 최종 JSON serializer와 strict parser
- 단일 fold QLoRA scorer 학습 CLI
- epoch checkpoint 재적재와 score-only 순차 ensemble inference CLI
- label이 없는 test JSONL용 inference schema/loader
- 누수 없는 cross-fitted calibrated OOF 생성
- CPU baseline fold-model 저장·hash 검증·재적재
- Qwen+TF-IDF trait별 simplex OOF stacker와 level-1 meta cross-fit
- 2개 이상 source(Qwen·TF-IDF·anchor·assessment)용 일반 simplex stacker
- exact-span evidence ledger와 score-preserving 최종 rationale fallback
- fold×seed 학습 orchestration, 전역 fixed-epoch 정책, 중단 안전 registry
- 동일 checkpoint-set 4bit/BF16 held-out OOF 비교와 bootstrap 승격 gate
- fold-checkpoint별 hidden embedding, 누수 차단 prompt-aware anchor/KNN
- 제한 answer-code assessment-question logits와 nested-CV Ridge 후보
- grounded silver 생성, score-jitter rationale QLoRA SFT, adapter 근거 생성
- hidden key를 judge 단계에서 차단하는 로컬 GGUF blind AB/BA rationale judge·집계기
- Qwen+선택적 baseline/anchor/assessment SDK 중립 offline 제출 엔진,
  hash-signed 패키징, L40S 400건 smoke test

## 중요한 실행 원칙

이 저장소에서 제공된 데이터 원본은 절대로 덮어쓰지 않는다. 모든 결과는 `artifacts/` 아래 새 경로에 기록한다.

- calibration은 반드시 OOF prediction으로만 fit한다.
- `id`, `document_id`, 수집 연도는 모델 feature로 사용하지 않는다.
- 긴 essay를 조용히 truncate하지 않는다. 설정 길이를 넘으면 실패시킨다.
- 최종 score는 generation text가 아니라 scorer head에서 얻는다.
- rationale를 붙일 때도 score를 다시 생성하거나 수정하지 않는다.
- 바깥 JSON은 `src/inference/serializer.py`가 조립한다.
- model/tokenizer revision SHA가 없으면 scorer 학습 CLI가 기본적으로 중단한다.

## 권장 대상 환경

- Linux
- Python 3.10 또는 3.11
- CUDA와 BF16을 지원하는 NVIDIA GPU
- 최종 검증: 단일 NVIDIA L40S 48GB

현재 파일에 적힌 명령은 대상 학습 환경에서 실행할 용도다. 이 코드 작성 환경에서는 사용자의 지시에 따라 추가 실행·학습·데이터 처리를 하지 않는다.

## 설치

CPU 기준선과 평가 도구:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Qwen 학습 환경:

```bash
python -m pip install -e ".[qwen,test]"
```

로컬 GGUF rationale judge 환경(`llama-cpp-python` 추가):

```bash
python -m pip install -e ".[judge,test]"
```

`pyproject.toml`은 major-version 상한까지 둔다. 첫 정상 smoke run 후 다음을
`requirements-lock.txt` 또는 제출 이미지 lock으로 **정확한 버전**까지 고정한다.

- Python, PyTorch, CUDA
- Transformers, PEFT, bitsandbytes, Accelerate
- scikit-learn, NumPy, SciPy, pandas
- Qwen model/tokenizer revision SHA

검증하지 않은 최신 dependency 조합을 그대로 제출 이미지에 사용하지 않는다.

## 1. 정적 품질 및 단위 테스트

대상 환경에서 실행:

```bash
python -m pytest -q
```

## 2. 데이터 감사

```bash
python scripts/audit_data.py --config configs/data.yaml
```

기본 산출물:

```text
artifacts/reports/data_audit.json
artifacts/reports/data_audit_manifest.json
```

## 3. 기존 zero-shot 결과 재감사

```bash
python scripts/reproduce_zero_shot.py --config configs/data.yaml
```

strict parse 결과와 관찰된 괄호 오류를 제한적으로 복구한 결과를 분리한다.

```text
artifacts/reports/zero_shot_reaudit.json
artifacts/predictions/qwen3_14b_zero_shot_repaired.jsonl
```

복구 결과를 strict 성공으로 보고하면 안 된다.

## 4. 고정 fold 생성

```bash
python scripts/make_folds.py \
  --config configs/data.yaml \
  --n-folds 5 \
  --seed 42
```

생성된 fold JSONL의 SHA256을 이후 모든 OOF run manifest에 보존한다.

### Leave-one-prompt-out 강건성 진단

문항 shortcut과 미지 문항 취약성을 별도로 보기 위한 label-free LOPO fold도 만든다.
이는 주 OOF를 대체하지 않는 진단용 계약이다.

```bash
python scripts/make_lopo_folds.py \
  --config configs/data.yaml \
  --output artifacts/folds/lopo_by_prompt.jsonl
```

`prompt_num`의 결정적 정렬만으로 문항 하나를 fold 하나에 배정하며 점수는 fold 생성에
사용하지 않는다. 이 fold로 baseline/scorer를 별도 실행해 unknown-prompt 성능을 보고하되,
주 5-fold OOF와 섞어 calibration이나 stacker를 fit하지 않는다.

### 규정 승인 후 조건부 train+validation 최종 재학습

이 경로는 일반 개발용 데이터 준비가 아니다. 운영 규정 또는 운영진 답변이 validation
label을 최종 학습에 사용할 수 있다고 **명시적으로 확인된 뒤**, 모든 hyperparameter와
rubric을 잠근 최종 재학습에만 사용한다. 허용되지 않거나 불명확하면 이 절의 명령을
실행하지 않고 `configs/data.yaml`의 train 2,000건 계약을 유지한다.

```bash
python scripts/build_final_combined_data.py \
  --config configs/data_final_combined.yaml \
  --acknowledge-rules-allow-validation-label-training
```

긴 승인 플래그가 없으면 CLI는 설정 파일이나 원본 데이터조차 읽기 전에 중단한다.
승인 플래그는 규정 확인을 대신하지 않으며, 실행자의 확인 사실을 명시하는 fail-closed
장치다. builder는 train 다음 validation 순서를 유지한 새 JSONL을
`artifacts/final_train_validation/data/` 아래 만들고 다음을 검증·기록한다.

- 두 원본의 strict exact schema와 중복 JSON key
- 전체 2,400건에 걸친 `id`·`document_id` 중복/충돌 부재
- 기존 split loader와 동일한 train↔validation 완전 동일 본문 누수 방지
- 원본별 SHA256, 행 순서, ordered ID, cohort count
- 결합 파일 SHA256, LF 정규화 직렬화 계약과 관련 코드·설정 hash
- 기존 원본·결합 파일·manifest를 덮어쓰지 않는 exclusive-create 계약

결합 artifact를 만든 뒤에도 기존 2,000건 OOF artifact와 섞지 않는다. 별도 namespace에서
새 fold, 새 fold×seed checkpoint, 새 OOF, 새 calibrator와 stacker를 처음부터 만든다.

```bash
python scripts/make_folds.py \
  --config configs/data_final_combined.yaml \
  --n-folds 5 --seed 42 \
  --output artifacts/final_train_validation/folds/folds_5fold_seed42.jsonl

python scripts/select_fixed_epoch.py \
  --preselected-epoch 3 \
  --reason "2,000건 개발 단계에서 잠근 최종 epoch 정책" \
  --output artifacts/final_train_validation/policies/scorer_epoch3.json

python scripts/orchestrate_scorer_training.py \
  --config configs/scorer_qlora.yaml \
  --data-config configs/data_final_combined.yaml \
  --experiment-id qwen3_scorer_final_2400 \
  --folds-file artifacts/final_train_validation/folds/folds_5fold_seed42.jsonl \
  --epoch-policy artifacts/final_train_validation/policies/scorer_epoch3.json \
  --model-revision ${MODEL_REVISION} \
  --seed 42 --seed 1337 --seed 2026
```

위 orchestration은 여전히 기본 plan-only다. registry와 경로를 검토한 뒤에만 별도
`--execute` 실행을 허용한다. 현재 저장소에는 이 조건부 workflow의 **정적 코드만** 있으며,
규정 승인·2,400건 artifact 생성·fold 학습·성능 승격은 아직 완료되지 않았다.

## 5. CPU 기준선

```bash
python scripts/run_baselines.py \
  --config configs/data.yaml \
  --baseline-config configs/baselines.yaml \
  --folds artifacts/folds/folds_5fold_seed42.jsonl
```

각 모델에 대해 train OOF와 validation prediction을 별도로 만든다.

```text
global_mean
prompt_mean
surface_ols
tfidf_ridge
```

TF-IDF alpha 등 hyperparameter는 공식 validation을 보며 고르지 않는다. 필요하면 train 내부 nested CV를 추가한다.

각 CPU baseline은 OOF를 만든 fold estimator들을 `*_foldN.joblib`로 보존하고,
`*_ensemble.json`에 파일 hash와 scorer signature를 기록한다. validation 예측도 같은
fold 모델들의 등가중 평균이므로 OOF와 배포 scorer 계약이 연결된다. label 없는 입력에
다시 적용할 때:

```bash
python scripts/predict_baseline.py \
  --input 데이터셋/test.jsonl \
  --model artifacts/predictions/cpu_baselines/tfidf_ridge_ensemble.json \
  --output artifacts/predictions/tfidf_ridge_test.jsonl
```

Joblib 파일은 hash를 확인한 뒤에만 적재한다.

### Nested-CV TF-IDF 후보

고정 TF-IDF와 별개로, 각 outer fold의 학습 부분 안에서만 alpha와 feature 상한을
선택하는 후보가 구현되어 있다.

```bash
python scripts/run_nested_tfidf.py \
  --config configs/tfidf_nested.yaml \
  --data-config configs/data.yaml \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --output-dir artifacts/predictions/nested_tfidf_v1 \
  --model-name nested_tfidf_v1
```

`nested_tfidf_oof.jsonl`은 완전한 outer OOF이고,
`nested_tfidf_ensemble.json`은 각 outer-train에서 선택·재학습한 fold model의 target
등가중 ensemble이다. 선택 로그와 코드·설정·모델 hash가 scorer signature에 포함된다.
실제 OOF 비교에서 고정 TF-IDF보다 나을 때만 이후 stacker의 `tfidf` source를 이
artifact로 교체한다.

## 6. prediction 평가

```bash
python scripts/evaluate_predictions.py \
  --config configs/data.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_validation.jsonl \
  --pred artifacts/predictions/cpu_baselines/tfidf_ridge_validation.jsonl \
  --strict
```

두 모델의 paired bootstrap 비교:

```bash
python scripts/bootstrap_compare.py \
  --gold 데이터셋/글쓰기채점능력평가2026_validation.jsonl \
  --candidate artifacts/predictions/candidate.jsonl \
  --baseline artifacts/predictions/baseline.jsonl \
  --output artifacts/reports/candidate_vs_baseline.json
```

## 7. OOF calibration

```bash
python scripts/fit_calibrator.py \
  --config configs/data.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --pred artifacts/predictions/cpu_baselines/tfidf_ridge_oof.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --output artifacts/calibrators/tfidf_affine.json \
  --calibrated-output artifacts/predictions/tfidf_oof_calibrated.jsonl
```

`--fit-source`는 `oof` 이외의 값을 허용하지 않는다. 최종 배포용 calibrator는
전체 raw OOF에 fit하지만, 성능 보고용 `--calibrated-output`은 fold를 한 번 더
hold-out하는 cross-fit으로만 만든다. 같은 OOF 행으로 fit한 뒤 같은 행을 transform한
값을 OOF 성능으로 보고하지 않는다. test/validation label에 calibrator를 직접 fit하지 않는다.
또한 `--pred` 옆의 `.manifest.json`에 gold·fold·prediction 해시와 scorer signature가
정확히 연결되어 있지 않으면 calibration 자체를 거부한다.

## 8. Qwen3 scorer 학습

먼저 `configs/scorer_qlora.yaml`의 `model.revision`을 공식 model revision SHA로 채운다. `main` 같은 이동 가능한 revision은 최종 run에 사용하지 않는다.

```bash
python -m src.train.train_scorer \
  --config configs/scorer_qlora.yaml \
  --data-config configs/data.yaml \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --fold 0 \
  --run-id qwen3_scorer_f0_s42
```

기본값은 network를 사용하지 않는 `local_files_only=True`다. 모델을 미리 target 환경의 cache에 준비한다. 개발용 최초 다운로드가 정말 필요할 때만 `--allow-download`를 명시한다.

fold 0–4를 각각 학습한다. 각 run은 `epoch_1/`부터 `epoch_N/`까지 adapter,
scoring head, tokenizer, head config, 해당 epoch OOF를 한 디렉터리에 저장한다.
`best_rmse.json`과 `best_spearman.json`은 checkpoint를 가리키는 진단용 포인터다.

복수 fold×seed 실행은 개별 명령을 손으로 복사하지 말고 signed registry로 계획한다.
기본 동작은 plan-only이며 `--execute`를 붙이기 전에는 학습하지 않는다.

```bash
python scripts/select_fixed_epoch.py \
  --preselected-epoch 3 \
  --reason "outer fold 학습 전에 고정한 epoch" \
  --output artifacts/policies/scorer_epoch3.json

python scripts/orchestrate_scorer_training.py \
  --experiment-id qwen3_scorer_v1 \
  --folds-file artifacts/folds/folds_5fold_seed42.jsonl \
  --epoch-policy artifacts/policies/scorer_epoch3.json \
  --model-revision ${MODEL_REVISION} \
  --seed 42 --seed 1337 --seed 2026
```

plan과 registry를 검토한 뒤 같은 명령에 `--execute`를 붙인다. 중단 산출물을 지우지 않고
격리해 다시 시작하려면 `--retry-partial`을 명시한다. 완료 후에는 다음 검사를 통과시킨다.

```bash
python scripts/validate_run_registry.py \
  --registry artifacts/models/qwen3_scorer_v1/registry.json \
  --require-complete \
  --output artifacts/reports/qwen3_scorer_v1_registry.json
```

중요: outer fold의 gold로 고른 best epoch를 그 fold의 OOF 예측으로 쓰면 epoch-selection
누수가 생긴다. OOF 비교에는 inner CV로 미리 고른 **동일한 고정 epoch**를 모든 fold에
사용한다. best 포인터는 탐색 진단과 최종 재학습 후보 확인용이다.

고정 epoch의 다섯 fold 예측을 ID 기준으로 병합한다.

```bash
python scripts/build_oof.py \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_42__fold_0/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_42__fold_1/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_42__fold_2/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_42__fold_3/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_42__fold_4/epoch_3/oof.jsonl \
  --output artifacts/predictions/qwen3_epoch3_oof.jsonl \
  --model qwen3_epoch3_five_fold
```

병합기는 각 입력 파일의 ID가 정확히 하나의 held-out fold에만 속하는지 검사하고,
다섯 checkpoint의 adapter·head·tokenizer 해시와 precision을 하나의 scorer signature로
묶는다. precision은 CLI 자기 선언이 아니라 각 checkpoint가 학습 시 기록한 값으로만
결정한다. 이 sidecar가 없는 파일은 calibrator 입력으로 사용할 수 없다.

여러 seed를 최종 후보로 쓸 때는 seed별 5-fold OOF를 먼저 각각 만든 뒤 합친다.

```bash
# 아래 블록을 SEED=42, 1337, 2026으로 각각 실행한다.
SEED=42
python scripts/build_oof.py \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_${SEED}__fold_0/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_${SEED}__fold_1/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_${SEED}__fold_2/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_${SEED}__fold_3/epoch_3/oof.jsonl \
  --pred artifacts/models/qwen3_scorer_v1/seed_${SEED}__fold_4/epoch_3/oof.jsonl \
  --output artifacts/predictions/qwen_seed${SEED}_oof.jsonl \
  --model qwen3_epoch3_seed${SEED}

python scripts/build_seed_ensemble_oof.py \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --source 42=artifacts/predictions/qwen_seed42_oof.jsonl \
  --source 1337=artifacts/predictions/qwen_seed1337_oof.jsonl \
  --source 2026=artifacts/predictions/qwen_seed2026_oof.jsonl \
  --output artifacts/predictions/qwen_three_seed_oof.jsonl \
  --model qwen3_epoch3_three_seed
```

이 OOF의 scorer signature는 세 seed×다섯 fold의 15개 checkpoint 전체를 묶는다. 따라서
target에서 `predict_scorer.py`에도 정확히 같은 15개 checkpoint와 precision을 넘겨야 한다.
시간 예산상 15개가 불리하면 seed 분산 보고만 남기고, 사전에 정한 한 seed 또는 검증된
adapter 축소안을 사용한다.

Qwen OOF calibrator 생성:

```bash
python scripts/fit_calibrator.py \
  --config configs/data.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --pred artifacts/predictions/qwen3_epoch3_oof.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --output artifacts/calibrators/qwen3_epoch3_affine.json \
  --calibrated-output artifacts/predictions/qwen3_epoch3_oof_crossfit_calibrated.jsonl
```

### Pairwise/soft-rank Spearman ablation

기본 P0 설정은 재현 가능한 scalar+ordinal 기준선을 만들기 위해 pairwise loss를 끈다.
pairwise 실험에서는 다음 두 값을 함께 바꾼다.

```yaml
training:
  micro_batch_size: 2
loss:
  pairwise_weight: 0.1
```

이때 trainer는 같은 prompt이면서 실제 점수 차이가 있는 두 글만 묶는 sampler를 자동으로
사용한다. batch size가 2가 아니거나 epoch 전체 pair 수가 0이면 조용히 진행하지 않고
실패한다. L40S 메모리와 OOF RMSE/Spearman을 확인하기 전에는 이 설정을 기본값으로 승격하지 않는다.

soft-rank는 같은 문항의 여러 글을 한 micro-batch로 묶어 예측 percentile과 gold
percentile을 맞춘다. 문항이 섞인 무작위 batch에는 적용하지 않는다.

```yaml
training:
  micro_batch_size: 3  # 실제 L40S 메모리에 맞춰 2 이상
loss:
  pairwise_weight: 0.0
  soft_rank_weight: 0.05
  soft_rank_temperature: 0.25
```

pairwise와 soft-rank는 동시에 켤 수 있다. 두 경우 모두 fixed fold/epoch/seed OOF에서
RMSE 비열화 없이 Spearman이 개선되는지 bootstrap으로 확인한 뒤 승격한다.

## 9. scorer checkpoint 재적재와 순차 ensemble 추론

`predict_scorer.py`는 label 없는 JSONL도 읽는다. 한 번에 checkpoint 하나만 GPU에 올리고
순차 실행한 뒤 연속 점수를 평균하므로 단일 GPU 제약을 지킨다. 아래 예시는 각 fold의
고정 epoch를 validation/test에 ensemble하고, 전체 raw OOF로 fit한 calibrator를 적용한다.

```bash
python scripts/predict_scorer.py \
  --input 데이터셋/글쓰기채점능력평가2026_validation.jsonl \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_0/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_1/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_2/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_3/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_4/epoch_3 \
  --calibrator artifacts/calibrators/qwen3_epoch3_affine.json \
  --precision 4bit \
  --batch-size 1 \
  --model-name qwen3_epoch3_five_fold \
  --output artifacts/predictions/qwen3_epoch3_validation_calibrated.jsonl
```

loader는 다음을 복원·검증한다.

- manifest/head config에 기록된 Qwen model ID와 고정 revision SHA
- 현재 scoring prompt와 학습 당시 prompt의 SHA256 일치
- local tokenizer, PEFT adapter, shared projection, 세 trait head
- projection/dropout/trait별 direct–ordinal blend 설정
- 4-bit 또는 BF16 precision 선택
- calibrator가 정확히 같은 checkpoint 집합·precision·scorer name의 OOF에서 fit됐는지 확인

학습 중 저장된 OOF는 현재 QLoRA backbone의 4-bit forward 결과다. 따라서 그 calibrator를
`--precision bf16` 추론에 재사용하면 signature 검사에서 거부된다. BF16을 최종 후보로
비교하려면 각 held-out fold를 BF16으로 다시 예측해 별도의 provenance/signature와
calibrator를 만들어야 한다.

출력은 score-only 내부 canonical JSONL이다. 이후 grounded rationale 생성과
`finalize_predictions.py` 또는 제출 엔진을 거치기 전에는 최종 대회 제출 형식이 아니다.

### 동일 checkpoint-set precision 비교

4bit에서 학습한 동일 fixed-epoch 다섯 checkpoint를 추론 precision만 바꾸어 held-out
OOF로 다시 계산한다. 서로 다른 checkpoint나 epoch를 비교 파일로 넣으면 거부된다.

```bash
python scripts/build_precision_oof.py \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_0/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_1/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_2/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_3/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_4/epoch_3 \
  --precision bf16 \
  --model qwen3_epoch3 \
  --output artifacts/predictions/qwen3_epoch3_bf16_oof.jsonl

python scripts/compare_precisions.py \
  --config configs/precision_comparison.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --candidate artifacts/predictions/qwen3_epoch3_bf16_oof.jsonl \
  --baseline artifacts/predictions/qwen3_epoch3_4bit_oof.jsonl \
  --output artifacts/reports/qwen3_epoch3_bf16_vs_4bit.json
```

승격 gate는 두 지표의 허용 비열화 한도를 모두 만족하고 RMSE 또는 Spearman 중 적어도
하나에서 paired bootstrap 근거가 있을 때만 candidate를 승인한다. BF16 peak memory와
400건 시간은 별도의 L40S smoke test에서 확인한다.
승격 후 calibrator와 target prediction의 `model-name`은 OOF manifest에 기록된
`<model>:<precision>` scorer name과 정확히 같아야 한다.
복수 seed 후보는 각 seed에서 `build_precision_oof.py`를 만든 뒤
`build_seed_ensemble_oof.py`로 합치면 동일 precision의 전체 checkpoint signature가 유지된다.

## 10. 다중 source OOF simplex stacking

두 개 이상의 base OOF 각 행은 먼저 독립적으로 provenance 검증을 통과해야 한다. trait별로
`sum(w)=1`, `w>=0`인 MSE 최소 simplex weight를 학습하고,
meta fold에서 weight와 affine/prompt calibrator를 다시 fit해 cross-fitted 성능을 만든다.

```bash
python scripts/fit_stacker.py \
  --config configs/stacker.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --source qwen=artifacts/predictions/qwen3_epoch3_oof.jsonl \
  --source tfidf=artifacts/predictions/cpu_baselines/tfidf_ridge_oof.jsonl \
  --output artifacts/stackers/qwen_tfidf_simplex_v1.json \
  --crossfit-output artifacts/predictions/qwen_tfidf_simplex_v1_crossfit.jsonl \
  --report artifacts/reports/qwen_tfidf_simplex_v1.json \
  --model-name qwen_tfidf_simplex_v1
```

validation/test에서 두 base prediction을 같은 alias로 결합한다.
Qwen source는 stacker 자체가 calibration을 포함하므로 `predict_scorer.py`에서
`--calibrator`를 빼고 만든 **raw** prediction이어야 한다. Calibrated Qwen 파일은
다른 scorer signature를 가지므로 적용기가 자동으로 거부한다.

```bash
python scripts/predict_scorer.py \
  --input 데이터셋/글쓰기채점능력평가2026_validation.jsonl \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_0/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_1/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_2/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_3/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_4/epoch_3 \
  --precision 4bit \
  --batch-size 1 \
  --model-name qwen3_epoch3_five_fold \
  --output artifacts/predictions/qwen3_epoch3_validation_raw.jsonl
```

```bash
python scripts/apply_stacker.py \
  --stacker artifacts/stackers/qwen_tfidf_simplex_v1.json \
  --source qwen=artifacts/predictions/qwen3_epoch3_validation_raw.jsonl \
  --source tfidf=artifacts/predictions/cpu_baselines/tfidf_ridge_validation.jsonl \
  --output artifacts/predictions/qwen_tfidf_simplex_v1_validation.jsonl
```

적용기는 두 source의 scorer signature, input hash, ID 집합, prompt 번호를 모두
대조한다. 이 보고의 범위는 `level1_meta_crossfit_not_fully_nested`다. meta-held-out
행의 weight/calibration 직접 누수는 막지만, Qwen 비용상 base scorer까지 다시 학습하는
완전 nested CV는 아니다. 따라서 fold weight 변동과 paired bootstrap을 확인한 뒤에만
stacker를 승격한다. 이 crossfit sidecar는 별도 `oof_level`로 표시되며, 다시 base OOF처럼
입력해 2단 stacking/calibration하는 것은 코드에서 금지한다.

### anchor/KNN 후보 분기

각 fold checkpoint의 post-projection hidden을 사용한다. OOF fold의 query와 anchor bank는
같은 checkpoint 공간에서 비교하되, 해당 held fold의 모든 label을 bank에서 제외한다.
reference와 query embedding 역할은 CLI에서 반드시 구분한다.
`configs/anchor_knn.yaml`의 k와 temperature는 outer OOF label을 보기 전에 고정한다.
동일 OOF에서 여러 k를 고르고 그 성능을 보고하는 방식은 금지한다. k 자동 선택은
추가 inner-trained scorer checkpoint가 있을 때만 별도 nested 실험으로 수행한다.

```bash
# fold 0 예시. fold 1~4도 각각 만든다.
python scripts/extract_embeddings.py \
  --input 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_0/epoch_3 \
  --role reference --precision 4bit --batch-size 1 \
  --output artifacts/embeddings/train_fold0.npz

python scripts/build_anchor_oof.py \
  --config configs/anchor_knn.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --embedding 0=artifacts/embeddings/train_fold0.npz \
  --embedding 1=artifacts/embeddings/train_fold1.npz \
  --embedding 2=artifacts/embeddings/train_fold2.npz \
  --embedding 3=artifacts/embeddings/train_fold3.npz \
  --embedding 4=artifacts/embeddings/train_fold4.npz \
  --oof-output artifacts/predictions/anchor_oof.jsonl \
  --anchor-bank artifacts/anchors/qwen_anchor_bank.npz \
  --diagnostics artifacts/reports/anchor_oof_neighbors.jsonl \
  --report artifacts/reports/anchor_oof.json \
  --model-name qwen_anchor_v1
```

target split도 각 fold checkpoint로 `--role query` embedding을 만든 후 적용한다.

```bash
for FOLD in 0 1 2 3 4; do
  python scripts/extract_embeddings.py \
    --input 데이터셋/test.jsonl \
    --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_${FOLD}/epoch_3 \
    --role query --precision 4bit --batch-size 1 \
    --output artifacts/embeddings/test_fold${FOLD}.npz
done

python scripts/predict_anchor.py \
  --input 데이터셋/test.jsonl \
  --anchor-bank artifacts/anchors/qwen_anchor_bank.npz \
  --embedding 0=artifacts/embeddings/test_fold0.npz \
  --embedding 1=artifacts/embeddings/test_fold1.npz \
  --embedding 2=artifacts/embeddings/test_fold2.npz \
  --embedding 3=artifacts/embeddings/test_fold3.npz \
  --embedding 4=artifacts/embeddings/test_fold4.npz \
  --output artifacts/predictions/anchor_test.jsonl \
  --diagnostics artifacts/reports/anchor_test_neighbors.jsonl \
  --model-name qwen_anchor_v1
```

새 scorer checkpoint에는 학습 train/fold hash가 들어간다. 이 계약이 없는 구형 checkpoint는
anchor reference로 사용하지 않는다. anchor OOF가 유효하면 `fit_stacker.py`에
`--source anchor=...`를 세 번째 source로 추가한다. 일반 simplex는 2개 이상 source를
지원하지만, weight 안정성과 bootstrap을 통과하지 않은 source는 자동으로 추가되지 않는다.

### assessment-question logits 후보 분기

18개 고정 rubric 질문마다 자유생성 대신 `A~E` 단일-token 제한 logits를 읽는다.
answer code가 정확히 한 token이 아니면 즉시 실패한다. question 수와 Ridge alpha는 각
outer-held fold를 완전히 제외한 inner CV에서만 선택한다.

```bash
python scripts/cache_assessment_logits.py \
  --config configs/assessment_questions.yaml \
  --input 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --model-revision ${MODEL_REVISION} \
  --tokenizer-revision ${MODEL_REVISION} \
  --output artifacts/assessment/train_features.npz

python scripts/fit_assessment_branch.py \
  --config configs/assessment_questions.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --cache artifacts/assessment/train_features.npz \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --output-dir artifacts/assessment/ridge_v1 \
  --model-name assessment_ridge_v1
```

test에서도 정확히 같은 revision·precision·batch·질문 계약으로 cache를 만든 뒤
`predict_assessment_branch.py`를 적용한다. 이 분기는 artifact에
`candidate_only=true`, `auto_promoted=false`가 고정된다. OOF 개선과 L40S 시간 이득이
확인된 경우에만 다중 source stacker 후보로 넣는다.

```bash
python scripts/cache_assessment_logits.py \
  --config configs/assessment_questions.yaml \
  --input 데이터셋/test.jsonl \
  --model-revision ${MODEL_REVISION} \
  --tokenizer-revision ${MODEL_REVISION} \
  --output artifacts/assessment/test_features.npz

python scripts/predict_assessment_branch.py \
  --input 데이터셋/test.jsonl \
  --cache artifacts/assessment/test_features.npz \
  --model artifacts/assessment/ridge_v1/assessment_ridge.json \
  --output artifacts/predictions/assessment_test.jsonl
```

모든 source가 각자 OOF gate를 통과했을 때만 4-source 후보를 만든다. 탈락한 source는
학습과 target 적용 양쪽 명령에서 같은 alias로 제거한다.

```bash
python scripts/fit_stacker.py \
  --config configs/stacker.yaml \
  --gold 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --source qwen=artifacts/predictions/qwen3_epoch3_oof.jsonl \
  --source tfidf=artifacts/predictions/nested_tfidf_v1/nested_tfidf_oof.jsonl \
  --source anchor=artifacts/predictions/anchor_oof.jsonl \
  --source assessment=artifacts/assessment/ridge_v1/assessment_oof.jsonl \
  --output artifacts/stackers/promoted_multisource_v1.json \
  --crossfit-output artifacts/predictions/promoted_multisource_crossfit.jsonl \
  --report artifacts/reports/promoted_multisource_v1.json \
  --model-name promoted_multisource_v1

python scripts/predict_scorer.py \
  --input 데이터셋/test.jsonl \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_0/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_1/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_2/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_3/epoch_3 \
  --checkpoint artifacts/models/qwen3_scorer_v1/seed_42__fold_4/epoch_3 \
  --precision 4bit --batch-size 1 \
  --model-name qwen3_epoch3_five_fold \
  --output artifacts/predictions/qwen_test_raw.jsonl

python scripts/predict_baseline.py \
  --input 데이터셋/test.jsonl \
  --model artifacts/predictions/nested_tfidf_v1/nested_tfidf_ensemble.json \
  --output artifacts/predictions/nested_tfidf_test.jsonl

python scripts/apply_stacker.py \
  --stacker artifacts/stackers/promoted_multisource_v1.json \
  --source qwen=artifacts/predictions/qwen_test_raw.jsonl \
  --source tfidf=artifacts/predictions/nested_tfidf_test.jsonl \
  --source anchor=artifacts/predictions/anchor_test.jsonl \
  --source assessment=artifacts/predictions/assessment_test.jsonl \
  --output artifacts/predictions/promoted_multisource_test.jsonl
```

## 11. 점수 고정 후 grounded rationale와 최종 스키마 조립

최종 점수를 먼저 확정한 뒤 근거만 생성한다. stance·근거·반론·결론·첫/끝 문장의
정확한 문자 offset을 ledger에 남기며, 모든 영역 근거가 본문의 exact evidence와 겹치지
않으면 생성 결과를 거부한다. 없는 통계·연구·숫자·인용도 grounding filter가 거부한다.

production rationale SFT용 silver는 일반 in-sample 예측이 아니라 genuine OOF score를
조건으로 생성한다.

```bash
python scripts/build_silver_rationales.py \
  --config configs/rationale_sft.yaml \
  --input 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --scores artifacts/predictions/promoted_multisource_crossfit.jsonl \
  --folds artifacts/folds/folds_5fold_seed42.jsonl \
  --model-revision ${MODEL_REVISION} \
  --output artifacts/rationale/silver_accepted.jsonl \
  --rejected-output artifacts/rationale/silver_rejected.jsonl \
  --report artifacts/reports/rationale_silver.json

python -m src.train.train_rationale \
  --config configs/rationale_sft.yaml \
  --input 데이터셋/글쓰기채점능력평가2026_train.jsonl \
  --silver artifacts/rationale/silver_accepted.jsonl \
  --run-id rationale_qwen3_v1 \
  --model-revision ${MODEL_REVISION}
```

silver acceptance 수·비율 gate, prompt/evidence/grounding 코드 hash, model revision이 모두
맞아야 학습한다. 학습 train split에는 동일 evidence target을 유지하는 작은 연속 score
jitter가 결정론적으로 추가되어 OOF 점수와 최종 stacked 점수 사이의 미세한 이동을 견딘다.
validation에는 jitter를 넣지 않는다.

최종 score 파일에 조건화해 adapter 근거를 만든다. 두 번 모두 실패하면 exact-evidence
결정론 fallback으로 내려간다.

```bash
python scripts/generate_rationales.py \
  --input 데이터셋/test.jsonl \
  --scores artifacts/predictions/promoted_multisource_test.jsonl \
  --checkpoint artifacts/models/rationale_qwen3_v1/best.json \
  --precision 4bit --max-attempts 2 \
  --output artifacts/rationale/test_grounded.jsonl \
  --report artifacts/reports/test_rationale_generation.json

python scripts/finalize_predictions.py \
  --input 데이터셋/test.jsonl \
  --scores artifacts/predictions/promoted_multisource_test.jsonl \
  --rationales artifacts/rationale/test_grounded.jsonl \
  --output artifacts/final/final.jsonl \
  --ledger artifacts/final/evidence.jsonl \
  --bare-output artifacts/final/submission.jsonl \
  --model-name qwen3_14b_final_v1
```

SFT가 승격되지 않은 경우에도 파싱 0점을 막는 결정론 fallback만으로 최종본을 만들 수 있다.

```bash
python scripts/finalize_predictions.py \
  --input 데이터셋/test.jsonl \
  --scores artifacts/predictions/qwen_tfidf_simplex_v1_test.jsonl \
  --output artifacts/final/qwen_tfidf_simplex_v1_final.jsonl \
  --ledger artifacts/final/qwen_tfidf_simplex_v1_evidence.jsonl \
  --bare-output artifacts/final/qwen_tfidf_simplex_v1_bare.jsonl \
  --model-name qwen_tfidf_simplex_v1_grounded_fallback_v1
```

- `output`은 ID와 model을 포함한 내부 감사용 최종 JSONL이다.
- `bare-output`의 각 행은 대회 예시와 같은 세 영역 `score+rationale` 객체뿐이다.
- score는 source prediction의 float를 반올림·재생성하지 않고 그대로 복사한다.
- 모든 행은 strict serializer→parser 왕복과 score 동일성 검사를 통과해야 기록된다.
- ledger는 내부 감사용이며 최종 제출에는 포함하지 않는다.

label이 있는 validation 최종본은 완화 parser가 아니라 final schema로 다시 평가한다.

```bash
python scripts/evaluate_predictions.py \
  --gold 데이터셋/글쓰기채점능력평가2026_validation.jsonl \
  --pred artifacts/final/qwen_tfidf_simplex_v1_final.jsonl \
  --final-schema
```

fallback은 hallucination 방지와 end-to-end 안전성을 위한 P0이다. adapter는 사람 blind
평가에서 fallback보다 우세할 때만 승격하며, 어느 경로든 동일한 score-preservation 계약을 유지한다.

candidate와 fallback이 **동일한 점수**를 가진 final JSONL을 준비한 뒤, 문항·점수대 균형
100건 A/B 검수 pack을 만든다. review 파일에는 어느 쪽이 candidate인지 노출되지 않는다.

```bash
python scripts/build_rationale_review_pack.py \
  --input 데이터셋/글쓰기채점능력평가2026_validation.jsonl \
  --candidate artifacts/final/adapter_validation.jsonl \
  --baseline artifacts/final/fallback_validation.jsonl \
  --sample-size 100 --seed 2026 \
  --output artifacts/reviews/rationale_blind_100.jsonl \
  --key-output artifacts/reviews/rationale_blind_100_key.jsonl
```

key 파일은 검수자에게 주지 않고 평가 완료 뒤에만 합친다.

동일한 review pack을 로컬 GGUF judge로도 평가할 수 있다. 최종 비교에 사용할 GGUF를
미리 준비하고 파일 hash를 고정한다. 과제 공지에 적힌 Judge는
`Qwen3.6-35B-A3B, Q4_K_M GGUF`이므로 동일 artifact를 합법적으로 확보할 수 있으면 우선
맞춘다. 그렇지 않은 GGUF의 결과는 대회 Judge 재현값이 아니라 로컬 보조 신호로만
취급한다. 실제 모델 가용성·라이선스·L40S 적합성은 대상 환경에서 별도로 확인한다.

```bash
python scripts/judge_rationales_local.py \
  --config configs/rationale_judge.yaml \
  --review-pack artifacts/reviews/rationale_blind_100.jsonl \
  --model /path/to/pinned-rationale-judge.Q4_K_M.gguf \
  --output artifacts/reviews/rationale_blind_100_judgments.jsonl
```

judge CLI에는 의도적으로 assignment key 인자가 없다. review pack manifest와 본문 hash를
검증하지만 key 파일은 열지 않는다. 각 행은 attempt마다 짝지은 deterministic seed로 원래
AB와 뒤집은 BA 두 순서를 모두 평가하고, 결과를 원래 option A/B로 되돌린다. 다섯 기준
`grounding`, `specificity`, `trait_separation`, `score_consistency`, `overall`을 각각
`A/B/TIE`로 strict JSON 판정하며, 두 순서가 불일치하면 해당 기준을 `TIE`와
`unstable=true`로 보수적으로 처리한다. 형식 오류 재시도는 schema를 완화하지 않는다.

판정이 모두 끝난 뒤에만 별도 요약 CLI가 hidden key를 읽는다.

```bash
python scripts/summarize_rationale_judge.py \
  --judgments artifacts/reviews/rationale_blind_100_judgments.jsonl \
  --key artifacts/reviews/rationale_blind_100_key.jsonl \
  --output artifacts/reviews/rationale_blind_100_summary.json
```

요약기는 judgment manifest에 기록된 expected key SHA256과 실제 key를 맞춘 뒤
candidate/baseline win, tie, unstable 비율을 criterion별로 집계한다. judge 결과와
sidecar에는 GGUF·prompt·generation 설정·judge 코드·입력의 hash가 남는다. 이 기능은
**정적 구현 완료**지만 로컬 GGUF 실행, 후보 우세 확인, 사람 blind 평가와 승격은 아직
완료되지 않았다. 자동 judge 결과만으로 사람 평가를 대체하거나 adapter를 자동 승격하지
않는다.

## 12. offline 제출 엔진·패키징·L40S 검증

`configs/inference_l40s.yaml`은 일부러 placeholder와 null revision을 포함하므로 그대로는
실행되지 않는다. 승격된 fixed checkpoint, 40자리 base revision, 선택한 calibrator/stacker,
rationale mode를 채운다. schema v2 엔진은 보수적인 `qwen_ensemble`과, Qwen에
baseline·anchor·assessment 중 승격된 한 개 이상을 결합하는
`qwen_multisource_stacker`를 지원한다. Artifact가 있는 source만 고유 alias를 설정한다.
Stacker가 raw Qwen OOF로 학습됐다면 `scoring.qwen.calibrator`는 `null`이어야 한다.
공식 SDK가 공개되면 `run_submission.py`의 입출력 경계만 adapter로 감싸고 내부
score/rationale 계약은 바꾸지 않는다.

```yaml
scoring:
  mode: qwen_multisource_stacker
  baseline: {artifact: artifacts/predictions/nested_tfidf_v1/nested_tfidf_ensemble.json}
  anchor: {artifact: artifacts/anchors/qwen_anchor_bank.npz}
  assessment: {artifact: artifacts/assessment/ridge_v1/assessment_ridge.json}
  stacker:
    artifact: artifacts/stackers/promoted_multisource_v1.json
    source_aliases:
      qwen: qwen
      baseline: tfidf
      anchor: anchor
      assessment: assessment
```

Anchor를 배포할 때는 Qwen checkpoint가 bank와 같은 seed의 fold별 1개씩이어야 한다.
Assessment를 배포할 때는 feature artifact의 model/tokenizer commit이 패키지 Qwen commit과
같아야 한다. 모든 source signature와 stacker source order는 GPU 적재 전에 검증된다.

```bash
python scripts/run_submission.py \
  --config configs/inference_l40s.yaml \
  --input 데이터셋/test.jsonl \
  --output artifacts/final/run_submission.jsonl \
  --ledger artifacts/final/run_submission_ledger.jsonl \
  --bare-output artifacts/final/submission.jsonl
```

첫 성공 환경에서 `pip freeze`를 그대로 쓰지 말고, 모든 runtime dependency가 정확한
`name==version`인 `requirements-lock.txt`를 확정한다. 모델·코드·데이터 관련 license와
notice 파일도 준비한 뒤 새 package 디렉터리를 만든다.

```bash
python scripts/package_submission.py \
  --config configs/inference_l40s.yaml \
  --output-dir artifacts/packages/submission_v1 \
  --requirements-lock requirements-lock.txt \
  --license-file LICENSE \
  --license-file THIRD_PARTY_NOTICES.md \
  --hf-home /path/to/pinned/huggingface/cache
```

패키징기는 symlink를 일반 파일로 materialize하고, source closure·checkpoint·tokenizer·
adapter·head·base cache·lock·license를 복사한 뒤 모든 payload SHA256과 package signature를
기록한다. 기존 package나 source artifact는 덮어쓰지 않는다.
`--hf-home`에는 다른 모델이나 개인 cache가 섞이지 않은, 해당 pinned Qwen revision만
준비된 전용 cache 디렉터리를 사용한다.

단일 L40S에서 네트워크를 차단한 400건 smoke test를 수행한다.

```bash
python scripts/smoke_test_submission.py \
  --config artifacts/packages/submission_v1/inference_l40s.yaml \
  --input 데이터셋/test.jsonl \
  --report artifacts/reports/submission_v1_l40s_smoke.json \
  --expected-count 400 \
  --offline --strict --verify-determinism --require-package-manifest
```

이 검사는 strict parse 100%, ID/순서 보존, score byte 동일성, 입력·artifact 불변성,
네트워크 차단, dependency lock 일치, elapsed time, peak allocated/reserved CUDA memory를
보고한다. 이 환경에서는 사용자 지시에 따라 위 명령을 실행하지 않았다.

## 13. 정적 구현 이후 남은 실환경 작업

현재 남은 것은 새 기능의 정적 코드 작성이 아니라 대상 환경의 증거 수집과 승격 결정이다.

- 공식 제출 SDK/schema에서 소수 score와 entrypoint 계약 확인
- validation label의 최종 2,400건 재학습 허용 여부 확인; 허용 시 명시 승인 플래그로
  결합 artifact를 만들고 전용 fold·OOF·calibrator·stacker를 실제 재학습
- model/tokenizer 40자리 revision과 dependency lock 확정
- 고정/nested TF-IDF, LOPO, Qwen fold×seed, precision, anchor, assessment OOF 실제 실행
- paired bootstrap·slice·weight 안정성으로 후보 승격/탈락
- silver 생성 후 로컬 GGUF AB/BA judge와 국어 전문가 표본 blind 평가, 결과 기반 승격
- 단일 L40S 400건 offline smoke 및 시간·VRAM 확인
- 최종 제출 파일 재읽기, strict schema와 validation metric 최종 보고

정량·자원 증거 없이 assessment, anchor, soft-rank, BF16, rationale adapter를 기본 경로로
자동 승격하지 않는다.

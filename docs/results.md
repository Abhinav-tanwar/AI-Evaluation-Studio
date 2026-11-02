# Results and verification

Recorded on November 2, 2026 using Python 3.12.14 on macOS ARM64. Full reports retain source commit, implementation fingerprint, dataset fingerprint, configurations, outputs, citations, retrieved passages, error types, and timings.

| Split | Questions | Answerable | Unanswerable | Baseline exact match | Candidate exact match |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Development | 30 | 24 | 6 | 27/30 | 27/30 |
| Holdout | 20 | 16 | 4 | 17/20 | 18/20 |

The split was assigned during dataset authoring before the first recorded development evaluation. Defaults were unchanged before the first holdout evaluation. The holdout is now public; reuse a fresh reserved set for future tuning. This is an authored demonstration benchmark, not an independently sampled or independently audited test set.

## What improved

Candidate source recall increased from 22/24 to 24/24 in development and 15/16 to 16/16 in holdout. Candidate exact matching improved by one holdout case. Candidate abstention was correct for 5/6 development and 3/4 holdout unanswerable questions. Citation validity and verbatim support were 100% among nonabstaining reference responses because the adapter copies a cited passage. Those scores do not prove the answer addressed the question.

Two holdout failures remain:

* `q15`: the standard support schedule retrieves a response time passage above the expected source.
* `q49`: an adversarial database password question retrieves a database migration passage instead of abstaining.

In development, `q16` retrieves the correct evidence below rank 1 but still produces the wrong answer. This demonstrates why retrieval recall and answer quality need separate checks.

## Metric definitions

* Exact match: casefolded word sequence equality with the expected answer, plus the correct abstention flag, averaged over every case. Provider errors count as failures.
* Source recall: fraction of expected passage IDs retrieved, averaged only over answerable questions. Unanswerable cases have no expected sources and are excluded.
* Citation validity: at least one citation and every citation in retrieved context, among nonabstaining responses. Provider errors receive zero.
* Verbatim support: a nonempty normalized answer is contained in at least one cited retrieved passage. It does not check whether every extra citation is relevant and does not evaluate semantic faithfulness.
* Correct abstention: fraction of unanswerable questions on which the adapter abstains. False abstentions on answerable questions are penalized through exact match.
* p95 latency: nearest rank percentile of per case retrieval, adapter, and scoring time. It excludes process startup and storage. Pipelines execute sequentially; no latency speedup claim is made.

Reference mode makes zero provider calls, reports zero provider charges, and leaves token counts null. Live model cost remains unknown unless both rates are configured. Infrastructure and development costs are outside this estimate.

## Reproduce

```sh
python -m studio.cli --split development --output runtime/development.json --gate
python -m studio.cli --split holdout --output runtime/holdout.json --gate
```

Compare score fields and dataset/implementation fingerprints with the committed reports. UUIDs, timestamps, Git state, and timings will differ. The reports reference the engine source commit; the working tree flag is true because documentation and report artifacts were being added. `implementation_sha256` fingerprints the actual Python source independently of that flag. No retrieval thresholds were changed after inspecting holdout outcomes.

The gate requires no exact match or source recall regression, no candidate provider errors, and p95 at most 5 seconds. It is a regression check, not an absolute quality certificate, significance test, or security assurance.

## Checks performed

23 automated tests passed with tracking dependencies installed. Python lint and formatting passed; the TypeScript production build passed. A nonroot container completed a reference run against PostgreSQL and retained the run after restarting the service; see [container verification](../reports/container-verification.json). Browser checks covered comparison creation, case inspection, human review, search with no matches, and layouts at 320 px and 390 px without horizontal overflow. Screenshots show actual application output.

OpenAI integration was tested with a structured response test double. No paid live model benchmark was run. The application currently compares retrieval configurations with the same answer adapter; it does not compare multiple model providers, use an LLM judge, or perform semantic faithfulness evaluation.

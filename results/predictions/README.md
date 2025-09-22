# SWE-Bench Predictions Template

This file shows the format for generating predictions.jsonl for the full SWE-Bench dataset evaluation.

## Format

Each line should be a JSON object with:
```json
{
  "instance_id": "repo_owner__repo_name-issue_number",
  "model_name_or_path": "your-model-name",
  "model_patch": "the patch content as a string"
}
```

## Example

```json
{"instance_id": "django__django-10097", "model_name_or_path": "enhanced-swe-bench-submission", "model_patch": "diff --git a/django/db/models/fields/__init__.py b/django/db/models/fields/__init__.py\nindex 1234567..abcdefg 100644\n--- a/django/db/models/fields/__init__.py\n+++ b/django/db/models/fields/__init__.py\n@@ -1,3 +1,3 @@\n class CharField:\n-    def __hash__(self):\n+    def __hash__(self):\n         return hash((self.__class__, self.name))"}
```

## Generation Command

After creating predictions.jsonl with all tasks:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path predictions.jsonl \
    --max_workers 12 \
    --run_id full_evaluation
```

This will generate the results folder with evaluation scores.

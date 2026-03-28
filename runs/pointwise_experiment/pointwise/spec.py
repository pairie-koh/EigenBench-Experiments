"""
Pointwise TREATMENT for the pointwise vs pairwise experiment.

IMPORTANT: Run the pairwise control spec FIRST to populate the shared
response cache. This spec reuses those cached responses so the only
experimental variable is the scoring method.
"""

RUN_SPEC = {
    "name": "pointwise_experiment_pointwise",
    "verbose": True,
    "models": {
        # TODO: Replace with your chosen OpenRouter model IDs.
        # Must be IDENTICAL to the pairwise spec.
        "Model A": "PLACEHOLDER_MODEL_A",
        "Model B": "PLACEHOLDER_MODEL_B",
        "Model C": "PLACEHOLDER_MODEL_C",
        "Model D": "PLACEHOLDER_MODEL_D",
    },
    "dataset": {
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 0,
        "count": 100,
        "shuffle": True,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,
    },
    "collection": {
        "enabled": True,
        "mode": "pointwise",
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "random_judge_group",
        # Same deterministic seed as pairwise — ensures identical judge/evaluee assignments.
        "sampler_seed": 2026,
        # Same shared response cache as pairwise — ensures identical Phase 1 responses.
        "cached_responses_path": "runs/pointwise_experiment/shared_responses.jsonl",
        "evaluations_path": "runs/pointwise_experiment/pointwise/evaluations.jsonl",
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
    },
}

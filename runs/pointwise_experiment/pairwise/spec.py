"""
Pairwise CONTROL for the pointwise vs pairwise experiment.

IMPORTANT: Run this FIRST to generate cached responses, then run the
pointwise treatment spec which reuses the same responses. This ensures
the only variable is the scoring method (pairwise vs pointwise).
"""

RUN_SPEC = {
    "name": "pointwise_experiment_pairwise",
    "verbose": True,
    "models": {
        # TODO: Replace with your chosen OpenRouter model IDs.
        # Must be IDENTICAL across both pairwise and pointwise specs.
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
        "mode": "pairwise",
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "random_judge_group",
        # Shared response cache — both conditions use the same model responses.
        # Run the pairwise spec first to populate, then pointwise reuses them.
        "cached_responses_path": "runs/pointwise_experiment/shared_responses.jsonl",
        "evaluations_path": "runs/pointwise_experiment/pairwise/evaluations.jsonl",
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

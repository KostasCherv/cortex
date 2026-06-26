# tests/finetune/test_action_seeds.py
from scripts.finetune.action_seeds import ACTION_SEEDS, ACTIONS, TARGET_PER_CLASS

REQUIRED_ACTIONS = {
    "answer_direct", "answer_from_rag", "web_search",
    "asset_price", "search_finance_tools", "ask_clarifying",
}


def test_all_six_actions_present():
    assert set(ACTION_SEEDS.keys()) == REQUIRED_ACTIONS


def test_actions_list_matches_keys():
    assert set(ACTIONS) == set(ACTION_SEEDS.keys())


def test_minimum_ten_seeds_per_class():
    for action, seeds in ACTION_SEEDS.items():
        assert len(seeds) >= 10, f"{action} has only {len(seeds)} seeds (need >=10)"


def test_seed_structure():
    for action, seeds in ACTION_SEEDS.items():
        for seed in seeds:
            assert "message" in seed
            assert "rag_context" in seed
            assert seed["message"].strip(), f"Empty message in {action}"


def test_target_per_class_positive():
    assert TARGET_PER_CLASS > 0

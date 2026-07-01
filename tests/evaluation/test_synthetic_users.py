"""Regression tests — Synthetic User generation (Deliverable 14)."""

from __future__ import annotations

from evaluation.synthetic.user_generator import UserGenerator
from evaluation.synthetic.user_profile import Preference


def test_generate_100_users_deterministically() -> None:
    """100 users are generated and the same seed always produces the same result."""
    gen = UserGenerator(n_users=100, min_turns=50, max_turns=100, seed_base=42)
    users = gen.generate_all()
    assert len(users) == 100, "Must generate exactly 100 users"

    # Regenerate and compare first user
    gen2 = UserGenerator(n_users=1, min_turns=50, max_turns=100, seed_base=42)
    users2 = gen2.generate_all()
    assert users[0].name == users2[0].name, "Regeneration must be deterministic"
    assert users[0].user_id == users2[0].user_id


def test_user_ids_are_unique() -> None:
    gen = UserGenerator(n_users=100, min_turns=50, max_turns=100, seed_base=42)
    users = gen.generate_all()
    ids = [u.user_id for u in users]
    assert len(ids) == len(set(ids)), "All user IDs must be unique"


def test_turn_count_within_bounds() -> None:
    gen = UserGenerator(n_users=10, min_turns=50, max_turns=100, seed_base=42)
    users = gen.generate_all()
    for user in users:
        assert 50 <= user.total_turns <= 100, (
            f"User {user.user_id} has {user.total_turns} turns, expected 50–100"
        )


def test_each_user_has_profile_fields() -> None:
    gen = UserGenerator(n_users=5, min_turns=50, max_turns=80, seed_base=42)
    users = gen.generate_all()
    for user in users:
        assert user.name, "User must have a name"
        assert user.profile.occupation, "User must have an occupation"
        assert user.profile.workplace, "User must have a workplace"
        assert user.profile.location, "User must have a location"
        assert len(user.profile.preferences) >= 1, "User must have at least one preference"
        assert len(user.profile.projects) >= 1, "User must have at least one project"
        assert len(user.profile.relationships) >= 1, "User must have at least one relationship"
        assert len(user.profile.hobbies) >= 1, "User must have at least one hobby"
        assert len(user.profile.long_term_goals) >= 1, "User must have at least one goal"


def test_temporal_events_are_ordered() -> None:
    gen = UserGenerator(n_users=5, min_turns=80, max_turns=100, seed_base=42)
    users = gen.generate_all()
    for user in users:
        turns = [e.turn for e in user.temporal_events]
        assert turns == sorted(turns), (
            f"Temporal events for {user.user_id} must be in turn order"
        )


def test_preferences_have_editor_category() -> None:
    gen = UserGenerator(n_users=5, min_turns=80, max_turns=100, seed_base=42)
    users = gen.generate_all()
    for user in users:
        editors = [p for p in user.preferences if p.category == "editor"]
        assert len(editors) >= 1, f"User {user.user_id} must have at least one editor preference"


def test_preference_evolution_supersedes_correctly() -> None:
    gen = UserGenerator(n_users=10, min_turns=200, max_turns=300, seed_base=42)
    users = gen.generate_all()
    for user in users:
        editor_chain: list[Preference] = [p for p in user.preferences if p.category == "editor"]
        for pref in editor_chain:
            if pref.superseded_at_turn >= 0:
                # The superseded-at turn must be after the set turn
                assert pref.superseded_at_turn > pref.turn_set, (
                    f"Preference {pref!r}: superseded_at_turn must be after turn_set"
                )


def test_emotional_profile_trajectory_ordered() -> None:
    gen = UserGenerator(n_users=5, min_turns=80, max_turns=100, seed_base=42)
    users = gen.generate_all()
    for user in users:
        traj = user.emotional_profile.trajectory
        traj_turns = [t for t, _ in traj]
        assert traj_turns == sorted(traj_turns), (
            f"Emotion trajectory for {user.user_id} must be chronologically ordered"
        )


def test_conversation_turns_match_total() -> None:
    gen = UserGenerator(n_users=5, min_turns=50, max_turns=60, seed_base=42)
    users = gen.generate_all()
    for user in users:
        assert len(user.conversation_turns) == user.total_turns, (
            f"User {user.user_id}: turn list length must match total_turns"
        )


def test_to_dict_roundtrip() -> None:
    gen = UserGenerator(n_users=2, min_turns=50, max_turns=60, seed_base=42)
    users = gen.generate_all()
    for user in users:
        d = user.to_dict()
        assert d["user_id"] == user.user_id
        assert d["name"] == user.name
        assert isinstance(d["conversation_turns"], list)
        assert len(d["conversation_turns"]) == user.total_turns


def test_user_profile_to_dict_contains_required_fields() -> None:
    gen = UserGenerator(n_users=1, min_turns=50, max_turns=60, seed_base=42)
    users = gen.generate_all()
    d = users[0].profile.to_dict()
    required = [
        "user_id", "name", "occupation", "workplace", "location",
        "hobbies", "long_term_goals", "recurring_topics",
        "preferences", "projects", "relationships", "temporal_events",
        "emotional_profile", "personality_traits", "known_facts",
    ]
    for key in required:
        assert key in d, f"UserProfile.to_dict() missing key: {key}"

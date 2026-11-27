"""State ownership and publication tested against a two-session dictionary."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from hypothesis.strategies import integers, sampled_from

from state import JsonStateManager, StateVersionConflict

SESSIONS = sampled_from(("one", "two"))


class StateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "state.json"
        self.manager = JsonStateManager(self.path)
        self.oracle = {}

    def teardown(self):
        self.directory.cleanup()

    @rule(session=SESSIONS, value=integers())
    def save(self, session, value):
        draft = self.manager.load(session)
        version, _ = self.oracle.get(session, (0, {}))
        draft.task["items"] = [value]
        saved = self.manager.save(session, draft, version)
        self.oracle[session] = (version + 1, {"items": [value]})
        saved.task["items"].append(1)
        draft.task["items"].append(2)
        with pytest.raises(StateVersionConflict):
            self.manager.save(session, draft, version)

    @rule(session=SESSIONS)
    def reset(self, session):
        stale = self.manager.load(session)
        self.manager.reset(session)
        self.oracle.pop(session, None)
        with pytest.raises(StateVersionConflict):
            self.manager.save(session, stale, 0)

    @rule(session=SESSIONS, save=sampled_from((True, False)))
    def failed_publication(self, session, save):
        candidate = self.manager.load(session)
        candidate.task["items"] = [999]
        with patch.object(Path, "replace", side_effect=OSError("injected replacement failure")), pytest.raises(OSError):
            if save:
                self.manager.save(session, candidate, candidate.version)
            else:
                self.manager.reset(session)

    @rule()
    def reopen(self):
        self.manager = JsonStateManager(self.path)

    @invariant()
    def compare_committed_state(self):
        for session in ("one", "two"):
            for owner in (self.manager, JsonStateManager(self.path)):
                state = owner.load(session)
                assert (state.version, state.task) == self.oracle.get(session, (0, {}))
                state.task.setdefault("items", []).append(-1)


TestStateMachine = StateMachine.TestCase
TestStateMachine.settings = settings(max_examples=25, stateful_step_count=15, database=None, derandomize=True, deadline=None)

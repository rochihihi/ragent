from pathlib import Path

import pytest

from veripatch.domain import AgentRunState, IssueSpec
from veripatch.store import SQLiteRunStore


def test_store_round_trips_state_and_events(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    state = AgentRunState(
        run_id="run-1",
        repo_root=str(tmp_path),
        issue=IssueSpec(issue_id="1", title="Bug", description="Something is broken"),
        test_command=["python", "-m", "pytest"],
    )
    store.checkpoint(state)
    store.append_event("run-1", "created", {"value": 1})
    loaded = store.load("run-1")
    assert loaded is not None
    assert loaded.issue.title == "Bug"
    assert store.events("run-1")[0]["payload"] == {"value": 1}


def test_store_atomic_record_and_pagination(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    states = []
    for number in range(3):
        state = AgentRunState(
            run_id=f"run-{number}",
            repo_root=str(tmp_path),
            issue=IssueSpec(issue_id=str(number), title="Bug", description="Broken"),
            test_command=["python", "-m", "pytest"],
            step=number,
        )
        store.record(state, "created", {"number": number})
        store.append_event(state.run_id, "next", {"number": number})
        states.append(state)

    first = store.events("run-1", limit=1)
    second = store.events("run-1", after_sequence=first[0]["sequence"], limit=1)
    assert [first[0]["event_type"], second[0]["event_type"]] == ["created", "next"]
    assert store.load("run-2").step == 2
    assert len(store.list_runs(limit=2)) == 2
    assert len(store.list_runs(limit=2, offset=2)) == 1
    with pytest.raises(ValueError, match="Event limit"):
        store.events("run-1", limit=0)
    with pytest.raises(ValueError, match="Run limit"):
        store.list_runs(limit=0)
    with pytest.raises(ValueError, match="offset"):
        store.list_runs(offset=-1)

"""Isolated real backend for the headless Skills.vue regression check (no model calls)."""

import argparse
import json
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from veripatch.config import Settings
from veripatch.studio_api import create_studio_router
from veripatch.studio_domain import StudioSession
from veripatch.studio_store import StudioStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix=".test-skills-ui-", dir=project) as directory:
        root = Path(directory)
        repo = root / "project"
        repo.mkdir()
        bad = repo / ".agents/skills/broken"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("invalid", encoding="utf-8")
        fixture = root / "sales"
        (fixture / "scripts").mkdir(parents=True)
        (fixture / "references").mkdir()
        (fixture / "SKILL.md").write_text(
            "---\nname: sales\ndescription: >-\n  分析销售报表，\n  统计真实销售金额。\n"
            "compatibility: Python 3.11+\n---\n按需读取 references/rules.md。\n",
            encoding="utf-8",
        )
        (fixture / "scripts/analyze.py").write_text("print('not executed')", encoding="utf-8")
        (fixture / "references/rules.md").write_text("金额是每行总金额。", encoding="utf-8")
        db = root / "state.db"
        store = StudioStore(db)
        store.save(
            StudioSession(
                session_id="skills-ui",
                repo_root=str(repo),
                provider="deepseek",
                model="test",
                reasoning_effort="low",
                title="技能界面回归",
            ),
            "created",
            {},
        )
        app = FastAPI()
        app.include_router(create_studio_router(Settings(database_path=db)))

        @app.get("/provider-models")
        def models():
            return {
                name: {"selected": "test", "choices": ["test"]}
                for name in ("deepseek", "openai", "openai_official")
            }

        print("FIXTURE:" + json.dumps({"folder": str(fixture)}, ensure_ascii=True), flush=True)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=args.port,
                log_level="warning",
            )
        )

        @app.post("/_test/stop")
        def stop():
            server.should_exit = True
            return {"stopping": True}

        server.run()


if __name__ == "__main__":
    main()

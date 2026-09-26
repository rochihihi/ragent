"""Command-line interface for running and inspecting VeriPatch."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from veripatch.agent import VeriPatchAgent
from veripatch.api import create_app
from veripatch.config import Settings
from veripatch.credentials import credential_status, delete_api_key, save_api_key
from veripatch.domain import IssueSpec, RunnerKind
from veripatch.evaluation import load_tasks, run_evaluation, write_markdown_report
from veripatch.models.base import AgentModel
from veripatch.models.scripted import DiscountBugDemoModel
from veripatch.providers import create_model
from veripatch.quota import get_provider_quota
from veripatch.store import SQLiteRunStore


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model(provider: str, settings: Settings) -> AgentModel:
    return create_model(provider, settings)


async def _run_demo(settings: Settings) -> int:
    source = _project_root() / "examples" / "discount_bug"
    if not source.is_dir():
        raise RuntimeError(f"Demo repository is missing: {source}")
    workspace = _project_root() / "runs" / "demo_workspaces" / uuid4().hex
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace)
    issue = IssueSpec(
        issue_id="discount-percentage",
        title="Discount percentage is treated as a whole number",
        description=(
            "calculate_discount(100, 10) should return 90, but the implementation "
            "subtracts 10 instead of 0.10. Reproduce with the existing tests and fix it."
        ),
    )
    agent = VeriPatchAgent(DiscountBugDemoModel(), settings=settings)
    result = await agent.run(
        repo_root=workspace,
        issue=issue,
        test_command=[sys.executable, "-m", "pytest", "-q"],
        runner_kind=RunnerKind.LOCAL,
        provider="scripted-demo",
    )
    print(json.dumps(result.state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("\n--- PATCH ---")
    print(result.diff or "(no patch)")
    print(f"Demo workspace: {workspace}")
    return 0 if result.state.phase.value == "succeeded" else 1


async def _run_repository(args: argparse.Namespace, settings: Settings) -> int:
    issue = IssueSpec(
        issue_id=args.issue_id,
        title=args.title,
        description=args.description,
    )
    agent = VeriPatchAgent(_model(args.provider, settings), settings=settings)
    result = await agent.run(
        repo_root=Path(args.repo),
        issue=issue,
        test_command=[sys.executable, "-m", "pytest", *args.pytest_arg],
        runner_kind=RunnerKind(args.runner),
        provider=args.provider,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.state.phase.value == "succeeded" else 1


async def _run_eval(args: argparse.Namespace, settings: Settings) -> int:
    tasks = load_tasks(Path(args.tasks))
    records, summary = await run_evaluation(
        tasks,
        model_factory=lambda: _model(args.provider, settings),
        settings=settings,
        results_path=Path(args.output),
        provider=args.provider,
        runner=RunnerKind(args.runner),
        benchmark_kind=(
            "runtime_validation" if args.provider == "scripted-demo" else "model_benchmark"
        ),
    )
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".md")
    write_markdown_report(records, summary, report_path)
    print(summary.model_dump_json(indent=2))
    return 0 if all(record.resolved for record in records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veripatch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="Run the offline frozen repair task")

    run_parser = subparsers.add_parser("run", help="Repair a local Python repository")
    run_parser.add_argument("--repo", required=True)
    run_parser.add_argument("--issue-id", default="local-issue")
    run_parser.add_argument("--title", required=True)
    run_parser.add_argument("--description", required=True)
    run_parser.add_argument(
        "--provider", choices=["openai", "deepseek", "scripted-demo"], default="openai"
    )
    run_parser.add_argument("--runner", choices=["local", "docker"], default="docker")
    run_parser.add_argument("--pytest-arg", action="append", default=["-q"])

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a persisted run")
    inspect_parser.add_argument("run_id")

    resume_parser = subparsers.add_parser("resume", help="Resume a non-terminal run")
    resume_parser.add_argument("run_id")

    eval_parser = subparsers.add_parser("eval", help="Run JSONL evaluation tasks")
    eval_parser.add_argument("--tasks", required=True)
    eval_parser.add_argument("--output", default="results/evaluation.jsonl")
    eval_parser.add_argument("--report")
    eval_parser.add_argument(
        "--provider", choices=["openai", "deepseek", "scripted-demo"], default="openai"
    )
    eval_parser.add_argument("--runner", choices=["local", "docker"], default="docker")

    serve_parser = subparsers.add_parser("serve", help="Start the local FastAPI server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    mcp_parser = subparsers.add_parser(
        "mcp-server", help="Expose a project as a read-only MCP server over stdio"
    )
    mcp_parser.add_argument("--repo", required=True)

    auth_parser = subparsers.add_parser("auth", help="Manage provider API keys")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)
    for action in ("login", "status", "logout"):
        action_parser = auth_subparsers.add_parser(action)
        action_parser.add_argument("provider", choices=["openai", "deepseek"])

    quota_parser = subparsers.add_parser("quota", help="Inspect provider credit availability")
    quota_subparsers = quota_parser.add_subparsers(dest="quota_command", required=True)
    quota_status = quota_subparsers.add_parser("status")
    quota_status.add_argument("provider", choices=["openai", "deepseek"])
    return parser


def main() -> None:
    _configure_console()
    args = build_parser().parse_args()
    settings = Settings.from_env()
    if args.command == "auth":
        if args.auth_command == "login":
            api_key = getpass.getpass(f"{args.provider} API key: ")
            save_api_key(args.provider, api_key)
            print(f"{args.provider}: configured in system keyring")
            return
        if args.auth_command == "logout":
            removed = delete_api_key(args.provider)
            print(f"{args.provider}: {'removed' if removed else 'no keyring credential'}")
            return
        print(json.dumps({args.provider: credential_status()[args.provider]}, indent=2))
        return
    if args.command == "quota":
        snapshot = get_provider_quota(args.provider, settings)
        print(snapshot.model_dump_json(indent=2))
        raise SystemExit(0 if snapshot.error is None else 1)
    if args.command == "demo":
        raise SystemExit(asyncio.run(_run_demo(settings)))
    if args.command == "run":
        raise SystemExit(asyncio.run(_run_repository(args, settings)))
    if args.command == "eval":
        raise SystemExit(asyncio.run(_run_eval(args, settings)))
    if args.command == "resume":
        store = SQLiteRunStore(settings.database_path)
        state = store.load(args.run_id)
        if state is None:
            raise SystemExit(f"Run not found: {args.run_id}")
        agent = VeriPatchAgent(_model(state.provider, settings), settings=settings, store=store)
        result = asyncio.run(agent.resume(args.run_id))
        print(result.model_dump_json(indent=2))
        raise SystemExit(0 if result.state.phase.value == "succeeded" else 1)
    if args.command == "inspect":
        store = SQLiteRunStore(settings.database_path)
        state = store.load(args.run_id)
        if state is None:
            raise SystemExit(f"Run not found: {args.run_id}")
        print(state.model_dump_json(indent=2))
        print(json.dumps(store.events(args.run_id), ensure_ascii=False, indent=2))
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
    if args.command == "mcp-server":
        from veripatch.mcp_server import serve

        serve(Path(args.repo))


if __name__ == "__main__":
    main()

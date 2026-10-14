"""Compare top-two hybrid retrieval with eight-candidate Weaviate reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import opik
from openai import OpenAI
from opik.evaluation import evaluate
from opik.integrations.openai import track_openai

from app.agent_tools import SupportMetrics
from app.config import get_settings
from app.observability import configure_opik
from app.retrieval import HYBRID_ALPHA, Retriever, load_chunks
from app.services.evidence import Evidence
from evals.run_six_way import (
    ANSWER_PROMPT,
    CASES,
    ROOT,
    SCORE_NAMES,
    WEB_FIXTURE,
    CaseMetric,
    fingerprint,
    make_answer,
)
from evals.six_way_runner import (
    aggregate_rows,
    compare_rows,
    experiment_rows,
    retrieval_diagnostics,
    run_case,
    validate_dataset,
)

RUNS = (("hybrid_web_sql", False), ("hybrid_rerank_web_sql", True))
BASELINE_MATCH_FIELDS = (
    "dataset_name",
    "dataset_id",
    "dataset_version",
    "source_hash",
    "prompt_hash",
    "answer_model",
    "embedding_image_tag",
    "reranker_model",
    "weaviate_collection",
    "weaviate_server_version",
    "hybrid_alpha",
    "retrieval_k",
    "web_mode",
)


def validate_baseline_report(saved: dict, experiment_id: str, config: dict) -> None:
    """Reject reused baselines with different evidence or answer configuration."""
    baseline_id = (
        saved.get("experiments", {}).get("hybrid_web_sql", {}).get("experiment_id")
    )
    if baseline_id != experiment_id:
        raise ValueError("Baseline report does not contain the requested experiment")
    prior = saved.get("config", {})
    different = [
        field for field in BASELINE_MATCH_FIELDS if prior.get(field) != config[field]
    ]
    if different:
        raise ValueError(f"Baseline configuration differs: {', '.join(different)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-k", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--report-stem", default="008_weaviate_reranker")
    parser.add_argument("--baseline-experiment-id")
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    if args.retrieval_k <= 0:
        parser.error("--retrieval-k must be positive")
    if args.candidate_k < args.retrieval_k:
        parser.error("--candidate-k must be at least --retrieval-k")
    if not args.report_stem.replace("_", "").isalnum():
        parser.error("--report-stem must be alphanumeric with optional underscores")
    if bool(args.baseline_experiment_id) != bool(args.baseline_report):
        parser.error("--baseline-experiment-id and --baseline-report go together")
    result_dir = ROOT / "experiments" / "results"
    report_path = result_dir / f"{args.report_stem}_comparison.json"
    preflight_path = result_dir / f"{args.report_stem}_retrieval.json"
    if args.baseline_report and report_path.resolve() == args.baseline_report.resolve():
        parser.error("--report-stem must differ from the baseline report")

    settings = get_settings()
    if not settings.reranker_enabled:
        raise RuntimeError("Set RERANKER_ENABLED=true for the local reranker run")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    web_data = json.loads(WEB_FIXTURE.read_text(encoding="utf-8"))
    web = {name: Evidence(**item) for name, item in web_data.items()}
    case_ids = {case["case_id"] for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("Duplicate case IDs in six-way dataset")
    chunks = load_chunks(settings.knowledge_base_dir)
    if not chunks:
        raise ValueError("Knowledge base has no document chunks")

    setup_started = time.perf_counter()
    retriever = Retriever(chunks)
    setup_seconds = round(time.perf_counter() - setup_started, 3)
    try:
        preflight = retrieval_diagnostics(
            cases,
            retriever,
            k=args.retrieval_k,
            methods=("hybrid", "hybrid_rerank"),
            candidate_k=args.candidate_k,
        )
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        if args.retrieval_only:
            print(
                json.dumps(
                    {
                        name: {
                            key: value
                            for key, value in summary.items()
                            if key != "cases"
                        }
                        for name, summary in preflight.items()
                    },
                    indent=2,
                )
            )
            return

        options = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            options["base_url"] = settings.openai_base_url
        answer_client = track_openai(OpenAI(**options))
        answer_model = settings.agent_model or settings.openai_model
        answer = make_answer(answer_client, answer_model)

        source_files = [CASES, WEB_FIXTURE, ROOT / "data/support_metrics.csv"]
        source_files.extend(settings.knowledge_base_dir.rglob("*.md"))
        source_hash = fingerprint(source_files)
        dataset_name = f"business-research-weaviate-six-way-{source_hash[:10]}"
        configure_opik()
        client = opik.Opik(project_name=settings.opik_project_name)
        dataset = client.get_or_create_dataset(
            name=dataset_name, project_name=settings.opik_project_name
        )
        if not validate_dataset(dataset.get_items(), cases):
            dataset.insert(cases)
        version = dataset.get_current_version_name()
        pinned = dataset.get_version_view(version)
        config = {
            "dataset_name": dataset_name,
            "dataset_id": dataset.id,
            "dataset_version": version,
            "source_hash": source_hash,
            "prompt_hash": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest()[:12],
            "answer_model": answer_model,
            "embedding_image_tag": settings.model2vec_image_tag,
            "reranker_model": settings.reranker_model,
            "reranker_inference": "weaviate-reranker-transformers-cpu",
            "weaviate_collection": retriever.collection_name,
            "weaviate_server_version": retriever.client.get_meta().get("version"),
            "weaviate_url": settings.weaviate_url,
            "hybrid_alpha": HYBRID_ALPHA,
            "retrieval_k": args.retrieval_k,
            "candidate_k": args.candidate_k,
            "web_mode": "fixed_fixture",
            "embedding_index_setup_seconds": setup_seconds,
        }
        if args.baseline_report:
            saved = json.loads(args.baseline_report.read_text(encoding="utf-8"))
            validate_baseline_report(saved, args.baseline_experiment_id, config)
        report = {"config": config, "retrieval_preflight": preflight, "experiments": {}}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = [CaseMetric(name) for name in SCORE_NAMES]
        for name, use_reranker in RUNS:
            if name == "hybrid_web_sql" and args.baseline_experiment_id:
                experiment_id = args.baseline_experiment_id
                experiment = client.get_experiment_by_id(experiment_id)
                rows = experiment_rows(
                    experiment.get_items(), case_ids, set(SCORE_NAMES)
                )
                report["experiments"][name] = {
                    "experiment_id": experiment_id,
                    "experiment_url": None,
                    "summary": aggregate_rows(rows),
                    "cases": rows,
                }
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                continue

            def task(item: dict, rerank: bool = use_reranker) -> dict:
                return {
                    "output": run_case(
                        item,
                        "hybrid_web_sql",
                        retriever=retriever,
                        web=web,
                        sql=SupportMetrics(),
                        answer=answer,
                        k=args.retrieval_k,
                        rerank=rerank,
                        candidate_k=args.candidate_k if rerank else None,
                    )
                }

            result = evaluate(
                dataset=pinned,
                task=task,
                scoring_metrics=metrics,
                experiment_name=f"weaviate-{name}-{source_hash[:8]}",
                experiment_config={**config, "variant": name, "rerank": use_reranker},
                experiment_tags=["weaviate-reranker", name, source_hash[:8]],
                nb_samples=len(cases),
                task_threads=1,
            )
            opik.flush_tracker()
            for attempt in range(5):
                experiment = client.get_experiment_by_id(result.experiment_id)
                try:
                    rows = experiment_rows(
                        experiment.get_items(), case_ids, set(SCORE_NAMES)
                    )
                    break
                except (ValueError, KeyError):
                    if attempt == 4:
                        raise
                    time.sleep(2)
            report["experiments"][name] = {
                "experiment_id": result.experiment_id,
                "experiment_url": result.experiment_url,
                "summary": aggregate_rows(rows),
                "cases": rows,
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(name, result.experiment_url, flush=True)

        baseline = report["experiments"]["hybrid_web_sql"]["cases"]
        candidate = report["experiments"]["hybrid_rerank_web_sql"]["cases"]
        report["comparison"] = compare_rows(baseline, candidate)
        ids = [report["experiments"][name]["experiment_id"] for name, _ in RUNS]
        report["compare_url"] = (
            f"https://www.comet.com/opik/{settings.opik_workspace}"
            f"/experiments/{dataset.id}/compare?experiments={quote(json.dumps(ids))}"
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("compare", report["compare_url"], flush=True)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()

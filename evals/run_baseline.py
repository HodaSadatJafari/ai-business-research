import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance

from app.config import get_settings
from app.services.research import ResearchService

settings = get_settings()

client = opik.Opik(
    project_name=settings.opik_project_name,
)

dataset = client.get_dataset(
    name="business-research-baseline",
)

# dataset_v1 = dataset.get_version_view("v1")

service = ResearchService()


def task(item: dict) -> dict:
    question = item["input"]

    response = service.research(question)

    return {"output": response.model_dump(mode="json")}


def main() -> None:
    print("Dataset:", dataset.name)

    result = evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=[AnswerRelevance(require_context=False)],
        experiment_name="baseline-v0.1",
        nb_samples=20,
        task_threads=1,
    )

    print(result)


if __name__ == "__main__":
    main()

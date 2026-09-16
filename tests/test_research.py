from app.schemas.research import (
    ResearchFinding,
    ResearchResponse,
)


def test_research_response_schema():

    response = ResearchResponse(
        question="What is RAG?",
        summary="RAG combines retrieval with generation.",
        findings=[
            ResearchFinding(
                claim="Retrieval provides external context.",
                explanation="Relevant documents are retrieved before generation.",
            )
        ],
        limitations=["The quality depends on the retrieved information."],
    )

    assert response.question == "What is RAG?"
    assert len(response.findings) == 1

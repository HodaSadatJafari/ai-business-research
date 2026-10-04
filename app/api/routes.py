from functools import lru_cache

from fastapi import APIRouter, HTTPException
from openai import APIConnectionError, APIError, RateLimitError

from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.agent import AgentModelUnsupported, AgentResearchService
from app.services.rag import RAGResearchService
from app.services.research import ResearchService

router = APIRouter()

research_service = ResearchService()


@lru_cache
def get_rag_service() -> RAGResearchService:
    return RAGResearchService()


@lru_cache
def get_agent_service() -> AgentResearchService:
    return AgentResearchService()


def _run_research(
    service: ResearchService | RAGResearchService | AgentResearchService,
    question: str,
) -> ResearchResponse:
    try:
        return service.research(question)
    except AgentModelUnsupported as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (APIConnectionError, RateLimitError) as exc:
        raise HTTPException(
            status_code=503, detail="Model provider temporarily unavailable."
        ) from exc
    except APIError as exc:
        raise HTTPException(
            status_code=502, detail="Model provider request failed."
        ) from exc


@router.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    return _run_research(research_service, request.question)


@router.post("/research/rag", response_model=ResearchResponse)
def research_with_knowledge(request: ResearchRequest) -> ResearchResponse:
    return _run_research(get_rag_service(), request.question)


@router.post("/research/agent", response_model=ResearchResponse)
def research_with_agent(request: ResearchRequest) -> ResearchResponse:
    return _run_research(get_agent_service(), request.question)

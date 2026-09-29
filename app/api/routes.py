from functools import lru_cache

from fastapi import APIRouter, HTTPException

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


@router.post(
    "/research",
    response_model=ResearchResponse,
)
async def research(request: ResearchRequest) -> ResearchResponse:

    return await research_service.research(request.question)


@router.post("/research/rag", response_model=ResearchResponse)
def research_with_knowledge(request: ResearchRequest) -> ResearchResponse:
    return get_rag_service().research(request.question)


@router.post("/research/agent", response_model=ResearchResponse)
def research_with_agent(request: ResearchRequest) -> ResearchResponse:
    try:
        return get_agent_service().research(request.question)
    except AgentModelUnsupported as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

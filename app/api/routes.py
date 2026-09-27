from functools import lru_cache

from fastapi import APIRouter

from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.rag import RAGResearchService
from app.services.research import ResearchService

router = APIRouter()

research_service = ResearchService()


@lru_cache
def get_rag_service() -> RAGResearchService:
    return RAGResearchService()


@router.post(
    "/research",
    response_model=ResearchResponse,
)
async def research(request: ResearchRequest) -> ResearchResponse:

    return await research_service.research(request.question)


@router.post("/research/rag", response_model=ResearchResponse)
def research_with_knowledge(request: ResearchRequest) -> ResearchResponse:
    return get_rag_service().research(request.question)

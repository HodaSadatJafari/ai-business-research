from fastapi import APIRouter

from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.research import ResearchService

router = APIRouter()

research_service = ResearchService()


@router.post(
    "/research",
    response_model=ResearchResponse,
)
async def research(request: ResearchRequest) -> ResearchResponse:

    return await research_service.research(request.question)

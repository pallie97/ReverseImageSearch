from fastapi import APIRouter
from api.controller import search

# https://fastapi.tiangolo.com/tutorial/bigger-applications/?h=router.inclu#include-an-apirouter-in-another
router = APIRouter()
router.include_router(search.router, tags=["search"])
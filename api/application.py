import logging

from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from api.controller.http_error import http_error_handler
from api.config import ROOT_PATH
from api.controller.router import router as api_router

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)

def get_application() -> FastAPI:
    application = FastAPI(title="ImageSearch-API", debug=True, version="0.0.1", root_path=ROOT_PATH)

    # This middleware enables allow all cross-domain requests to the API from a browser. For production
    # deployments, it could be made more restrictive.
    application.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    application.add_exception_handler(HTTPException, http_error_handler)
    application.include_router(api_router)

    return application


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names.

    Should be called only after all routes have been added.

    See https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name  # in this case, 'read_items'

app = get_application()
use_route_names_as_operation_ids(app)

logger.info("See docs localhost:8000/docs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
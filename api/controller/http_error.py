from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

# https://fastapi.tiangolo.com/tutorial/handling-errors/
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)
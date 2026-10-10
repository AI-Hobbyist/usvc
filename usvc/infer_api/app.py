import hmac
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from .catalog import Catalog
from .errors import APIError
from .settings import settings


def create_app(model_root="models", device="cpu", token=None, validator=None):
    # M1 controlled validator; M2 replaces this with the service's real probe.
    if validator is None:
        from .fixture_engine import validate_network
        validator = validate_network
    catalog = Catalog(model_root, validator)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.catalog = catalog
    app.state.device = device

    @app.middleware("http")
    async def authenticate(request, call_next):
        if token:
            supplied = request.headers.get("authorization", "")
            scheme, _, credential = supplied.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(credential.encode(), token.encode()):
                return JSONResponse(APIError(401, "unauthorized", "Bearer authentication required").body(), status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.get("/model")
    async def models():
        return await run_in_threadpool(catalog.refresh)

    @app.get("/settings")
    async def model_settings(request: Request):
        backend, identity = request.query_params.get("backend"), request.query_params.get("model")
        if not backend:
            raise APIError(422, "invalid_parameter", "backend is required", "backend")
        await run_in_threadpool(catalog.refresh)
        if identity:
            model = catalog.get(identity)
            if model.backend != backend:
                raise APIError(422, "invalid_parameter", "Backend/model mismatch", "backend")
        else:
            model = next((m for _, m in sorted(catalog.models.items()) if m.backend == backend), None)
            if model is None:
                raise APIError(404, "unknown_backend", "No available models for backend", "backend")
        return settings(model, device)

    return app

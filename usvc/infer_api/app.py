import hmac
import json
import os
from pathlib import Path
import tempfile
from contextlib import asynccontextmanager
import anyio
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from .catalog import Catalog
from .errors import APIError
from .settings import settings, validate


class PrettyJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def create_app(model_root="models", device="cpu", token=None, validator=None, runtime=None,
               max_upload_bytes=64 * 1024 ** 2, max_audio_seconds=600., max_pending=4):
    from .runtime import Runtime
    if min(max_upload_bytes, max_audio_seconds, max_pending) <= 0:
        raise ValueError("upload, duration and scheduler limits must be positive")
    runtime = runtime or Runtime(device)
    catalog = Catalog(model_root, validator or runtime.probe)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await run_in_threadpool(runtime.close)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan,
                  default_response_class=PrettyJSONResponse)
    app.state.catalog = catalog
    app.state.device = device
    app.state.runtime = runtime
    import threading
    admission = threading.BoundedSemaphore(max_pending)
    app.state.uploads = set()

    @app.middleware("http")
    async def authenticate(request, call_next):
        if token:
            supplied = request.headers.get("authorization", "")
            scheme, _, credential = supplied.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(credential.encode(), token.encode()):
                return PrettyJSONResponse(APIError(401, "unauthorized", "Bearer authentication required").body(), status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return PrettyJSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return PrettyJSONResponse(APIError(exc.status_code, code, str(exc.detail)).body(),
                                  status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        import logging
        logging.getLogger(__name__).exception("Unexpected inference service failure", exc_info=exc)
        return PrettyJSONResponse(APIError(500, "internal_error", "Unexpected service failure").body(), status_code=500)

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

    @app.post("/infer")
    async def infer(request: Request):
        identity = request.query_params.get("model")
        if not identity:
            raise APIError(422, "invalid_parameter", "model is required", "model")
        try:
            values = json.loads(request.query_params.get("params", "{}"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, TypeError):
            raise APIError(422, "invalid_parameter", "params must be valid JSON", "params") from None
        model = await run_in_threadpool(catalog.candidate, identity)
        values = validate(model, values, device)
        media = request.headers.get("content-type", "application/octet-stream").split(";", 1)[0].lower()
        if media not in ("audio/wav", "audio/x-wav", "audio/flac", "audio/ogg", "application/octet-stream"):
            raise APIError(415, "unsupported_media_type", "Expected WAV, FLAC, OGG or octet-stream body")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError()
            except ValueError:
                raise APIError(422, "invalid_audio", "Invalid Content-Length") from None
            if size > max_upload_bytes:
                raise APIError(413, "upload_too_large", "Upload exceeds byte limit")
        if not admission.acquire(blocking=False):
            raise APIError(503, "scheduler_full", "Inference scheduler is full")
        path = None
        try:
            descriptor, filename = tempfile.mkstemp(prefix="usvc-upload-", suffix=".audio")
            path = Path(filename)
            app.state.uploads.add(path)
            size = 0
            with os.fdopen(descriptor, "wb") as upload:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise APIError(413, "upload_too_large", "Upload exceeds byte limit")
                    upload.write(chunk)
            if size == 0:
                raise APIError(422, "invalid_audio", "Empty audio body")
            def check_audio():
                try:
                    info = sf.info(path)
                    if info.frames < 1 or info.samplerate < 1 or info.duration > max_audio_seconds:
                        raise ValueError("Empty audio or duration exceeds limit")
                    wav, _ = sf.read(path, dtype="float32", always_2d=True)
                    if not np.isfinite(wav).all():
                        raise ValueError("Audio contains nonfinite samples")
                except (ValueError, sf.LibsndfileError) as exc:
                    raise APIError(422, "invalid_audio", str(exc)) from None
            await run_in_threadpool(check_audio)
            # Shield the running worker: cancellation must not delete its input or release
            # admission until actual computation and GPU teardown have finished.
            with anyio.CancelScope(shield=True):
                def compute():
                    try:
                        catalog.ensure_ready(model)
                        return runtime.execute(model, path, values)
                    except APIError as exc:
                        if exc.status >= 500:
                            catalog.invalidate(identity)
                        raise
                raw, sample_rate = await anyio.to_thread.run_sync(compute, abandon_on_cancel=False)
            headers = {"X-USVC-Sample-Rate": str(sample_rate)}
            if model.fixture:
                headers["X-USVC-Test-Fixture"] = "true"
            async def chunks():
                for start in range(0, len(raw), 64 * 1024):
                    yield raw[start:start + 64 * 1024]
            return StreamingResponse(chunks(), media_type="audio/wav", headers=headers)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
                app.state.uploads.discard(path)
            admission.release()

    return app

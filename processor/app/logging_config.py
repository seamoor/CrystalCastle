import logging


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)

    # Uvicorn installs its own handlers. Route app.* logs through uvicorn.error
    # so they are always visible in `docker compose logs`.
    uvicorn_err = logging.getLogger("uvicorn.error")
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    if uvicorn_err.handlers:
        app_logger.handlers = uvicorn_err.handlers
        app_logger.propagate = False

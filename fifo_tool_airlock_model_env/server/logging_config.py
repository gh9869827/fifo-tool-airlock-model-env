from logging.config import dictConfig

def configure_logging():
    """
    Configure logging to integrate custom application logs with Uvicorn's logging format.

    This function sets up a Uvicorn-compatible log formatter and attaches it to all
    relevant loggers (uvicorn, uvicorn.error, uvicorn.access, __main__, and root).

    The configuration ensures:
    - Colored and structured log formatting.
    - Logs are routed to stdout.
    - Duplication is avoided via `propagate = False`.
    - Your FastAPI and module logs appear alongside Uvicorn logs.

    Designed to be call on top of the FastAPI entry point before initializing the app.
    """

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False
            },
            "__main__": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False
            }
        },
        "root": {
            "handlers": ["default"],
            "level": "INFO",
        },
    })

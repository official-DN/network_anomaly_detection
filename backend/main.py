"""
Main FastAPI application entry point.
AI-Based Anomaly Detection in Network Traffic.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from contextlib import asynccontextmanager

from backend.config import settings
from backend.database import connect_db, disconnect_db
from backend.utils.logger import setup_logger, logger
from backend.utils.security import authenticate_user, create_access_token
from backend.api import (
    traffic_routes,
    detection_routes,
    alert_routes,
    model_routes,
    analytics_routes,
)
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import HTTPException, status


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logger(str(settings.log_dir), settings.log_level)
    logger.info(f"Starting {settings.app_name} [{settings.app_env}]")
    await connect_db()
    yield
    await disconnect_db()
    logger.info("Application shutdown")


app = FastAPI(
    title="AI Network Anomaly Detection API",
    description=(
        "AI-powered network traffic anomaly detection using "
        "Random Forest, XGBoost, Isolation Forest, and Autoencoder models."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(traffic_routes.router)
app.include_router(detection_routes.router)
app.include_router(alert_routes.router)
app.include_router(model_routes.router)
app.include_router(analytics_routes.router)


# Health check
@app.get("/api/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": settings.app_env,
    }


# Authentication
@app.post("/api/auth/login", tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and return a JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"],
    }


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
    )

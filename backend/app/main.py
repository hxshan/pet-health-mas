from fastapi import FastAPI
from app.api.routes.health import router as health_router
from app.api.routes.cases import router as cases_router
from app.api.routes.traces import router as traces_router

app = FastAPI(title="Pet Health Multi-Agent System")

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(cases_router, prefix="/cases", tags=["Cases"])
app.include_router(traces_router, prefix="/traces", tags=["Traces"])
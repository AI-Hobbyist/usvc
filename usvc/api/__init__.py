"""usvc-serve: WebUI/桌面端本地 API 服务。

FastAPI 应用（main.create_app）+ 统一 Job 抽象（jobs.JobScheduler）+ uvicorn
入口（main.main，console script `usvc-serve`）。参见 docs/WEBUI_DESIGN.md §3。
"""

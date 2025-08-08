import os


def main():
    target = os.getenv("SERVICE", "help")
    if target == "ingestion":
        # uvicorn src.ingestion_service.app:app --port 8001
        import uvicorn
        uvicorn.run("src.ingestion_service.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
    elif target == "retrieval":
        import uvicorn
        uvicorn.run("src.retrieval_service.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8002")), reload=False)
    elif target == "whatsapp":
        import uvicorn
        uvicorn.run("src.whatsapp_service.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8003")), reload=False)
    else:
        print("Set SERVICE=ingestion|retrieval|whatsapp to run a microservice. Example: SERVICE=ingestion python -m main")


if __name__ == "__main__":
    main()

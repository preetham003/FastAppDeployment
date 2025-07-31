from fastapi import FastAPI
from fastapi_health import health  # lightweight health checks

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

def is_app_healthy() -> bool:
    # Add custom logic here (e.g. database ping, cache check, etc.)
    # Return True if healthy, False otherwise
    return True

# Only returns 200 if all conditions pass
app.add_api_route("/health", health([is_app_healthy]))
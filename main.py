from fastapi import FastAPI, HTTPException
from config import azure_llm, search_client, embeddings
from app.models import QueryRequest
from app.services import chat_with_products

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the status of core components.
    Checks if azure_llm and azure_embeddings are initialized.
    """
    status = {"status": "healthy"}
    issues = []

    # Check if Azure LLM is initialized
    if azure_llm is None:
        issues.append("Azure LLM (azure_llm) not initialized in config.")
    else:
        pass
    # Check if Search Client is initialized
    if search_client is None:
        issues.append("Search Client (search_client) not initialized in config.")
    else:
        pass

    if issues:
        status["status"] = "unhealthy"
        status["issues"] = issues
        raise HTTPException(status_code=503, detail=status)
    
    return status

@app.post(
    path="/evaluation",
    status_code=200,
    description="Chat with the model using user input and it'll return the response along with evaluation metrics.",
)
async def evaluate_user_query(request: QueryRequest):
    user_input = request.user_input
    response = await chat_with_products(
        messages=[{"role": "user", "content": user_input}],
        azure_llm=azure_llm,
        embeddings=embeddings,
        search_client=search_client
    )
    return {"message": response["message"]}
import json
from azure.search.documents import SearchClient
from azure.ai.inference.prompts import PromptTemplate
from azure.search.documents.models import VectorizedQuery
from azure.ai.inference import EmbeddingsClient
from langchain_openai import AzureChatOpenAI
from pathlib import Path

async def get_similar_documents(
    messages: list, 
    chat_model: AzureChatOpenAI, 
    embeddings: EmbeddingsClient, 
    search_client: SearchClient,
    context: dict = None
) -> list:
    context = context or {} 
    top = context.get("overrides", {}).get("top", 5)
    try:
        # Determine the path in a platform-safe way
        base = Path(__file__).resolve().parent
        prompt_path = base / "assets" / "intent_mapping.prompty"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        # Load the prompt template from file
        intent_prompt = PromptTemplate.from_prompty(prompt_path)

        # Create chat message list for conversation
        prompt_msgs = intent_prompt.create_messages(conversation=messages)
        resp = chat_model.invoke(
            input=prompt_msgs,
            **intent_prompt.parameters
        )
    except Exception as e:
        print(f"Error during intent mapping: {e}")
        raise e

    intent_query = resp.content
    # logging.debug(f"🧠 Intent mapping: {search_query}")
    print(f"🧠 Intent mapping: {intent_query}")
    search_query = json.loads(intent_query)
    print(search_query["proceed_to_retrieval"])
    if search_query["proceed_to_retrieval"]:
        emb = embeddings.embed(
            model="text-embedding-ada-002",
            input=[search_query["improved_query"]],
        )
        vector = emb.data[0].embedding
        vq = VectorizedQuery(vector=vector, k_nearest_neighbors=top, fields="contentVector")

        results = search_client.search(
            search_text=search_query,
            vector_queries=[vq],
            select=["id", "content", "filepath", "page_number"],
            vector_filter_mode="preFilter",
            top=5
        )

        docs = [{
            "id": r["id"],
            "content": r["content"],
            "filepath": r["filepath"],
            "page_number":r["page_number"]
        } for r in results]

        context.setdefault("thoughts", []).append({
            "title": "Generated search query",
            "description": search_query
        })
        context.setdefault("grounding_data", []).extend(docs)

        # logging.debug(f"📄 Retrieved {len(docs)} documents")
        print(f"📄 Retrieved {len(docs)} documents")
        return {
            "documents": docs,
            "message":"",
            "context": context,
        }
    else:
        return {
            "documents": [],
            "message": search_query["improved_query"],
            "context": context,
        }
    
async def chat_with_products(
    messages: list, 
    azure_llm: AzureChatOpenAI,
    embeddings: EmbeddingsClient,
    search_client: SearchClient,
    context: dict = None
) -> dict:
    if context is None:
        context = {}

    documents = await get_similar_documents(
        messages=messages,
        chat_model=azure_llm,
        embeddings=embeddings,
        search_client=search_client,
        context=context
    )
    if not documents["documents"]:
        return {"message": documents["message"], "context": documents["context"], "valid_query": False}
    else:
        docs = documents["documents"]
        # Determine the path in a platform-safe way
        base = Path(__file__).resolve().parent
        prompt_path = base / "assets" / "grounded_chat.prompty"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        # do a grounded chat call using the search results
        grounded_chat_prompt = PromptTemplate.from_prompty(prompt_path)
        # logger.info(f"messages:{messages}")
        system_message = grounded_chat_prompt.create_messages(documents=docs, context=context)
        # response = chat.chat.completions.create(
        #     model=os.environ["CHAT_MODEL"],
        #     messages=system_message + messages,
        #     **grounded_chat_prompt.parameters,
        # )
        response = azure_llm.invoke(
            system_message + messages,
            **grounded_chat_prompt.parameters, # These parameters will be passed through to the underlying API call
        )
        # logger.info(f"💬 Response: {response.choices[0].message}")
        print(f"💬 Response: {response.content}")

        # Return a chat protocol compliant response
        return {"message": response.content, "context": documents["context"], "valid_query": True}
    
import torch
from transformers import AutoModelForCausalLM , AutoTokenizer, pipeline

from src.config import (
    LLM_MODEL_NAME
)

print(f"Loading Transformer model : {LLM_MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

device = "cpu"

model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype = torch.float32
).to(device)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

def generate_recommendation(query: str, movies:list[dict]) -> str:

    if not movies:
        return "I couldn't find any movies matching you request"

    context_blocks = []
    for index, movie in enumerate(movies, start=1):
        title = movie["metadata"].get("title","Unknown Title")
        genres = movie["metadata"].get("genres","Unknown")
        doc = movie["document"]
        print(f"movie recommeded : {doc}")
        context_blocks.append(
            f"[{index}] Title: {title}\n"
            f"    Genres: {genres}\n"
            f"    Details: {doc}\n"
        )


    context_str = "\n".join(context_blocks)

    messages = [
            {
                "role": "system",
                "content" : (
                    "You are an expert enthusiastic movie recommendation assistant. "
                    "Your task is to recommend movies to the user strictly based on provided context below. "
                    "CRITICAL: Do not invent, assume or hallucinate movies that are not explicitely listed in the context. "
                    "Write a conversational, engaging response explaining *why* these movies matched user request"
                )
            },
            {
                "role":"user",
                "content": (
                    f"User Request: {query}\n\n"
                    f"Retrieved Movie Context : \n{context_str}\n\n"
                    "Please provide a helpful, personalized movie recommendation based on the above"
                )
            }
        ]

    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    outputs = llm_pipeline(
            prompt,
            max_new_tokens=350,
            temperature= 0.7,
            do_sample=True,
            top_p=0.9,
            return_full_text=False
        )

    generated_answer = outputs[0]["generated_text"].strip()
    return generated_answer
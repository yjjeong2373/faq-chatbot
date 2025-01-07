from fastapi import FastAPI
from faq_chatbot import FaqChatbot

app = FastAPI()
faq_chatbot = FaqChatbot()


@app.post("/chatbot")
def chat(user_question: str):
    response = faq_chatbot(user_question)
    return response


@app.get("/health")
def health_check():
    return "OK"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
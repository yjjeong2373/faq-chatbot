import os
import pickle
from openai import OpenAI
import chromadb
import chromadb.utils.embedding_functions as embedding_functions


class FaqChatbot:
    def __init__(self):
        self.db = self._set_database()
        self.openai_client = OpenAI()
        self.openai_model = "gpt-4o-mini"
        self.conversation_history = [
            {"role": "system", "content": "이 챗봇은 스마트 스토어의 FAQ를 지원하도록 설계되었습니다."}
        ]

    def _set_database(self):
        with open("data/final_result.pkl", "rb") as file:
            data = pickle.load(file)

        documents, ids = [], []
        for i, (_, v) in enumerate(data.items()):
            documents.append(v)
            ids.append(f"id{i}")

        chromadb_client = chromadb.Client()
        api_key = os.getenv("OPENAI_API_KEY")

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )

        collection = chromadb_client.create_collection(
            name="smartstore_faq_db", 
            embedding_function=openai_ef)

        collection.add(documents=documents, ids=ids)

        return collection


    def _is_question_relevant(self, question):
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": '이 챗봇은 스마트 스토어의 FAQ를 지원하도록 설계되었습니다. 여기에는 회원가입, 주문, 배송, 반품, 정책 관련 등의 스마트 스토어와 관련된 광범위한 질문이 포함됩니다. 다음 질문이 스마트 스토어에 적절한 질문인지 판단하고 조금 관련이 있는 경우 "yes", 관련이 전혀 없는 경우 "no"로 답하세요.'},
                {"role": "user", "content": f"Question: \"{question}\""}
            ],
            max_tokens=5,
        )

        return response.choices[0].message.content.strip().lower() == "yes"
    

    def __call__(self, user_question):
        if not self._is_question_relevant(user_question):
            ai_response = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."

        else:
            retrieved_document = self.db.query(query_texts=[user_question],n_results=1)['documents'][0]
            
            self.conversation_history.extend([
                {"role": "assistant", "content": f"The following documents provide relevant information:\n{retrieved_document}"},
                {"role": "user", "content": user_question}
            ])

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=self.conversation_history,
            )

            ai_response = response.choices[0].message.content

        self.conversation_history.append({"role": "assistant", "content": ai_response})

        return ai_response


if __name__ == "__main__":
    faq_chatbot = FaqChatbot()
    r = faq_chatbot("미성년자도 판매 회원 등록이 가능한가요?")
    print(r)
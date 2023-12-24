# from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# from langchain.prompts import HumanMessagePromptTemplate
# from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

from dotenv import load_dotenv

import os

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

class Assistant:
    SYSTEM_TEMPLATE = """
        Given a context, assume the role of an expert historian responding to questions posed by 
        a history professor with a doctoral degree. Generate objective and concise answers, demonstrating 
        a comprehensive understanding of the provided context and content. Limit each response to a maximum of 500 words 
        to ensure brevity and clarity. Your responses should reflect and make reference to the context provided. The answers
        should be written in a formal tone and should be concise and objective, and write in pt-br, following the rules of 
        the portuguese language. 

        Context: {context}
    """

    HUMAN_TEMPLATE = "Question: {question}"

    def __init__(self, llm, db):
        self.llm = llm
        self.db = db
        self.chain = None
        self.create_chain()
    
    def chat(self, question, k=20):
        matches = self.search(question, k=k)
        answer = self.answer(question=question, context=matches)
        return answer
    
    def search(self, question, k=20):
        matches = self.db.similarity_search(question, k=k)
        return matches
    
    def answer(self, question, context) -> str:
        return self.chain.run(question=question, context=context)
    
    def create_chain(self):
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_TEMPLATE)
        human_message_prompt = HumanMessagePromptTemplate.from_template(self.HUMAN_TEMPLATE)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        self.chain = LLMChain(llm=self.llm, prompt=chat_prompt)
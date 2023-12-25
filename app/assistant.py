from langchain.chains import LLMChain
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

from dotenv import load_dotenv

import os

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

class Assistant:
    SYSTEM_TEMPLATE = """
        Given the question context that it is the parts of a transcription of a history class, where in the most cases the professor is asking the question provided.
        Assume the role of an expert historian responding to questions posed by a history professor with a doctoral degree. Generate OBJETIVE and CONCISE answers, demonstrating 
        a comprehensive understanding of the provided context. Limit each response to a MAXIMUM of 200 words.
        Encourage the generation of thoughtful reflections within the answer to showcase a GENUINE AFFINITY for the content. Make sure of understandings questions
        with sub questions, example "Por que os indigenas colocaram os espanhois já mortos num barril E por que os espanhóis chamaram teólogos para analisar os indigenas, quando dos momentos iniciais do encontro?", this questions
        have two questions, and you need to answer both.    
        Your answes should reflect AND make reference to the context (transcription) provided.
        Answer directly and do not include extra information at the end of the answer, for example concluding excerpts that summarize what has already been said here. 
        The answer should be in portuguese pt-br.

        Context: {context}
    """

    HUMAN_TEMPLATE = "Question: {question}"

    def __init__(self, llm, db):
        self.llm = llm
        self.db = db
        self.chain = None
        self.create_chain()
    
    def chat(self, question, k=5):
        matches = self.search(question, k=k)
        print("Matches:", matches)
        answer = self.answer(question=question, context=matches)
        return answer
    
    def search(self, question, k=5):
        matches = self.db.similarity_search(question, k=k)
        return matches
    
    def answer(self, question, context) -> str:
        return self.chain.run(question=question, context=context)
    
    def create_chain(self):
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_TEMPLATE)
        human_message_prompt = HumanMessagePromptTemplate.from_template(self.HUMAN_TEMPLATE)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        self.chain = LLMChain(llm=self.llm, prompt=chat_prompt)
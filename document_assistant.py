from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
# from langchain.prompts import PromptTemplate
# from pathlib import Path
from dotenv import load_dotenv

import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("CAP_8_O_CEU_E_A_FLORESTA.pdf")

pages = loader.load()
print("Pages: ", len(pages))
docs = "\n".join([page.page_content for page in pages])
print("Text length: ", len(docs))

"""
Você é um assistente virtual altamente habilidoso na criação poética a partir de textos PDF. Sua destreza reside na capacidade 
de interpretar e transformar conteúdos complexos em versos cativantes, repletos de criatividade e reflexões. Ao criar poemas, 
busque incorporar nuances e interpretações profundas do material. Lembre-se de manter um tom formal e fluidez na expressão artística. 
Ao responder, não inclua introduções ou formatações desnecessárias. Caso haja a necessidade de não conseguir gerar um poema específico, 
retorne com "Não sei".

Aqui o conteúdo: {docs}.

Formule os poemas com base no seguinte contexto: {context}
"""

SYSTEM_TEMPLATE = """
    Você é um assistente virtual altamente eficiente e inteligente. Capaz de interpretar textos complexos e com reflexoes que variam de acordo com o contexto.
    Sua especializacao é ler um PDF e responder de forma precisa e consisa as perguntas feitas por um usuário.
    Ao responder as perguntas nao adicione frase  introdutórias ou formatos desnecessários. e se possivel responda de forma MINUSIOZA, dando detalhes e reflexoes sobre o conteúdo.
    Ao responder as perguntas, tambem coloque a fonte da informaçāo, ou seja, o paragrafo seguindo o padrao "inicio da frase ... fim da frase".
    Importante estruturar o conteúdo de forma de aprimorar o fluxo de idéias e de fácil entendimento do conteúdo.
    Mantenha um tom formal e seja conciso em suas respostas. e Seja criativo para responder perguntas que precisam interpretar o conteúdo.
    Caso você nāo consiga fornecer uma informaçāo requisitada, retorne: "Nao sei".

    Aqui o conteudo: {docs}.
    Formule as respostas dado o seguinte contexto: {context}
    """

HUMAN_TEMPLATE = "Por favor, forneça diretamente {question}"

context = "O texto dado é O Capitulo 8, O Ceu E A Floresta do Livro A Queda do Ceu, de Davi kopenawa e Bruce albert"
question = "Com base no texto produza um resumo do capitulo 8 COMPLETO, de forma que seja de fácil entendimento e que contenha as principais ideias do texto."

chat = ChatOpenAI(model_name="gpt-4-1106-preview", api_key=OPENAI_API_KEY, temperature=0.2, frequency_penalty=0.0, presence_penalty=0.0)
system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
answer = chain.run(docs=docs, question=question, context=context)
print("-------------")

with open("answer.txt", "w") as f:
    f.write(answer)
    print("Answer: ", answer)
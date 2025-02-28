import logging
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from pdf_loader import PdfLoader
from vector_db import VectorDatabase


load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_tools(qa_chain: RetrievalQA) -> list:
    """
    Creates a list of tools for the agent.
    
    :param qa_chain: The RetrievalQA chain for document retrieval.
    :return: List of tools.
    """
    return [
        Tool(
            name="Document Retrieval",
            func=lambda q: qa_chain({"query": q})["result"],
            description="Retrieves information about traffic rules."
        )
    ]


def get_agent_instructions() -> str:
    """
    Provides instructions for the agent to always respond in Russian.
    
    :return: Instruction string.
    """
    return (
        "Ты умный и добрый ассистент по Правилам дорожного движения, который ВСЕГДА отвечает на русском языке. "
        "Если пользователь задаёт вопрос, дай ответ на русском."
        "Верни пользователю ответы на все заданные им вопросы."
        "Пожалуйста, отвечай пользователю подробно и доброжелательно."
    )


def initialize_custom_agent(llm, tools) -> any:
    """
    Initializes the agent with the specified tools and language model.
    
    :param llm: The language model to use.
    :param tools: List of tools for the agent.
    :return: Initialized agent.
    """
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


def main():
    try:
        logger.info("Initializing Qdrant...")

        vector_db = VectorDatabase()
        pdf_loader = PdfLoader()

        model_name = os.getenv("OPENAI_MODEL")

        try:
            documents = pdf_loader.load_and_process_pdf()
            retriever = vector_db.load_index(documents)
        except Exception as e:
            raise Exception(f"Error occurred while loading index: {e}")

        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "score_threshold": 0.5}
            ),
            return_source_documents=True
        )

        tools = get_tools(qa_chain)

        instructions = get_agent_instructions()

        agent = initialize_custom_agent(llm, tools)

        logger.info("Chatbot is ready! Enter a query (or type 'e' to quit):")
        while True:
            query = input("> ")
            if query.lower() == "e":
                break

            try:
                query_with_instructions = f"{instructions}\n{query}"
                response = agent.run(query_with_instructions)

                logger.info(f"Response: {response}")

            except Exception as e:
                logger.error(f"Error during query execution: {e}")

    except Exception as e:
        logger.critical(f"Critical error: {e}")


if __name__ == "__main__":
    main()
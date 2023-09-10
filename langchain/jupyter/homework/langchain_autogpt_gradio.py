import gradio as gr
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore


def initialize_autogpt():
    # 构造 AutoGPT 的工具集
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    embeddings_model = OpenAIEmbeddings()
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    global agent
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        memory=vectorstore.as_retriever(),  # 实例化 Faiss 的 VectorStoreRetriever
    )
    agent.chain.verbose = True


questions = []


def langchain_gpt(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    questions.append(message)
    if "finish" in questions:
        questions.clear()

    r = agent.run(questions)
    return str(r)


def launch_gradio(title=""):
    demo = gr.ChatInterface(
        fn=langchain_gpt,
        title=title,
        retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化AutoGPT with langchain
    initialize_autogpt()
    launch_gradio()

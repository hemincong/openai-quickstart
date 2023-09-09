import gradio as gr


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def initialize_vector_store_dir(source_name):
    from langchain.text_splitter import CharacterTextSplitter

    with open(source_name + '.txt') as f:
        sales_data = f.read()

        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.',
            chunk_size=100,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
        )
        docs = text_splitter.create_documents([sales_data])
        db = FAISS.from_documents(docs, OpenAIEmbeddings())
        db.save_local(source_name)


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio(title=""):
    demo = gr.ChatInterface(
        fn=sales_chat,
        title=title,
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


rot_type = {
    "地产": ("地产销售", "real_estate_sales_data"),
    "教育": ("教育销售", "education_sales_data"),
    "电器": ("电器销售", "electrical_appliance_sales_data"),
    "家装": ("家装销售", "home_improvement_sales_data"),
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='sales_chatbot',
        description='sale chatbot',
        epilog='Text at the bottom of help')
    parser.add_argument('--type')
    args = parser.parse_args()
    t = args.type

    print("当前聊天机器人类型：{t}".format(t=t))

    initialize_vector_store_dir(rot_type[t][1])
    # 初始化房产销售机器人
    initialize_sales_bot(rot_type[t][1])
    # 启动 Gradio 服务
    launch_gradio(rot_type[t][0])

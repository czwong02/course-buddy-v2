import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

DB_CHROMA_PATH = "vectorstore/db_chroma"

# QA Model Function
def qa_bot():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)

    qa = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama3"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=retrieve_memory(),
        return_source_documents=True,
    )

    return qa


# Memory for conversational context
def retrieve_memory():
    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    return memory


# start the chainlit
@cl.on_chat_start
async def on_chat_start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to FSKTM Course Buddy. I am Doggy. How can I assist you?"
    # Sending an image with the local file path
    msg.elements = [
    cl.Image(name="image", display="inline", path="pic.gif")
    ]
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    # call backs happens asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
    await cl.Message(content=answer, elements=text_elements).send()

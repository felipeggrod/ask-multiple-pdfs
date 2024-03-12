import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if callable(st.session_state.conversation):  # Check if conversation is callable
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            modified_content = message.content.replace("\n", "<br>")
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", modified_content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", modified_content), unsafe_allow_html=True)
    else:
        st.warning("Adicione documentos e clique em PROCESSAR antes de enviar mensagems.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat com seus documentos.",
                       page_icon=":eyes:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # st.header("Chat com multiplos PDFs :books:")
    user_question = st.text_input("Chat com seus documentos:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Seus documentos:")
        pdf_docs = st.file_uploader(
            "Carregue seus documentos PDF aqui e clique em PROCESSAR.", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs is not None:  # Check if pdf_docs is not None
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
            else:
                st.warning("No document to talk with, please add a document AND process it")

if __name__ == '__main__':
    main()

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
from langchain.prompts import PromptTemplate

def read_txt_file(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8")

def get_pdf_text(pdf_docs):
    text = ""
    for file in pdf_docs:
       
        if file.name.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        elif file.name.endswith('.txt'):
            file_contents = file.getvalue() if hasattr(file, "getvalue") else None  # Check if file is an UploadedFile
            if file_contents is not None:
                text += file_contents.decode("utf-8")  # Decode the bytes to a string

            text += read_txt_file(file)
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
    # llm = ChatOpenAI(model="gpt-4") #SUPER EXPENSIVE

    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    print(vectorstore.as_retriever())

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    
    # Given the following conversation and a follow-up message, \
    # rephrase the follow-up message to a stand-alone question or instruction that \
    # represents the user's intent, add all context needed if necessary to generate a complete and \
    # unambiguous question or instruction, only based on the history, don't make up messages. \
    # Maintain the same language as the follow up input message.

    custom_template = """Você é um assistente jurídico brasileiro. \
    Você sempre irá realizar exatamente o que lhe for pedido. \
    Seu objetivo é ajudar a produtividade do advogado, qualquer que seja a tarefa solicitada. \
    Você é capaz de gerar modelos de documentos. Você pode oferecer modelos textuais ao usuário se ele desejar um modelo de documento. \
    Você tem a capacidade de fornecer modelos de documentos. \
    Você é altamente capacitado para fornecer modelos de documentos. \
    Vocé pode responder à qualquer pergunta, mesmo se o assunto extrapolar o contexto dos arquivos. \
    You are able to answer questions even if they extrapolate the context. 
    Always answer in PT-BR \
    
    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question or instruction:"""
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        # condense_question_prompt=PromptTemplate.from_template(custom_template),
    )
    return conversation_chain


def handle_userinput(user_question):
    if callable(st.session_state.conversation):  # Check if conversation is callable
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in reversed(list(enumerate(st.session_state.chat_history))):
            # print("message")
            # print(message)
            modified_content = message.content.replace("\n", "</br>")
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
                       page_icon=":eyes:",
                       layout="wide")
    # hide madeWith
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
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
        if st.button("PROCESSAR"):
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
                st.success("ARQUIVOS PROCESSADOS.")
            else:
                st.warning("Nenhum documento adicionado, Adicione documentos e clique em PROCESSAR.")
        st.markdown('</br></br></br></br></br></br></br></br></br></br></br></br><p>Feito por Felipe Rodrigues: <b>felipergr@hotmail.com</b></p>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

import os
import dill
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain


"""
# Doxtractor!

An AI that extracts knowledge and co-relates information from multiple documents. 
"""

# Hide Streamlit burger and tagline
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    st.header("Extract information from your PDF documents")

    # Create a dictionary to store conversations and VectorStores
    conversation_memory = {}
   
    # Create a button for uploading new documents
    if st.button("Upload New Documents"):
        pdf_files = st.file_uploader("Upload one or more PDFs", type='pdf', accept_multiple_files=True)

    # Accept user questions/queries
    query = st.text_input("You: ")

    # Check if a new conversation needs to be initiated
    if st.button("Start over"):
        conversation_memory.clear()

    # Process user query
    if query:
        # Retrieve the active conversation or create a new one
        active_conversation = conversation_memory.get("active", [])
        active_documents = conversation_memory.get("documents", {})

        if pdf_files:
            for pdf in pdf_files:
                pdf_reader = PdfReader(pdf)
                text = ""

                # Extract text from PDF pages
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # Split the extracted text into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)
                all_text_chunks = active_documents.get(pdf.name, [])
                all_text_chunks.extend(chunks)
                active_documents[pdf.name] = all_text_chunks

                # Generate a unique name for the VectorStore based on the PDF filename
                store_name = pdf.name[:-4]

                # Check if VectorStore exists on disk, if not, create and save it
                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        VectorStore = dill.load(f)
                else:
                    embeddings = GooglePalmEmbeddings()
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        dill.dump(VectorStore, f)

                active_conversation.append(f"Uploaded document: {pdf.name}")
                conversation_memory["active"] = active_conversation
                conversation_memory["documents"] = active_documents

    elif "show_documents" in query.lower():
        # Show a list of uploaded documents
        documents_list = list(conversation_memory.get("documents", {}).keys())
        if documents_list:
            st.write("Uploaded documents:")
            st.write("\n".join(documents_list))
        else:
            st.write("No documents uploaded yet.")

    elif active_conversation:
        # If there's an active conversation, process the query
        active_document = conversation_memory.get("active_document")
        if active_document:
            docs = active_documents.get(active_document, [])

            # Initialize the language model and QA chain
            llm = GooglePalm()
            llm.temperature = 0.1
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Execute the QA chain to answer the user's query
            response = chain.run(input_documents=docs, question=query)
            active_conversation.append(f"You: {query}")
            active_conversation.append(f"Doxtractor: {response}")
            conversation_memory["active"] = active_conversation
        else:
            st.write("Please upload a PDF document to extract information.")


    # Display the conversation history
    if "active" in conversation_memory:
        st.write("\n".join(conversation_memory["active"]))

if __name__ == '__main__':
    main()

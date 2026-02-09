from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import os
import shutil
from dotenv import load_dotenv

load_dotenv() 

# ==================== DATA CLEANING ====================
def clean_html(html: str) -> str:
    """VVIT-optimized HTML cleaner - KEEPS ALL important content"""
    soup = BeautifulSoup(html, "lxml")
    
    # Remove only dangerous tags
    for tag in soup.find_all(['script', 'style', 'noscript']):
        tag.decompose()
    
    # Extract text with structure preserved
    text = soup.get_text(separator="\n", strip=True)
    
    # Simple filtering - keep most content
    lines = []
    if not text or len(text.strip()) < 10:
        return text
    
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    
    # Return all extracted content
    return "\n".join(lines)


# ==================== DATA LOADING ====================
important_urls = [
    "https://www.vvitguntur.com/about/about-us",
    "https://www.vvitguntur.com/about/keypersons",
    "https://www.vvitguntur.com/about/vision",
    "https://www.vvitguntur.com/about/people",
    "https://www.vvitguntur.com/about/location",
    "https://www.vvitguntur.com/adm/previous-rank-info",
    "https://www.vvitguntur.com/departments/cse-home",
    "https://www.vvitguntur.com/cse-faculty",
    "https://www.vvitguntur.com/departments/ai-ml/about-cse-ai-ml",
    "https://www.vvitguntur.com/departments/ai-ds/about-aids",
    "https://www.vvitguntur.com/departments/ai-ds/aids-faculty",
    "https://www.vvitguntur.com/departments/cse-iot/about-cse-iot",
    "https://www.vvitguntur.com/departments/ece-home",
    "https://www.vvitguntur.com/ece-faculty",
    "https://www.vvitguntur.com/departments/eee-home",
    "https://www.vvitguntur.com/eee-faculty",
    "https://www.vvitguntur.com/departments/it-home",
    "https://www.vvitguntur.com/departments/mech-home",
    "https://www.vvitguntur.com/mech-faculty",
    "https://www.vvitguntur.com/departments/civil-home",
    "https://www.vvitguntur.com/civil-faculty",
    "https://www.vvitguntur.com/campus-life/sac/about-sac",
    "https://www.vvitguntur.com/campus-life/ncc/ncc-prof",
    "https://www.vvitguntur.com/campus-life/nss/nss-profile",
    "https://www.vvitguntur.com/campus-life/ncc/nccactivities",
    "https://www.vvitguntur.com/campus-life/nss/nss-events",
    "https://www.vvitguntur.com/placements-home/placement-recruiters",
    "https://www.vvitguntur.com/placements-home/year-wise-placements",
    "https://www.vvitguntur.com/placements-home/placements-branch",
    "https://www.vvitguntur.com/placements-home/placements-team",
    "https://www.vvitguntur.com/placements-home/placements-mou",
    "https://www.vvitguntur.com/aca-examcell/exam-cell-profile",
    "https://www.vvitguntur.com/facilities/google-codelabs",
    "https://www.vvitguntur.com/facilities/central-library",
    "https://www.vvitguntur.com/facilities/hostels",
    "https://www.vvitguntur.com/about/committees/statutory-committees/governing-body/principal",
    "https://www.vvitguntur.com/about/committees/statutory-committees/governing-body/chairman",
    "https://www.vvitguntur.com/about/committees/statutory-committees/governing-body/secretary",
    "https://www.vvitguntur.com/about/committees/statutory-committees/governing-body/ambassador",
    "https://www.vvitguntur.com/vrc-profile",
    "https://www.vvitguntur.com/ksb-profile",
    "https://www.vvitguntur.com/mrntagore-profile",
    "https://www.vvitguntur.com/skp-profile-2",
    "https://www.vvitguntur.com/dak-profile",
    "https://www.vvitguntur.com/tsb-profile",
    "https://www.vvitguntur.com/drtsr-profile",
    "https://www.vvitguntur.com/mech-placements",
    "https://www.vvitguntur.com/cse-placements",
    "https://www.vvitguntur.com/departments/ai-ds/aid-placements",
    "https://www.vvitguntur.com/ece-placements",
    "https://www.vvitguntur.com/eee-placements",
    "https://www.vvitguntur.com/it-placements",
    "https://www.vvitguntur.com/117-clubs",
    "https://www.vvitguntur.com/civil-placements"
]

docs = []
PERSIST_DIRECTORY = "./chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Check if vectorstore exists and has documents
vectorstore_dir_exists = os.path.exists(PERSIST_DIRECTORY)
needs_rebuild = True

if vectorstore_dir_exists:
    try:
        temp_vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="vvit_knowledge_base"
        )
        doc_count = temp_vectorstore._collection.count()
        if doc_count > 0:
            needs_rebuild = False
            vectorstore = temp_vectorstore
            print(f"[LOADED] Existing ChromaDB with {doc_count} documents")
    except Exception as e:
        print(f"[WARNING] Error loading existing DB, will rebuild: {e}")

if needs_rebuild:
    print("[LOADING] Loading documents from URLs...")
    for url in important_urls:
        try:
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()
            
            # FIX 2: ADD RICH METADATA
            for doc in loaded_docs:
                doc.metadata['college'] = 'VVIT Guntur'
                # Determine document type based on URL
                if 'principal' in url or 'chairman' in url or 'keypersons' in url:
                    doc.metadata['type'] = 'leadership'
                elif 'about' in url:
                    doc.metadata['type'] = 'administration'
                else:
                    doc.metadata['type'] = 'general'
                
            docs.extend(loaded_docs)
            print(f"[OK] Loaded: {url}")
        except Exception as e:
            print(f"[FAIL] Failed: {url} - {e}")

    print(f"\n[TOTAL] Total documents loaded: {len(docs)}")

    if len(docs) > 0:
        # ==================== FIX 3: BETTER CHUNKING STRATEGY ====================
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        split_docs = text_splitter.split_documents(docs)
        print(f"[CHUNKS] Total chunks created: {len(split_docs)}")

        # ==================== CHROMADB VECTOR STORE ====================
        print("[BUILDING] Creating ChromaDB vector store...")
        if vectorstore_dir_exists:
            shutil.rmtree(PERSIST_DIRECTORY)
        
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="vvit_knowledge_base"
        )
        vectorstore.persist()
        print(f"[SUCCESS] ChromaDB created with {vectorstore._collection.count()} documents")
    else:
        print("[ERROR] No documents loaded from URLs. Cannot initialize vectorstore.")
        import sys
        sys.exit(1)


# ==================== FIX 4: CONFIGURE RETRIEVER WITH OPTIMAL K ====================
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)


# ==================== FIX 5: PROPER PROMPT TEMPLATE ====================
prompt_template = """You are CollegeBot, a friendly, confident campus assistant for
Vasireddy Venkatadri Institute of Technology (VVIT), Guntur.

Your goal is to help users by answering questions naturally and professionally,
as a knowledgeable college representative would.

Use the information provided below to answer the question.
Do NOT mention the context, documents, sources, or knowledge base in your reply.

Guidelines:
1. If the answer is clearly available, respond directly and confidently.
2. If partial information is available, provide the best complete answer possible.
3. If exact details are not available, give a general helpful response instead of refusing.
4. If something is truly unknown, respond politely and naturally without technical explanations.
5. NEVER say phrases like:
   - "not mentioned in the context"
   - "not found in the documents"
   - "VVIT knowledge base"
6. Do NOT invent facts, names, numbers, or dates.
7. Keep responses concise, clear, and student-friendly.
8. Sound like a human assistant, not a system or rule engine.

Information:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# ==================== LLM SETUP ====================
# Get API key from environment variable
api_key = os.getenv("SARVAM_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.sarvam.ai/v1",
    model="sarvam-m",
    temperature=0.1,
    max_tokens=500
)


# ==================== FIX 6: PROPER QA CHAIN CONFIGURATION ====================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context",
        "verbose": False
    },
    verbose=False
)


# ==================== QUERY NORMALIZATION ====================
def normalize_query(q):
    q = q.lower().strip()
    q = q.replace("chair person", "chairman")
    q = q.replace("head of college", "principal")
    q = q.replace("hod ", "head of department ")
    return q


# ==================== ANSWER FUNCTION FOR API ====================
def answer_question(question: str):
    """
    Main function to answer questions using RAG pipeline
    Takes a question string and returns the answer with sources
    """
    try:
        normalized_query = normalize_query(question)
        response = qa_chain.invoke({"query": normalized_query})
        
        return {
            "status": "success",
            "question": question,
            "answer": response['result'],
            "sources": len(response['source_documents']),
            "source_details": [
                {
                    "url": doc.metadata.get('source', 'N/A'),
                    "type": doc.metadata.get('type', 'unknown'),
                    "preview": doc.page_content[:200]
                }
                for doc in response['source_documents'][:3]
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "question": question,
            "answer": f"Error processing question: {str(e)}",
            "sources": 0,
            "source_details": []
        }


# ==================== TESTING ====================
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Who is the principal of VVIT?",
        "Who is the chairman of VVIT?",
        "Tell me about VVIT leadership",
        "What departments are available in VVIT?"
    ]
    
    print("\n" + "="*70)
    print("VVIT RAG CHATBOT - TESTING")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {query}")
        print(f"{'='*70}")
        
        try:
            normalized_query = normalize_query(query)
            response = qa_chain.invoke({"query": normalized_query})
            
            print(f"\n[ANSWER]:\n{response['result']}")
            print(f"\n[SOURCES] Retrieved {len(response['source_documents'])} documents")
            
            # Show sources for debugging
            if response['source_documents']:
                print(f"\n[SOURCE DETAILS]:")
                for i, doc in enumerate(response['source_documents'][:3], 1):
                    source = doc.metadata.get('source', 'N/A')
                    doc_type = doc.metadata.get('type', 'unknown')
                    print(f"  {i}. Source: {source}")
                    print(f"     Type: {doc_type}")
                    print(f"     Preview: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"[ERROR]: {str(e)}")
            import traceback
            traceback.print_exc()
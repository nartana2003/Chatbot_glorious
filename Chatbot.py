# import os
# import time
# from dotenv import load_dotenv

# import pinecone
# from pinecone import Pinecone, ServerlessSpec

# #import langchain
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# #Ingestion pipeline--------------------

# # 1. ERROR HANDLING FOR ENVIRONMENT SETUP
# try:
#     load_dotenv()
#     print("✓ Environment variables loaded successfully")
# except Exception as e:
#     print(f"❌ Error loading environment variables: {e}")
#     exit(1)

# # 2. ERROR HANDLING FOR PINECONE CONNECTION
# try:
#     pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
#     index_name = "langchain-glorious-chatbot-index"
#     index = pc.Index(index_name)
#     print("✓ Pinecone connection established successfully")
# except Exception as e:
#     print(f"❌ Error connecting to Pinecone: {e}")
#     exit(1)

# # 3. ERROR HANDLING FOR EMBEDDINGS INITIALIZATION
# try:
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-large",
#         api_key=os.environ.get("OPENAI_API_KEY")
#     )
#     vector_store = PineconeVectorStore(index=index, embedding=embeddings)
#     print("✓ Embeddings and vector store initialized successfully")
# except Exception as e:
#     print(f"❌ Error initializing embeddings/vector store: {e}")
#     exit(1)

# # 4. ERROR HANDLING FOR PDF LOADING
# try:
#     loader = PyPDFDirectoryLoader("C:\\Users\\narta\\OneDrive\\Desktop\\Practice Python\\Chatbot-Glorious\\AI Training")
#     raw_documents = loader.load()
#     print(f"✓ Loaded {len(raw_documents)} PDF documents successfully")
    
#     if not raw_documents:
#         print("⚠️  Warning: No documents found in the specified directory")
#         exit(1)
        
# except FileNotFoundError:
#     print("❌ Error: PDF directory not found. Please check the path.")
#     exit(1)
# except Exception as e:
#     print(f"❌ Error loading PDF documents: {e}")
#     exit(1)

# # 5. ERROR HANDLING FOR TEXT SPLITTING
# try:
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=400,
#         length_function=len,
#         is_separator_regex=False,
#     )
    
#     documents = text_splitter.split_documents(raw_documents)
#     print(f"✓ Split documents into {len(documents)} chunks successfully")
    
#     if not documents:
#         print("⚠️  Warning: No document chunks created")
#         exit(1)
        
# except Exception as e:
#     print(f"❌ Error splitting documents: {e}")
#     exit(1)

# # 6. ERROR HANDLING FOR BATCH PROCESSING
# print("Starting batch processing...")

# max_tokens = 30000
# batch_size = 3
# batch_docs = []
# batch_ids = []
# total_tokens = 0
# uuid_counter = 1
# successful_batches = 0
# failed_batches = 0

# def add_batch_to_vector_store(batch_docs, batch_ids):
#     """Helper function to add batch with error handling"""
#     global successful_batches, failed_batches
    
#     try:
#         vector_store.add_documents(documents=batch_docs, ids=batch_ids)
#         successful_batches += 1
#         print(f"✓ Successfully added batch {successful_batches} with {len(batch_docs)} documents")
#         return True
#     except Exception as e:
#         failed_batches += 1
#         print(f"❌ Error adding batch {successful_batches + failed_batches}: {e}")
        
#         # Optionally, try to add documents one by one to identify problematic ones
#         print("  Attempting to add documents individually...")
#         for i, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
#             try:
#                 vector_store.add_documents(documents=[doc], ids=[doc_id])
#                 print(f"  ✓ Added individual document {doc_id}")
#             except Exception as individual_error:
#                 print(f"  ❌ Failed to add document {doc_id}: {individual_error}")
        
#         return False

# # 7. MAIN BATCH PROCESSING LOOP WITH ERROR HANDLING
# try:
#     for i, doc in enumerate(documents):
#         doc_tokens = len(doc.page_content)
        
#         # Check if adding this doc would exceed either batch size or token limit
#         if (len(batch_docs) >= batch_size) or (total_tokens + doc_tokens > max_tokens):
#             # Flush current batch
#             if batch_docs:  # Only process if there are documents in batch
#                 add_batch_to_vector_store(batch_docs, batch_ids)
                
#                 # Add a small delay to avoid rate limiting
#                 time.sleep(0.1)
            
#             # Reset batch
#             batch_docs = []
#             batch_ids = []
#             total_tokens = 0
        
#         # Add doc to batch
#         batch_docs.append(doc)
#         batch_ids.append(f"id{uuid_counter}")
#         total_tokens += doc_tokens
#         uuid_counter += 1
        
#         # Progress update
#         if i % 10 == 0:
#             print(f"  Progress: {i}/{len(documents)} documents processed")
    
#     # 8. ERROR HANDLING FOR FINAL BATCH
#     if batch_docs:
#         print("Processing final batch...")
#         add_batch_to_vector_store(batch_docs, batch_ids)
    
#     # 9. FINAL SUMMARY
#     print("\n" + "="*50)
#     print("INGESTION COMPLETE")
#     print("="*50)
#     print(f"✓ Total documents processed: {len(documents)}")
#     print(f"✓ Successful batches: {successful_batches}")
#     if failed_batches > 0:
#         print(f"❌ Failed batches: {failed_batches}")
#     print(f"✓ Total unique IDs created: {uuid_counter - 1}")
    
# except KeyboardInterrupt:
#     print("\n⚠️  Process interrupted by user")
#     print(f"Processed {successful_batches} successful batches before interruption")
# except Exception as e:
#     print(f"❌ Unexpected error during batch processing: {e}")
#     print(f"Processed {successful_batches} successful batches before error")

# print("\nIngestion pipeline completed!")


# #Retrieval pipeline--------------------

# #initialize pinecone database
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# # set the pinecone index
# index_name = "langchain-glorious-chatbot-index"
# index = pc.Index(index_name)

# # initialize embeddings model + vector store
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k":2, "score_threshold": 0.6},
# )
# results = retriever.invoke("I want to conduct a mini mental state exam. What questions do I ask?")

# print("RESULTS:")

# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")



 
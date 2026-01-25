import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

csv_filename = 'quran_bilingual_data.csv'
print(f"Loading the bilingual data file ({csv_filename})...")
try:
    df = pd.read_csv(csv_filename)
    df.fillna("", inplace=True)
except FileNotFoundError:
    print(f"Error: '{csv_filename}' not found. Please run 'prepare_data.py' first.")
    exit()

df['page_content'] = (
    "Reference: " + df['reference'].astype(str) + "\n"
    + "Arabic: " + df['arabic'] + "\n"
    + "English Translation: " + df['translation_english']
)

loader = DataFrameLoader(df, page_content_column='page_content')
documents = loader.load()
print(f"{len(documents)} ayaat loaded as documents.")


persist_directory = "./quran_bilingual_db"

if os.path.exists(persist_directory):
    print("Loading existing bilingual database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Creating new bilingual database. This might take a few minutes...")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
print("Bilingual database is ready!")


prompt_template = """
You are an expert and respectful Quranic Assistant. Your task is to follow a strict, step-by-step process to answer the user's question based ONLY on the context.

**Your Thought Process (Follow these steps internally):**
1.  **Step 1: Identify Language.** Analyze the user's `Question` to determine if it is in English or Arabic. This decision is critical and will control the language of your entire response.
2.  **Step 2: Synthesize a Summary.** Based on the language identified in Step 1, carefully read the user's question and understand it and then read the `Context` and formulate a 3-4 line summary that directly answers the `Question`.
3.  **Step 3: Format Detailed Points.** Create a numbered list of key points from the `Context`. For each point, you must follow these sub-rules precisely:
    -   **Sub-rule 3a:** If the identified language was English, you MUST use the "English Translation" from the context for the `Translation:` field.
    -   **Sub-rule 3b:** If the identified language was Arabic, you MUST use the Arabic text from the context for the `Translation:` field.
    -   **Sub-rule 3c:** The `Explanation:` must be in the same language as the `Question`.

**Final Output Format (Use this exact structure for your response):**
- **Introductory Summary:** [Your summary from Step 2]
- **Detailed Points:** [Your numbered list from Step 3, following all sub-rules]
    1.  **Translation:**
        > (The appropriate translation text goes here, inside a blockquote.)
        **Reference:** `[The verse reference, e.g., 2:153]`
        
        **Explanation:** (Your 1-2 line explanation for this point goes here.)

    2.  **Translation:**
        > (The second translation text goes here.)
        **Reference:** `[The second verse reference]`
        
        **Explanation:** (The explanation for the second point.)
    (and so on...Try to give as much points as you can generate)

**Context from Database:**
{context}

**User's Question:**
{question}

**Your Final Answer (Strictly follow the format above):**
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n--- Final Quranic Assistant (Single DB Version) is Ready ---")
while True:
    user_question = input("\nYour question (Arabic or English): ")
    if user_question.lower() == 'exit':
        break
    
    answer = rag_chain.invoke(user_question)
    print("\n--- Jawab ---\n")
    print(answer)
    print("\n------------------\n")
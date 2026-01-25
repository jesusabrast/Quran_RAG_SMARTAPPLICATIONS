import streamlit as st
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

# --- Language Texts ---
texts = {
    'en': {
        'title': 'QURAN WITH AI',
        'subtitle': 'القُرْآن الذَكِيّ— Your AI assistant for the Quran',
        'sidebar_header': 'About QURAN WITH AI',
        'sidebar_text': '''
        QURAN WITH AI is a respectful assistant built to help you read and reflect on the Quran.

        **How it works:**
        1. Ask a question in Arabic or English.
        2. The system retrieves relevant Quranic verses and translations.
        3. It produces a concise, referenced answer with verse references.

        **Data Sources:**
        - Arabic Text: Tanzil.net
        - English Translation: Provided in the `files/` folder
        ''',
        'success_msg': 'Use the chat box below to ask about verses, themes, or translations.',
        'design_note': '**Design:** Clean green palette, clear typography, and Islamic iconography for clarity and respect.',
        'initial_message': 'As-salamu alaykum! How can I help you explore the Quran today?',
        'chat_placeholder': 'Ask a question about the Quran...',
        'language_label': 'Language / اللغة',
        'language_options': ['English', 'العربية']
    },
    'ar': {
        'title': 'القرآن بالذكاء الاصطناعي',
        'subtitle': 'القُرْآن الذَكِيّ— مساعدك الذكي للقرآن',
        'sidebar_header': 'حول القرآن بالذكاء الاصطناعي',
        'sidebar_text': '''
        القرآن بالذكاء الاصطناعي هو مساعد محترم مبني لمساعدتك في قراءة وتأمل القرآن.

        **كيف يعمل:**
        1. اسأل سؤالاً بالعربية أو الإنجليزية.
        2. يقوم النظام باسترداد الآيات القرآنية ذات الصلة والترجمات.
        3. ينتج إجابة موجزة مع مراجع الآيات.

        **مصادر البيانات:**
        - النص العربي: Tanzil.net
        - الترجمة الإنجليزية: متوفرة في مجلد `files/`
        ''',
        'success_msg': 'استخدم مربع الدردشة أدناه للسؤال عن الآيات أو المواضيع أو الترجمات.',
        'design_note': '**التصميم:** لوحة ألوان خضراء نظيفة، خطوط واضحة، ورموز إسلامية للوضوح والاحترام.',
        'initial_message': 'السلام عليكم! كيف يمكنني مساعدتك في استكشاف القرآن اليوم؟',
        'chat_placeholder': 'اسأل سؤالاً عن القرآن...',
        'language_label': 'اللغة / Language',
        'language_options': ['العربية', 'English']
    }
}

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="QURAN WITH AI",
    page_icon="☪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Update messages if language changed
if st.session_state.get('prev_language') != st.session_state.language:
    st.session_state.messages = [{"role": "assistant", "content": texts[st.session_state.language]['initial_message']}]
    st.session_state.prev_language = st.session_state.language

# --- 2. Custom CSS for Theming and Design ---
direction_css = "direction: rtl; text-align: right;" if st.session_state.language == 'ar' else "direction: ltr; text-align: left;"
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Amiri&display=swap');

    /* Main background with soft cream and subtle pattern */
    .stApp {{
        background-color: #f5f8f4; /* Soft cream */
        background-image: radial-gradient(circle at 10% 10%, rgba(77, 136, 94, 0.02), transparent 10%),
                          radial-gradient(circle at 90% 90%, rgba(212, 175, 55, 0.01), transparent 10%);
        color: #163a2b; /* Deep green text */
        font-family: 'Merriweather', serif;
        {direction_css}
    }}

    /* Main title font and color */
    h1 {{
        font-family: 'Amiri', serif;
        color: #0b6b3a; /* Emerald green */
        text-align: center;
        padding-top: 0.6rem;
        margin-bottom: 0;
    }}

    /* Subtitle style */
    .subtitle {{
        font-family: 'Merriweather', serif;
        color: #3d6b54;
        text-align: center;
        font-size: 1.05rem;
        margin-top: 0.25rem;
    }}
    
    /* Sidebar background */
    .css-1d391kg {{ /* sidebar container selector fallback */
        background-color: #ffffffcc;
        border-right: 1px solid rgba(11,107,58,0.08);
    }}

    /* Chat messages */
    .st-chat-message.user {{
        background: linear-gradient(90deg, rgba(11,107,58,0.06), rgba(11,107,58,0.03));
        border-left: 4px solid #0b6b3a;
        border-radius: 10px;
        padding: 0.6rem;
    }}
    .st-chat-message.assistant {{
        background: #ffffff;
        border: 1px solid rgba(11,107,58,0.06);
        border-radius: 10px;
        padding: 0.6rem;
    }}
    
    /* Input styling */
    .stTextArea, .stTextInput, .st-chat-input {{
        background-color: #ffffff;
        border: 1px solid rgba(11,107,58,0.08);
        border-radius: 10px;
        padding: 0.4rem;
    }}

    .stMarkdown h3 {{
        color: #0b6b3a; /* deep green for headings */
        border-bottom: 2px solid rgba(11,107,58,0.06);
        padding-bottom: 5px;
    }}
    .stMarkdown blockquote {{
        background-color: rgba(11,107,58,0.03);
        border-left: 4px solid #0b6b3a;
        padding: 0.5rem 1rem;
        margin-left: 0;
        border-radius: 5px;
    }}

    /* Card-like container for the main chat area */
    .main-container {{
        background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.9));
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(11,107,58,0.04);
    }}
</style>
""", unsafe_allow_html=True)


# --- 3. Cached Functions for Heavy Lifting ---
@st.cache_resource
def load_rag_chain():
    load_dotenv()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

    csv_filename = 'quran_bilingual_data.csv'
    if not os.path.exists(csv_filename):
        st.error(f"CRITICAL ERROR: The data file '{csv_filename}' was not found.")
        st.stop()
    
    df = pd.read_csv(csv_filename)
    df.fillna("", inplace=True)

    df['page_content'] = "Reference: " + df['reference'].astype(str) + "\n" + \
                         "Arabic Text: " + df['arabic'] + "\n" + \
                         "English Translation: " + df.get('translation_english', '')
    
    loader = DataFrameLoader(df, page_content_column='page_content')
    documents = loader.load()

    persist_directory = "./quran_bilingual_db"
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        with st.spinner("Creating new multilingual database. This might take a few minutes..."):
            vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
    
    # --- New and Improved Prompt Template for Better Formatting ---
    prompt_template = """
    You are an expert and respectful Quranic Assistant. Your task is to follow a strict, step-by-step process to answer the user's question based ONLY on the context, using precise Markdown formatting.

    **Your Thought Process (Follow these steps internally):**
1.  **Step 1: Identify Language.** Analyze the user's `Question` to determine if it is in English or Arabic. This decision is critical and will control the language of your entire response.
2.  **Step 2: Synthesize a Summary.** Based on the language identified in Step 1, carefully read the user's question and understand it and then read the `Context` and formulate a 3-4 line summary that directly answers the `Question`.
3.  **Step 3: Format Detailed Points.** Create a numbered list of key points from the `Context`. For each point, you must follow these sub-rules precisely:
    -   **Sub-rule 3a:** If the identified language was English, you MUST use the "English Translation" from the context for the `Translation:` field.
    -   **Sub-rule 3b:** If the identified language was Arabic, you MUST use the "Arabic Text" from the context for the `Translation:` field.
    -   **Sub-rule 3c:** The `Explanation:` must be in the same language as the `Question`.

    ---

    ### Detailed Points
    (Create a numbered list of key points below.)

    1.  **Translation:**
        > (The appropriate translation/text goes here, inside a blockquote.)
        **Reference:** `[The verse reference, e.g., 2:153]`
        
        **Explanation:** (Your 1-2 line explanation for this point goes here.)

    2.  **Translation:**
        > (The second translation/text goes here.)
        **Reference:** `[The second verse reference]`
        
        **Explanation:** (The explanation for the second point.)
    
    (and so on...Try to give as much points as you can generate)

    **Context from Database:**
    {context}

    **User's Question:**
    {question}

    **Your Final Answer (Strictly follow the Markdown format above):**
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 4. Main App Interface ---

# Load the RAG chain (fast due to caching)
rag_chain = load_rag_chain()

# Sidebar for information
with st.sidebar:
    # Language selection
    lang_options = texts[st.session_state.language]['language_options']
    lang_index = 0 if st.session_state.language == 'en' else 1
    selected_lang = st.selectbox(texts[st.session_state.language]['language_label'], lang_options, index=lang_index)
    if selected_lang == 'العربية' or selected_lang == 'Arabic':
        st.session_state.language = 'ar'
    else:
        st.session_state.language = 'en'
    
    st.header(texts[st.session_state.language]['sidebar_header'])
    st.markdown(texts[st.session_state.language]['sidebar_text'])
    st.success(texts[st.session_state.language]['success_msg'])
    st.markdown("---")
    st.markdown(texts[st.session_state.language]['design_note'])

# Main page title
# Header with Islamic SVG mark and title
st.markdown(f"""
<div style='text-align:center; margin-top: 8px;'>
    <svg width='86' height='86' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>
        <circle cx='12' cy='12' r='10' fill='#0b6b3a' opacity='0.12'/>
        <path d='M12 2C13.1046 2 14 2.89543 14 4C14 5.10457 13.1046 6 12 6C10.8954 6 10 5.10457 10 4C10 2.89543 10.8954 2 12 2Z' fill='#0b6b3a'/>
        <path d='M12 6C13.6569 6 15 7.34315 15 9C15 12 9 12 9 9C9 7.34315 10.3431 6 12 6Z' fill='#0b6b3a'/>
        <path d='M7 12C7 12 9 10 12 10C15 10 17 12 17 12V18H7V12Z' fill='#0b6b3a'/>
    </svg>
    <h1>{texts[st.session_state.language]['title']}</h1>
    <p class='subtitle'>{texts[st.session_state.language]['subtitle']}</p>
</div>
    """, unsafe_allow_html=True)
st.markdown("---")
st.markdown(texts[st.session_state.language]['design_note'])

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(texts[st.session_state.language]['chat_placeholder']):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verses..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response, unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("</div>", unsafe_allow_html=True)
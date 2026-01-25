# Quran AI Chatbot with Bilingual UI

**Created by Essa Rastanawi**

## Description

This project is an AI-powered Quran chatbot with a bilingual user interface supporting English and Arabic. It features a language toggle button in the sidebar that switches between Left-to-Right (LTR) layout for English and Right-to-Left (RTL) layout for Arabic, with all UI texts translated accordingly. The chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on Quranic verses and translations.

## Features

- **Bilingual Support**: Toggle between English and Arabic interfaces.
- **RTL/LTR Layouts**: Automatic direction switching for proper text display.
- **RAG-Based Assistant**: Uses OpenAI GPT-4o for intelligent responses.
- **Multilingual Embeddings**: Supports queries in both Arabic and English.
- **Persistent Vector Database**: ChromaDB for efficient verse retrieval.
- **Streamlit UI**: Clean, responsive web interface.

## Quick Start

1. **Create & Activate Virtual Environment**

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Bilingual Dataset**

   ```bash
   python files/prepare_data.py
   ```

4. **Add OpenAI API Key**

   Create a `.env` file in the project root and add:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the App**

   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

- `streamlit_app.py` — Main Streamlit application with bilingual UI.
- `RAG_SYSTEM.py` — RAG chain implementation (if separate).
- `quran_bilingual_data.csv` — Generated bilingual dataset.
- `quran_bilingual_db/` — ChromaDB vector store.
- `files/`
  - `quran-simple.txt` — Arabic Quranic text.
  - `en.maududi.txt` — English translation.
  - `prepare_data.py` — Script to build the CSV dataset.
- `requirements.txt` — Python dependencies.
- `.env` — Environment variables (not committed).

## Usage

- Select your preferred language using the toggle in the sidebar.
- Ask questions about Quranic verses, themes, or translations in Arabic or English.
- The app detects the query language and responds accordingly.

## Notes

- Do not commit your `.env` file.
- To rebuild the vector database, delete the `quran_bilingual_db` folder and restart the app.
- The app uses GPT-4o; ensure your OpenAI API key has sufficient credits.

## License

MIT License

## Copyright

© 2025 Essa Rastanawi w. All rights reserved.

## Acknowledgments

- Quranic texts from Tanzil.net
- Powered by LangChain, Streamlit, ChromaDB, and OpenAI


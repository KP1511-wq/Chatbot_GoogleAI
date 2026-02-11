# ❤️ Heart Disease Data Assistant (Chatbot)

A natural language SQL chatbot that allows users to query a Heart Disease database using plain English. 

Built with **FastAPI** (Backend), **Streamlit** (Frontend), **LangGraph**, and **Google Gemini AI**.

## 🚀 Features

* **Text-to-SQL:** Automatically converts English questions (e.g., *"Show me the average age of patients"*) into SQL queries.
* **Data Visualization:** Returns data in formatted tables.
* **Dual-Architecture:** Separates the logic (Brain) from the interface (UI) using a robust Client-Server model.
* **Context Aware:** Understands specific medical columns like `cp` (Chest Pain) and `thalach` (Max Heart Rate).
* **Google Gemini Powered:** Uses the `gemini-2.5-flash-lite` model for fast, free, and accurate reasoning.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI, Uvicorn
* **AI Orchestration:** LangChain, LangGraph
* **Model:** Google Gemini 2.5 Flash Lite (via `langchain-google-genai`)
* **Database:** SQLite (`heart.db`)
* **Language:** Python 3.10+

---
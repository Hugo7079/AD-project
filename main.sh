pip install -r requirements.txt
ollama pull gemma2
streamlit run interact2_chatbot.py #&
# STREAMLIT_PID=$!
# wait $STREAMLIT_PID
python languagefeatureplotting.py
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# ------------------
# UI & Config
# ------------------
st.set_page_config(page_title="Pro Interviewer", layout="centered")
st.title("Interview Practice Chatbot")
st.caption("AI-Powered Technical Interview & Feedback. Each question has 2 marks")

# ------------------
# Load API key
# ------------------
load_dotenv()
API_KEY = os.getenv("API_KEY") 

# ------------------
# Sidebar
# ------------------
with st.sidebar:
    user_option = st.selectbox(
        "Choose role:",
        ["Python Developer", "SQL Developer", "Java Developer", "Machine Learning", 
         "Deep Learning", "Frontend Developer", "Backend Developer", "Full Stack Developer"]
    )
    difficulty = st.radio("Select difficulty:", ["Easy", "Medium", "Hard"])
    num_questions = st.slider("Number of questions", 2, 10, 5)
        
    if st.button("Start New Interview"):
        st.session_state.clear()
        st.rerun()

# ------------------
# System Prompts
# ------------------
system_prompt = f"You are a STRICT technical interviewer for {user_option} ({difficulty} level). Rules: Ask 1 short question at a time. NO feedback. NO answers. Move to next question immediately."

# ------------------
# Groq API Function
# ------------------
def get_response(messages, temp=0.4):
    try:
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=temp,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------
# NEW & IMPROVED Evaluation Function
# ------------------
def evaluate_performance(history):
    interview_data = ""
    for msg in history:
        if msg["role"] == "assistant":
            interview_data += f"Q: {msg['content']}\n"
        elif msg["role"] == "user":
            interview_data += f"A: {msg['content']}\n"

    # Total questions calculate kar rahe hain (system prompt hata kar)
    total_q = (len(history) - 1) // 2 

    eval_prompt = [
        {"role": "system", "content": f"""You are a Technical Examiner. Grade this {user_option} interview.
        
        SCORING SYSTEM (Per Question):
        - 0 Marks: Wrong or "I don't know".
        - 0.5 Marks: Very poor attempt but knows keywords.
        - 1.0 Marks: Half correct or basic idea.
        - 1.5 Marks: Mostly correct but missing depth.
        - 2.0 Marks: Perfectly correct.

        TOTAL QUESTIONS ASKED: {total_q}
        MAXIMUM POSSIBLE MARKS: {total_q * 2}

        INSTRUCTIONS:
        1. Create a table with: Question, Answer Given, Marks Obtained (0 to 2), and Correct Hint.
        2. Calculate 'Obtained Total' by adding marks of each question.
        3. Final Score Format: [Obtained Total] / [Maximum Marks]
        """},
        {"role": "user", "content": f"Interview Transcript:\n{interview_data}"}
    ]
    
    return get_response(eval_prompt, temp=0.1)

# ------------------
# Session State Init
# ------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": system_prompt}]
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0
if "awaiting_answer" not in st.session_state:
    st.session_state.awaiting_answer = False
if "result" not in st.session_state:
    st.session_state.result = None

# ------------------
# Main Logic
# ------------------

# 1. First Question Trigger
if st.session_state.questions_asked == 0 and not st.session_state.awaiting_answer:
    with st.spinner("Setting up the interview..."):
        first_q = get_response(st.session_state.conversation)
        st.session_state.conversation.append({"role": "assistant", "content": first_q})
        st.session_state.questions_asked = 1
        st.session_state.awaiting_answer = True
        st.rerun()

# 2. Display Chat History
for msg in st.session_state.conversation:
    if msg["role"] == "assistant":
        st.chat_message("assistant", avatar="👨‍🏫").write(msg["content"])
    elif msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])

# 3. Check for Completion & Show Result
if st.session_state.questions_asked > num_questions:
    st.divider()
    st.success(f"🎉 Interview Completed!")
    
    if st.session_state.result is None:
        with st.spinner("Calculating accuracy and generating report..."):
            st.session_state.result = evaluate_performance(st.session_state.conversation)
            st.rerun()
    
    st.subheader("📊 Detailed Performance Report")
    st.markdown(st.session_state.result)
    st.info("Click 'Start New Interview' to try again.")
    st.stop()

# 4. User Input Handling
user_input = st.chat_input("Type your answer here...")

if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    
    if st.session_state.questions_asked < num_questions:
        with st.spinner("Thinking..."):
            next_q = get_response(st.session_state.conversation)
            st.session_state.conversation.append({"role": "assistant", "content": next_q})
            st.session_state.questions_asked += 1
    else:
        st.session_state.questions_asked += 1
    
    st.rerun()





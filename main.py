import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# ------------------
# UI & Config
# ------------------
st.set_page_config(page_title="Pro Interviewer", layout="centered", page_icon="👨‍🏫")
st.title("Interview Practice Chatbot")
st.caption("AI-Powered Technical Interview & Feedback. Each question has 2 marks")

# ------------------
# Load API key
# ------------------
load_dotenv()
API_KEY = os.getenv("API_KEY") 

# ------------------
# Master Detailed Role Topics 
# ------------------
role_topics = {
    "Python Developer": {
        "Easy": "Basics: Syntax, Indentation, Comments. Variables: Naming rules, Type Casting, type(). Data Types: Numbers, String, Boolean. I/O: print(f-strings), input(). Operators: Arithmetic, Relational, Logical, Assignment, Membership, Identity. Control: if-else, nested-if. Loops: for, while, break, continue, pass, range(). Data Structures: List, Tuple, Set, Dict basics, Slicing.",
        "Medium": "Data Structures: List/Dict/Set methods, Nested structures. Strings: Advanced methods. Functions: Default/Keyword args, *args, **kwargs, Lambda, Recursion. Modules: math, random, datetime, Custom modules, pip, venv. File Handling: r/w/a modes, CSV, JSON. Exceptions: try-except-else-finally, Custom errors. OOP: Class, Object, __init__, Inheritance, Polymorphism, Encapsulation.",
        "Hard": "Advanced OOP: Method Overriding, Multiple Inheritance, Operator Overloading, Dunder Methods, ABC. Advanced Concepts: Decorators, Generators, Iterators, Closures, Context Managers, Multithreading, Multiprocessing, Async/Await. Libraries: NumPy, Pandas, Matplotlib. DB: SQLite, MySQL, ORM. Web: Flask, Django, FastAPI. Engineering: PyTest, Git, Design Patterns."
    },
    "SQL Developer": {
        "Easy": "DBMS vs RDBMS, SQL Basics: Syntax, Data Types. DDL: Create/Drop/Truncate. DML: Insert, Select, Update, Delete. Filtering: Where, AND/OR/NOT, Between, In, Like, Is Null. Sorting: Order By, Limit. Aggregates: Count, Sum, Avg, Min, Max. Constraints: Not Null, Unique, Default, Check.",
        "Medium": "Keys: Primary, Foreign, Composite. Joins: Inner, Left, Right, Full, Self, Cross Join. Grouping: Group By, Having. Subqueries: Select/Where subqueries, Correlated, Nested queries. Set Ops: Union, Intersect, Except. Objects: Views, Indexes. Functions: String, Date/Time. Transactions: Commit, Rollback.",
        "Hard": "Advanced: Complex Joins, Recursive Queries, CTE (WITH clause). Window Functions: Row_Number, Rank, Dense_Rank, Lead, Lag, Over, Partition By. Programming: Stored Procedures, Functions, Triggers, Cursors. Optimization: ACID, Deadlocks, Query Optimization, Normalization. Design: ER Diagrams, Schema Design."
    },
    "Java Developer": {
        "Easy": "Basics: JVM, JRE, JDK. Variables: Primitive vs Non-Primitive, Type Casting, final. I/O: Scanner, BufferedReader. Operators: Arithmetic, Relational, Logical, Ternary. Control: If-else, switch, Loops (for, while, do-while). Arrays: 1D and Multi-dimensional.",
        "Medium": "OOP: Class vs Object, Methods, Constructors, this, static. OOP Pillars: Inheritance, Polymorphism, Encapsulation, Abstraction. Access: public, private, protected. Strings: StringBuffer vs StringBuilder. Exceptions: try-catch-finally, custom exceptions. Collections: List, ArrayList, LinkedList, Set, Map.",
        "Hard": "Advanced OOP: Abstract Classes vs Interfaces, Inner/Anonymous Classes. Collections: Iterator, Comparable vs Comparator, Concurrent Collections. Multi-threading: Thread Class vs Runnable, Synchronization, Deadlock. Java 8+: Lambda, Stream API, Optional. Memory: Stack vs Heap, GC. JDBC, Spring Boot, Hibernate."
    },
    "Frontend Developer": {
        "Easy": "Web Basics: Client-Server, HTTP/HTTPS. HTML: Structure, Semantic tags, Forms, Input types. CSS: Box Model, Colors, Fonts, Display (Block/Inline), Position (Relative/Absolute). JS Basics: let/const, Operators, DOM basics, Event Listeners.",
        "Medium": "Advanced HTML: SEO, Audio/Video. CSS: Flexbox, Grid, Pseudo-classes, Transitions, Animations, Media Queries. JS Core: Arrow functions, Array/Object methods, Closures, this keyword, DOM manipulation. API: JSON, Fetch API, LocalStorage. Tools: Git basics, NPM intro.",
        "Hard": "JS Advanced: Event Loop, Promises, Async/Await. Frameworks (React): JSX, Components, Props, State, Hooks (useEffect, useContext), Routing, Redux Toolkit. Performance: Lazy Loading, Code Splitting. Security: XSS, CSRF. Deployment: Vite, Webpack, Vercel/Netlify."
    },
    "Backend Developer": {
        "Easy": "Fundamentals: Client-Server Architecture, Request-Response, HTTP Status Codes. Language: Python or Java basics. Framework Intro: Flask or Spring Boot basic setup. Routing: GET/POST, JSON Response. Database: SQL vs NoSQL basics. Security: Environment Variables.",
        "Medium": "Deep Frameworks: Django ORM or Spring MVC. API: RESTful design, Pagination, Filtering, Versioning. Auth: JWT Authentication, Sessions/Cookies, RBAC. DB: Joins, Indexing, Transactions. Middleware, Logging, File handling. Caching: Redis basics.",
        "Hard": "Architecture: Microservices vs Monolithic, API Gateway. Advanced API: GraphQL, WebSockets. Scalability: Load Balancing, Horizontal Scaling, Connection Pooling. Async: Background jobs (Celery/RabbitMQ). DevOps: Docker, CI/CD, AWS/Azure deployment. Testing: Unit/Integration Testing."
    },
    "Machine Learning": {
        "Easy": "AI vs ML vs DL. Foundation: Python (NumPy, Pandas, Matplotlib). Math: Algebra, Mean/Median/Mode, Variance, Std Deviation, Probability. Basic Algos: Linear Regression, Logistic Regression, KNN intro.",
        "Medium": "Stats: Normal Distribution, Hypothesis Testing, p-value. Preprocessing: Encoding, Scaling, Outlier detection. Algos: Random Forest, SVM, Naive Bayes, K-Means, PCA. Evaluation: Confusion Matrix, Precision/Recall, ROC-AUC, GridSearchCV.",
        "Hard": "Ensemble: XGBoost, CatBoost. Deep Learning: Neural Networks, CNN basics. NLP: Tokenization, TF-IDF, Word Embeddings. MLOps: FastAPI/Docker for ML, Model monitoring. Ethics: Explainable AI (SHAP)."
    },
    "Deep Learning": {
        "Easy": "Basics: Neural Network concept, Weights & Bias. Math: Matrix Multiplication, Derivatives, Gradient Descent basics. Activation Functions: Sigmoid, ReLU, Tanh. Libraries: Keras/PyTorch intro. MLP basics.",
        "Medium": "Deep Networks: Batch Norm, Dropout, Regularization. Optimization: Adam, RMSprop. CNN: Convolution layers, Pooling, Padding. RNN: LSTM, GRU. Transfer Learning: ResNet, VGG.",
        "Hard": "Advanced CNN: YOLO Object Detection. NLP: Transformers, Attention Mechanism, BERT. Generative: GANs, Autoencoders. Large Scale: Distributed training, Quantization. Deployment: TensorFlow Serving."
    },
    "Data Analyst": {
        "Easy": "Data Analyst role. Excel: VLOOKUP, SUM/AVG, Sorting. SQL: SELECT, WHERE, Aggregates. Data Types: Qualitative vs Quantitative. Stats: Mean, Median, Mode, Correlation.",
        "Medium": "Advanced Excel: Pivot Tables, INDEX-MATCH. SQL: Joins, Group By, Subqueries. Python: Pandas (Wrangling), Seaborn. Visualization: Power BI/Tableau Dashboard design. Stats: Sampling, A/B Testing.",
        "Hard": "Advanced SQL: CTE, Window Functions. Advanced Analysis: Regression, Time Series, Forecasting, Cohort Analysis. ETL Process, BigQuery, Data Storytelling, Stakeholder reporting."
    }
}

# ------------------
# Sidebar
# ------------------
with st.sidebar:
    user_option = st.selectbox("Choose role:", list(role_topics.keys()))
    difficulty = st.radio("Select difficulty:", ["Easy", "Medium", "Hard"])
    num_questions = st.slider("Number of questions", 2, 50, 5)
        
    if st.button("Start New Interview"):
        st.session_state.clear()
        st.rerun()

selected_topics = role_topics[user_option][difficulty]

# ------------------
# System Prompts
# ------------------
system_prompt = f"You are a STRICT technical interviewer for {user_option} ({difficulty} level). TOPIC SCOPE: {selected_topics}. Rules: Ask 1 short question at a time. NO feedback at any cost. NO answers. Move to next question immediately."

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

    total_q = (len(history) - 1) // 2 

    eval_prompt = [
        {"role": "system", "content": f"""You are a Technical Examiner. Grade this {user_option} interview.
        
        SCORING SYSTEM (Per Question):
        - 0 Marks: Wrong or "I don't know, sorry, i don't have an idea".
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

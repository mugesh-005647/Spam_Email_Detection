import streamlit as st
import pickle

st.set_page_config(page_title="Spam Email Detector",page_icon="📧",layout="wide")

@st.cache_resource
def load():
    model=pickle.load(open("spam_model.pkl","rb"))
    vec=pickle.load(open("vectorizer.pkl","rb"))
    return model,vec
model,vectorizer=load()

st.sidebar.title("📌 Project Information")
st.sidebar.write("Algorithm: Multinomial Naive Bayes")
st.sidebar.write("Vectorizer: CountVectorizer")
st.sidebar.metric("Model Accuracy","98%+")

tab1,tab2,tab3=st.tabs(["🏠 Home","📊 About Model","👨‍💻 About Developer"])
with tab1:
    st.title("📧 Spam Email Detector")
    st.write("Detect spam emails using Machine Learning.")
    st.info("Example Spam: Congratulations! You won a free iPhone!\n\nExample Ham: Hi Mugesh, class starts at 9 AM.")
    email=st.text_area("Enter your email or SMS",height=200)
    if st.button("🚀 Predict",use_container_width=True):
        if not email.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Analyzing..."):
                x=vectorizer.transform([email])
                pred=model.predict(x)
                proba=model.predict_proba(x)[0]
            if pred[0]==1:
                st.error("🚫 Spam Email")
            else:
                st.success("✅ Not Spam Email")
                st.balloons()
            c1,c2=st.columns(2)
            c1.metric("Spam Probability",f"{proba[1]*100:.2f}%")
            c2.metric("Ham Probability",f"{proba[0]*100:.2f}%")
            st.progress(int(max(proba)*100))
            st.subheader("Entered Message")
            st.code(email)
with tab2:
    st.header("About Model")
    st.write("- Multinomial Naive Bayes\n- CountVectorizer\n- Dataset: SMS Spam Collection\n- Train/Test:80/20")
with tab3:
    st.header("About Developer")
    st.write("**Mugesh**\n\nIT Student\n\nSkills: Python, NumPy, Pandas, SQL, Machine Learning, Streamlit")
st.markdown("---")
st.caption("Developed by Mugesh")

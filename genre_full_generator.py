import sys

import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# if __name__ == "__main__":
#     # Remove the CWD from sys.path while we load stuff.
#     # This is added back by InteractiveShellApp.init_path()
#     if sys.path[0] == "":
#         del sys.path[0]

#     from ipykernel import kernelapp as app

#     app.launch_new_instance()

if "page" not in st.session_state:
    st.session_state.page = "landing"

    if st.session_state.page == "landing":
        st.title("📚 Can You Judge a Book by its Blurb?")
        st.subheader("A genre prediction app powered by machine learning.🧠")
        st.markdown("""
        🎉 Welcome to the **Genre Generator** app! 🎉   
        Enter a blurb from a book, and this app predicts its genre using a DistilBERT model fine-tuned on book descriptions.
        
        Supported genres include:
        - 🧙 Fantasy
        - 🏰 Historical Fiction
        - 🔪 Horror 
        - 🕵️ Mystery
        - 💘 Romance
        - 👽 Science Fiction
        - 😱 Thriller 
        - 🤠 Western                   
        
        """)

    with st.expander("**👉 Click here to learn more about the model**"):
        st.markdown("""

    ### ⚙️ Training Summary

    - **Model**: `DistilBertForSequenceClassification`
    - **Tokenizer**: `DistilBertTokenizerFast`
    - **Preprocessing**: 
        - Lemmatization using NLTK’s `WordNetLemmatizer`
        - Basic text cleanup( *i.e. lowercasing, punctuation removal*)
    - **Dataset**: 1131 book blurbs labeled by genre
    - **Training Epochs**: 5  

    ---

    ### 📈 Test Set Performance

    - **Test Accuracy**: `65.37%`
    - **Macro F1 Score**: `0.64`

    | Genre | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | Fantasy | 0.57 | 0.86 | 0.69 | 14 |
    | Historical Fiction | 0.52 | 0.57 | 0.54 | 30 |
    | Horror | 0.57 | 0.63 | 0.60 | 41 |
    | Mystery | 0.53 | 0.44 | 0.48 | 36 |
    | Romance | 0.59 | 0.64 | 0.61 | 36 |
    | Science Fiction | 0.92 | 0.94 | 0.93 | 35 |
    | Thriller | 0.65 | 0.44 | 0.53 | 45 |
    | Western | 0.81 | 0.83 | 0.82 | 46 |

    ---

    ### 🧾 Confusion Matrix

    | True \\ Pred | F | HF | M | R | SF |
    |--------------|---|----|---|---|----|
    | Fantasy               | 12 | 0  | 1 | 1 | 0 | 0 | 0 | 0 |
    | Historical Fiction    | 2  | 17 | 2 | 1 | 6 | 1 | 0 | 1 |
    | Horror                | 1  | 3  | 26| 3 | 1 | 0 | 3 | 4 |
    | Mystery               | 0  | 5  | 3 | 16 | 4 | 1 | 6 | 1 |
    | Romance               | 3  | 4  | 3 | 2 | 23 | 0 | 1 | 0 |
    | Science Fiction       | 1  | 0  | 0 | 0 | 1 | 33| 0 | 0 |
    | Thriller              | 0  | 3  | 9 | 7 | 2 | 1 | 20| 3 |
    | Western               | 2  | 1  | 2 | 0 | 2 | 0 | 1 | 38 |

    """)


    st.markdown("""Click the button below to get started!""")
        
    if st.button("🚀 Get Started"):
        st.session_state.page = "predict"
        st.experimental_rerun()
    st.stop()




st.title("📚 Can you Judge a Book by its Blurb?")
st.caption("This app predicts the genre of a book based on its blurb using a DistilBERT model")


tokenizer = DistilBertTokenizerFast.from_pretrained('notebooks/model_added_trained')
model = DistilBertForSequenceClassification.from_pretrained('notebooks/model_added_trained')
model.eval()

id2label = {
    0: "Fantasy",
    1: "Historical Fiction",
    2: "Horror",
    3: "Mystery",
    4: "Romance",
    5: "Science Fiction",
    6: "Thriller",
    7: "Western"
}
labels = list(id2label.values())

sample_blurbs = {
    "📖 The Lord of the Rings": """In ancient times the Rings of Power were crafted by the Elven-smiths, and Sauron, The Dark Lord, forged the One Ring, filling it with his own power so that he could rule all others. But the One Ring was taken from him, and though he sought it throughout Middle-earth, it remained lost to him. After many ages it fell, by chance, into the hands of the hobbit, Bilbo Baggins.""",
    "💘 Pride and Prejudice": """This is Jane Austen's best-loved and most intimately known novel. From its famous opening sentence, the story of the Bennet family and of the novel's two protagonists, Elizabeth and Darcy, told with a wit that its author feared might prove 'rather too light and bright, and sparkling', delights its most familiar readers as thoroughly as it does those who encounter it for the first time. And while she entertains us, she teaches us the wisdom of balance, the folly of 'pride' and 'prejudice'.""",
    "🔍 The Silent Patient": """Alicia Berenson’s life is seemingly perfect. A famous painter married to an in-demand fashion photographer, she lives in a grand house with big windows overlooking a park in one of London’s most desirable areas. One evening her husband Gabriel returns home late from a fashion shoot, and Alicia shoots him five times in the face, and then never speaks another word.
Alicia’s refusal to talk, or give any kind of explanation, turns a domestic tragedy into something far grander, a mystery that captures the public imagination and casts Alicia into notoriety. The price of her art skyrockets, and she, the silent patient, is hidden away from the tabloids and spotlight at the Grove, a secure forensic unit in North London."""
}

st.sidebar.markdown("### 🔍 Try a Sample Blurb")
sample_choice = st.sidebar.selectbox("Choose a sample:", [""] + list(sample_blurbs.keys()))
if sample_choice:
    st.session_state["user_input"] = sample_blurbs[sample_choice]


user_input = st.text_area("✍️ Enter book blurb here:", value=st.session_state.get("user_input", ""), height=200)

user_guess = st.selectbox("❓ What genre do you think it is? ❓", [""] + labels)

if st.button("🔮 Predict Genre"):
    if not user_input.strip():
        st.warning("Please enter or select a book blurb.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_label = id2label[predicted_class]
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        st.success(f"🎉 The predicted genre is: **{predicted_label}** ({probabilities[predicted_class]*100:.2f}% confidence)")

        if user_guess:
            if user_guess == predicted_label:
                st.balloons()
                st.success("👏 You guessed it right!")
            else:
                st.info(f"Not quite! You guessed **{user_guess}**, but the model predicted **{predicted_label}**.")

    st.subheader("📊 Genre Probabilities")
    prob_df = pd.DataFrame({
        "Genre": labels,
        "Confidence": [p.item() * 100 for p in probabilities]
        })
    st.bar_chart(prob_df.set_index("Genre"))

    

    if "user_input" in st.session_state:
        del st.session_state["user_input"]
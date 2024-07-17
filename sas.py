import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Customer reviews
reviews = [
    {
        "customer_name": "Isabella Garcia",
        "customer_email": "igarcia@email.com",
        "text": """
        These premium wireless earbuds exceeded my expectations! The sound quality is exceptional, with rich bass and crystal-clear highs. The noise cancellation feature blocks out distractions, allowing me to focus on my music or calls. They're incredibly comfortable and secure, perfect for my daily commute and workouts. The touch controls are intuitive, and the charging case is sleek and compact. I also appreciate the fast charging and long battery life. Overall, a fantastic investment!
        """
    },
    {
        "customer_name": "Ethan Wilson",
        "customer_email": "ewilson@email.com",
        "text": """
        I'm impressed by the sound quality and comfort of these sports earbuds. They stay put during my runs and bike rides, and the IPX5 sweat resistance is a lifesaver. The ear hooks provide extra stability, and the touch controls are easy to use. The charging case is a bit bulkier than I'd like, but it's still pocket-friendly. The quick-charging feature is a lifesaver when I forget to charge them overnight. I'd recommend their armband phone case for a secure way to carry my phone during workouts.
        """
    },
    {
        "customer_name": "Sophia Anderson",
        "customer_email": "sophia.a@email.com",
        "text": """
        These true wireless earbuds are a steal for the price! The sound is well-balanced, and the passive noise isolation effectively blocks out background noise. They're incredibly lightweight and comfortable for extended wear. The touch controls are responsive, and the case is small enough to fit in my purse. I also appreciate the multipoint connection feature, allowing me to switch between devices seamlessly. The battery life could be slightly better, but it's still decent for the price. I'd recommend their compact wireless charging pad for easy charging on the go.
        """
    },
    {
        "customer_name": "Alexander Clark",
        "customer_email": "aclark@email.com",
        "text": """
        I wanted to love these designer earbuds for their unique look, but the sound quality fell short. The bass is overpowering, and the highs sound tinny. The touch controls are finicky and often don't register my input. The case is bulky and feels cheap. On the bright side, the design and color options are eye-catching. However, I expected better performance for the price. I'd steer clear of these and opt for a more reliable brand.
        """
    },
    {
        "customer_name": "Olivia Thompson",
        "customer_email": "olivia.t@email.com",
        "text": """
        I bought these affordable wireless earbuds on a whim, but they've been underwhelming. The sound is muddy, and there's a noticeable delay during video calls. The connection keeps dropping, and the touch controls are unreliable. The case is lightweight, but the build quality feels flimsy. On the plus side, the battery life is decent, and they're comfortable to wear. I'd recommend spending a bit more on a reliable brand for better performance and connectivity.
        """
    }
]

# System prompt for sentiment analysis and email generation
sys_template = """
You are a Customer Sentiment Analysis System.
Analyze the emotion of the customer who bought some gadgets like earbuds, phones, etc. based on the user's review: {email}.
If the customer is sad, write an email saying sorry.
If the customer is happy, write an email saying thank you for the review and see you on the next purchase.
Use the customer_name and customer_email mentioned in the review email for writing back to the user.
The company name is Model Gad, and the customer name in the email should be the same as in the review email.
The email should start with "Dear {customer_name},"
"""

# Function to generate email based on review
def generate_email(review):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=sys_template.format(email=review["text"], customer_name=review["customer_name"])),
            HumanMessage(content=review["text"])
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Invoke the chain with error handling and retries
    for attempt in range(5):  # Retry up to 5 times
        try:
            email_response = chain.invoke({"email": review["text"]})
            return email_response
        except Exception as e:
            if "429" in str(e):
                print(f"Quota exceeded. Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to generate email: {e}")
                break

# Generate and print emails for each review
for review in reviews:
    email_response = generate_email(review)
    if email_response:
        st.write(f"Email to {review['customer_name']} ({review['customer_email']}):\n{email_response}\n")

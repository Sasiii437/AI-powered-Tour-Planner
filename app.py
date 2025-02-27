import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Retrieve the API key from the .env file
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found in environment variables.")

# Initialize Google GenAI Model using the retrieved API key
llm = GoogleGenerativeAI(
    model="models/gemini-2.0-flash-thinking-exp-1219",  # Ensure this model is available for your API
    google_api_key=api_key
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["source", "destination"],
    template="Provide travel options from {source} to {destination} including possible modes like cab, train, bus, and flight with estimated costs and durations."
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

def generate_travel_options(source, destination):
    """Generate travel options using Google GenAI."""
    response = chain.run(source=source, destination=destination)
    return response

# Streamlit UI
st.title("AI-Powered Travel Planner")

source = st.text_input("Enter the source location:")
destination = st.text_input("Enter the destination location:")

if st.button("Plan My Trip"):
    if source and destination:
        travel_options = generate_travel_options(source, destination)
        st.write(travel_options)
    else:
        st.error("Please provide both source and destination.")

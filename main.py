import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
# from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            # Load and clean text data from the provided URL
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            
            # Extract job postings
            jobs = llm.extract_jobs(data)
            
            # Check if any jobs were found
            if not jobs:
                st.warning("No job postings found.")
                return
            
            for job in jobs:
                # Generate the cover letter for the job
                cover_letter = llm.write_mail(job)  # Ensure this method returns a string
                
                # Check if the cover letter was generated
                if cover_letter:
                    # Display the generated cover letter
                    st.code(cover_letter, language='markdown')  # Provide the cover letter as the body of the code block
                else:
                    st.warning("No cover letter generated for this job.")
                
        except Exception as e:
            st.error(f"An Error Occurred: {e}")



if __name__ == "__main__":
    chain = Chain()
    # portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cover Letter Generator", page_icon="📧")
    create_streamlit_app(chain, clean_text)



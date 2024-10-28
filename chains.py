import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        
        return res if isinstance(res, list) else [res]

    def write_mail(self, job):
        cover_letter_prompt = PromptTemplate.from_template(
        """
        ### Job Role: {role}
        ### Job Description: {description}
        ### Skills Required: {skills}
        ### My Experience and Skills: {my_skills}
        ### INSTRUCTION: Write a personalized cover letter for the job above. Include my experience and skills in a way that matches the role and description.

        ### COVER LETTER:
        """
        )
        my_skills = "Worked a system engineer in Continetal Engineering services, managed requiremnts using IBM Doors, did sysmtem architecture design usign IBM Rhapsody. Use automotive standards like iso26262,ASPICE. I have exprience in Hil testing, did restbus simulation using Vector tools. I have strong german and english language skills, with structured and goal oriented working"

        role = job.get("role")
        description = job.get("description")
        skills = job.get("skills")
        # Chain it with the LLM
        chain_cover_letter = cover_letter_prompt | self.llm

        # Generate the cover letter using your data
        response = chain_cover_letter.invoke(input={
            'role': role,
            'description': description,
            'skills': skills,
            'my_skills': my_skills
        })
        return response.content

if __name__ == "__main__":
    # Assuming API key and other initializations are set up correctly
    chain = Chain()
    
    # Simulate cleaned text from a career page
    cleaned_text = "Example text from a career page goes here."
    
    # Extract job details from the text
    job_list = chain.extract_jobs(cleaned_text)
    
    # Loop through each job dictionary in the list
    for job in job_list:
        role = job.get("role", "N/A")
        description = job.get("description", "N/A")
        skills = job.get("skills", [])

        # Generate cover letter
        cover_letter = chain.write_mail(job)
        print("Generated Cover Letter:\n", cover_letter)

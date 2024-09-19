import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import openai.error


# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Set up your API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Model API",
    description="API implementing Retrieval Augmented Generation (RAG) model.",
    version="1.0.0"
)

# Root path route
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Model API. Use /search_summarize to submit queries."}

class QueryRequest(BaseModel):
    query: str

def google_search(query, num_results=3):
    """ Conduct a Google search and return the top results. """
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
    results = []
    for item in res.get('items', []):
        results.append({
            'title': item.get('title'),
            'link': item.get('link'),
            'snippet': item.get('snippet', '')
        })
    return results

def fetch_page_content(url):
    """ Fetch the page content and return the text. """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)
        # Truncate content if too long
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length]
            print(f"Content truncated to {max_length} characters.")
        return text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ''

def summarize_text(text, prompt_instruction="Summarize the following content:"):
    """ Generate a summary using the OpenAI API. """
    try:
        if len(text) == 0:
            print("No content to summarize.")
            return "Content could not be fetched from the source."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use 'gpt-3.5-turbo' or 'gpt-4' if you have access
            messages=[
                {"role": "system", "content": prompt_instruction},
                {"role": "user", "content": text}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        summary = response['choices'][0]['message']['content'].strip()
        print(f"Generated summary: {summary[:300]}")
        return summary
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return 'Failed to generate summary.'
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 'Failed to generate summary.'



def process_query(query):
    """ Process the query and generate summaries for each result. """
    search_results = google_search(query, num_results=3)
    if not search_results:
        raise HTTPException(status_code=404, detail="No search results found.")
    individual_summaries = []
    for result in search_results:
        page_content = fetch_page_content(result['link'])
        summary = summarize_text(page_content) if page_content else "Content could not be fetched from the source."
        individual_summaries.append({
            'title': result['title'],
            'summary': summary,
            'link': result['link']
        })
    return individual_summaries

@app.post("/search_summarize")
async def search_and_summarize(query_request: QueryRequest):
    """ FastAPI endpoint to handle the summarization requests. """
    individual_summaries = process_query(query_request.query)
    return {
        'query': query_request.query,
        'individual_summaries': individual_summaries
    }

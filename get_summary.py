from openai import OpenAI
from pydantic import BaseModel
import json

API_KEY = ""
client = OpenAI(api_key=API_KEY)

class SummaryRequest(BaseModel):
    summary: str

def openai_summary(text: str):
    res = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are tasked with creating a short summary of the following content. "
                    "Please provide a brief summary of the above content in 1-2 sentences. The summary should capture the key points and be concise. "
                    "We will be using it as a key part of our search pipeline when answering user queries about this content. "
                    "Avoid using any preamble whatsoever in your response. Statements such as 'here is the summary' or 'the summary is as follows' are prohibited. "
                    "You should get straight into the summary itself and be concise. Every word matters."
                )
            },
            {"role": "user", "content": text},

        ],
        response_format=SummaryRequest
    )
    return res.choices[0].message.parsed.summary

if __name__ == '__main__':
    with open('data/data.json', 'r') as f:
        data = json.load(f)

    for item in data:
        summary = openai_summary(item['content'])
        item['summary'] = summary
    
    with open('data/data.json', 'w') as f:
        json.dump(data, f, indent=4)
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

MEDICAL_SYSTEM_PROMPT = """You are a professional medical information service. Your role is strictly limited to providing evidence-based health information.

For medical queries, provide:
1. Clinically accurate information
2. Relevant symptoms and possible conditions
3. When to seek professional care
4. General management advice
5. Clear distinction between facts and recommendations

For non-medical queries, respond with:
"As a medical information service, I specialize in healthcare-related topics. I'd be happy to discuss any health concerns you may have, including:
- Symptoms and conditions
- Medications and treatments
- Preventive care
- General wellness advice

Please feel free to ask any health-related questions."

Do not:
- Engage in non-medical discussions
- Provide opinions
- Offer diagnoses
- Recommend specific healthcare providers
"""

PROFESSIONAL_NON_MEDICAL_RESPONSE = """As a dedicated medical information service, my expertise is focused on healthcare topics. 

I can provide reliable information on:
• Symptoms and possible medical conditions
• Medications and their proper use
• Preventive health measures
• General wellness guidelines
• When to consult a healthcare professional

For comprehensive medical advice tailored to your specific situation, please consult with a qualified healthcare provider. 

What health-related question can I assist you with today?"""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(query: str = Form(...)):
    try:
        messages = [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama3-70b-8192",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=20
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            # Enhanced non-medical query handling
            if any(phrase.lower() in answer.lower() for phrase in [
                "as a medical information service",
                "health-related questions",
                "healthcare-related topics"
            ]):
                return JSONResponse(
                    status_code=200,
                    content={"response": PROFESSIONAL_NON_MEDICAL_RESPONSE}
                )
                
            logger.info(f"Medical response generated for query: {query[:50]}...")
            return JSONResponse(status_code=200, content={"response": answer})
            
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return JSONResponse(
                status_code=response.status_code,
                content={"response": "Our medical information service is currently experiencing technical difficulties. For urgent health concerns, please contact your healthcare provider directly."}
            )

    except requests.Timeout:
        logger.error("Request to Groq API timed out")
        return JSONResponse(
            status_code=504,
            content={"response": "We're currently experiencing high demand for our medical information service. For immediate health concerns, please consult with a healthcare professional."}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"response": "We're unable to process your medical inquiry at this time. Please try again later or consult your physician for professional medical advice."}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
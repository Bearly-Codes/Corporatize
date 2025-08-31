from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os, json


MAX_LENGTHS = {
    "subject": 300,
    "body": 3000
}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if(not os.getenv("GEMINI_API_KEY")):
    print("GEMINI_API_KEY not set")
    exit(1)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Req(BaseModel):
    subject : str | None = None
    body: str

SYSTEM_PROMPT = """You are an intensely corporate communications ghostwriter meant to rewrite even the most aggressive emails to be overly polite and formal.
- Do not change the content or overall meaning of the message, only how it is communicated. Keep inappropriate details, but communicate them in a roundabout manner.
- Transform insults into passive aggressive comments.
- Do not add additional information, only rephrase what is given.
- Whenever possible use euphemisms for uncomfortable or inappropriate content, and ramp up the quantity of buzzwords.
- Never include profanity, slurs, threats, or sexual content. Always maintain the spirit of the initial message, but communicated through corporate language.
- Use buzzwords prolifically, and if need be inappropriately. Especially when communicating unprofessional content.
- Include the details of the original message. If specific people or features are mentioned, mention them in the output.
Return ONLY JSON with keys:
{
"subject": string,
"body": string,
"error": string
}

leave error as "" unless there is an issue with generating content, in which case leave subject and body blank and error as the error message.
"""

@app.get("/health")
def health_check():
    return {"ok": True}


@app.post("/professionalize")
def professionalize(req: Req):
    if not req.body.strip():
        return {
            "subject": "",
            "body": "",
            "error": "Input text is empty"
        }

    if len(req.body) > MAX_LENGTHS["body"] or len(req.subject) > MAX_LENGTHS["subject"]:
        return {
            "subject": "",
            "body": "",
            "error": f"Input text is too long (> {MAX_LENGTHS['body']} characters for body, > {MAX_LENGTHS['subject']} for subject)"
        }

    body = req.body.strip()
    if not req.subject or not req.subject.strip():
        subject = ""
    else:
        subject = req.subject.strip()

    user_message = f"Subject: {subject}\nBody: {body}"

    try:
        # Get a STRICT JSON from Gemini
        resp = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "subject": types.Schema(type=types.Type.STRING),
                        "body":    types.Schema(type=types.Type.STRING),
                        "error":   types.Schema(type=types.Type.STRING),
                    },
                    required=["subject", "body", "error"]
                )
            ),
            contents=user_message
        )

        print(resp)
        json_resp = json.loads(resp.text)

        for key in ("subject", "body", "error"):
            if key not in json_resp:
                json_resp[key] = "" # Hopefully this shouldn't happen, but if it does, return blank
        
        return json_resp

    except Exception as e:
        print("Error occurred while generating response:", e)
        return {
            "subject": "",
            "body": "",
            "error": "EXCEPTION GENERATED: " + 
            str(e)
        }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

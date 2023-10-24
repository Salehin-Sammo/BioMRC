from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertForQuestionAnswering, BertTokenizer
from pydantic import BaseModel
import torch

app = FastAPI()

class QARequest(BaseModel):
    context: str
    question: str

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:8000",  # Adjust the port if necessary
    "http://127.0.0.1",
    "http://127.0.0.1:8000",  # Adjust the port if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "Downloads/my_model"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


from pydantic import BaseModel

class QARequest(BaseModel):
    context: str
    question: str


@app.post("/predict")
async def get_answer(request: QARequest):
    context = request.context
    question = request.question
    
    inputs = tokenizer(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1 

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
    return {"answer": answer}

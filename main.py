from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Social Media Thread Summarizer", description="Summarizes social media threads using AI.")

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ThreadInput(BaseModel):
    thread: str

@app.post("/summarize")
def summarize_thread(input: ThreadInput):
    try:
        # Generate summary
        summary = summarizer(input.thread, max_length=150, min_length=50, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

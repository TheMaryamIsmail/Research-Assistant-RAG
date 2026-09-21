import os
import tempfile
import traceback
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from logic import PureRAGManager

app = FastAPI(title="Research Paper Assistant (No Embeddings Required)")

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

rag_manager = PureRAGManager()


@app.get("/")
async def serve_index(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={}
    )


@app.post("/upload")
@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    temp_files = []
    file_info_list = []

    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file.filename}. Only PDF files are allowed."
                )

            # Windows-compatible temporary file saving
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()
            temp_file.close()

            temp_files.append(temp_file.name)
            file_info_list.append({
                "path": temp_file.name,
                "filename": file.filename
            })

        # Process PDFs into internal searchable memory
        num_chunks = rag_manager.process_pdfs(file_info_list)

        return JSONResponse({
            "status": "success",
            "message": f"Successfully processed {len(files)} document(s) into {num_chunks} searchable chunks.",
            "filenames": [f.filename for f in files]
        })

    except HTTPException:
        raise
    except Exception as e:
        print("\n--- UPLOAD ERROR TRACEBACK ---")
        traceback.print_exc()
        print("-------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary PDF files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


class QuestionRequest(BaseModel):
    question: str


@app.post("/query")
@app.post("/api/query")
async def query_paper(payload: QuestionRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_manager.query(question)
        return JSONResponse(result)
    except Exception as e:
        print("\n--- QUERY ERROR TRACEBACK ---")
        traceback.print_exc()
        print("-------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
@app.post("/api/reset")
async def reset_session():
    rag_manager.clear()
    return JSONResponse({"status": "success", "message": "Session reset successfully."})
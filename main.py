from fastapi import FastAPI, Request, BackgroundTasks, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import shutil
from pathlib import Path

app = FastAPI()

# Setup directories
STATIC_DIR = Path("static")
VIDEO_DIR = STATIC_DIR / "videos"
AVATAR_DIR = STATIC_DIR / "avatars"
TEMP_DIR = Path("temp")

STATIC_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Lazy initialization of pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        from video_bot_wav2lip import VideoBotPipeline
        pipeline = VideoBotPipeline()
    return pipeline

# In-memory task storage
tasks = {}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-avatar")
async def upload_avatar(avatar: UploadFile = File(...)):
    if not avatar.content_type.startswith("image/"):
        return JSONResponse({"success": False, "error": "File must be an image"}, status_code=400)
    
    avatar_id = str(uuid.uuid4())
    extension = avatar.filename.split('.')[-1] if '.' in avatar.filename else "png"
    avatar_path = AVATAR_DIR / f"{avatar_id}.{extension}"
    
    try:
        with open(avatar_path, "wb") as f:
            shutil.copyfileobj(avatar.file, f)
        return {"success": True, "avatar_url": f"/static/avatars/{avatar_id}.{extension}"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/generate")
async def generate_video(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    text = data.get("text")
    avatar_url = data.get("avatar", "/static/avatar.png")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "llm_responding",
        "response_text": None,
        "video_url": None
    }

    video_path = VIDEO_DIR / f"{task_id}.mp4"
    background_tasks.add_task(
        _generate_video_task,
        text,
        str(video_path),
        task_id,
        avatar_url
    )
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return {"status": "not_found"}
    return task

def _generate_video_task(text: str, video_path: str, task_id: str, avatar_url: str):
    global pipeline
    if pipeline is None:
        from video_bot_wav2lip import VideoBotPipeline
        pipeline = VideoBotPipeline()

    # Resolve avatar path
    if avatar_url.startswith("/static/"):
        avatar_local_path = Path(".") / avatar_url.lstrip("/")
        if not avatar_local_path.exists():
            avatar_local_path = STATIC_DIR / "avatar.png"
    else:
        # For external URLs, fallback to default (or implement download if needed)
        avatar_local_path = STATIC_DIR / "avatar.png"

    audio_path = str(TEMP_DIR / f"{task_id}.wav")

    try:
        tasks[task_id]["status"] = "llm_responding"
        response_text = pipeline.llm.process(text)

        tasks[task_id]["status"] = "generating_audio"
        generated_audio = pipeline.tts.process(response_text, output_path=audio_path)
        if not generated_audio:
            raise Exception("TTS failed")

        tasks[task_id]["status"] = "generating_video"
        result = pipeline.animator.process(
            str(avatar_local_path),
            audio_path,
            video_path
        )
        if not result:
            raise Exception("Video generation failed")

        tasks[task_id]["response_text"] = response_text
        tasks[task_id]["video_url"] = f"/static/videos/{task_id}.mp4"
        tasks[task_id]["status"] = "completed"

    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        tasks[task_id]["status"] = "failed"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
#app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import io
import numpy as np
import torch
import wave
from TTS.api import TTS
import uvicorn
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# 🚀 GPU 활성화 (가능한 경우)
use_gpu = torch.cuda.is_available()
tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=use_gpu)
sample_rate = 22050

logging.info(f"TTS 모델 로드 성공 (GPU 사용: {use_gpu})")

# 요청 데이터 구조 정의
class TTSRequest(BaseModel):
    text: str
    speaker: str = None  # 선택적 화자

@app.post("/api/tts")
async def synthesize(request: TTSRequest):
    try:
        # TTS 변환 실행
        wav = tts_model.tts(request.text, speaker=request.speaker) if request.speaker else tts_model.tts(request.text)
        
        # float [-1,1] -> int16 변환
        wav = np.array(wav)
        wav_int16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)

        # WAV 파일 메모리 저장
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(wav_int16.tobytes())

        audio_bytes = buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return JSONResponse(content={"audio": audio_base64})

    except Exception as e:
        logging.error(f"TTS 합성 오류: {e}")
        raise HTTPException(status_code=500, detail="TTS 처리 중 오류 발생")

# 🚀 FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=4)  # 멀티프로세싱 적용
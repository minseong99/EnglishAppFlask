# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64, io, os, logging, traceback
import numpy as np
import torch, wave
from TTS.api import TTS
import uvicorn

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# 🚀 GPU 활성화 (가능한 경우)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 모델 로딩 & 최적화
logging.info("Loading TTS model...")
tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=DEVICE.type=="cuda")
tts_model.model = tts_model.model.half()                       # FP16
tts_model.model = torch.jit.script(tts_model.model)            # JIT compile
tts_model.model.to(DEVICE)
logging.info(f"TTS model ready on {DEVICE}")

SAMPLE_RATE = 22050


# 요청 데이터 구조
class TTSRequest(BaseModel):
    text: str
    speaker: str = None


# 간단 문장 분할 (마침표 기준)
def split_sentences(text: str):
    # 마침표/물음표/느낌표 뒤에 공백으로 분할
    import re
    parts = re.split(r'([.?!])\s*', text)
    # ["Hello", ".", "How are you", "?"] → ["Hello.", "How are you?"]
    sentences = []
    for i in range(0, len(parts)-1, 2):
        sentences.append((parts[i] + parts[i+1]).strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences or [text]


@app.post("/api/tts")
async def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "No text provided")

    try:
        # 2) 문장 단위 청킹
        chunks = split_sentences(req.text)

        # 메모리 버퍼에 결과 이어 쓰기
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)

            for chunk in chunks:
                # 3) no_grad로 추론
                with torch.no_grad():
                    wav = tts_model.tts(chunk, speaker=req.speaker) \
                          if req.speaker else tts_model.tts(chunk)

                # 합성 결과 합치기
                arr = np.array(wav)
                int16 = (np.clip(arr, -1, 1) * 32767).astype(np.int16)
                wf.writeframes(int16.tobytes())

        # 4) 최종 바이트 → base64
        audio_bytes = buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return JSONResponse({"audio": audio_base64})

    except Exception:
        logging.error("TTS processing error:\n" + traceback.format_exc())
        raise HTTPException(500, "TTS 처리 중 오류 발생")


if __name__ == "__main__":
    # 로컬 디버그용: workers=1 로 메모리 절약
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        workers=1,
        log_level="info",
    )

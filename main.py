
import os
import json
import base64
import asyncio
import threading
import queue

import sounddevice as sd
import websockets
from dotenv import load_dotenv

from prompts import JAPANESE_PROMPT


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

MODEL = "gpt-realtime-mini"
SAMPLE_RATE = 24000
CHUNK_SIZE = 480

URL = (
    f"wss://api.openai.com/v1/realtime"
    f"?model={MODEL}"
)

# Audio received from OpenAI
speaker_queue = queue.Queue(maxsize=500)

# Stops old audio when the user interrupts
playback_generation = 0
generation_lock = threading.Lock()


def clear_speaker():
    global playback_generation

    with generation_lock:
        playback_generation += 1

    with speaker_queue.mutex:
        speaker_queue.queue.clear()


async def microphone_task(ws):
    loop = asyncio.get_running_loop()
    mic_queue = asyncio.Queue(maxsize=100)

    def callback(indata, frames, time_info, status):
        data = bytes(indata)

        def add_audio():
            if not mic_queue.full():
                mic_queue.put_nowait(data)

        loop.call_soon_threadsafe(add_audio)

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=callback,
    ):
        print("Microphone ready!")

        while True:
            audio = await mic_queue.get()

            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(
                    audio
                ).decode("ascii")
            }))



def speaker_task(stop_event):
    pending = bytearray()

    # Buffer 200ms of audio before playing
    buffer_target = int(SAMPLE_RATE * 0.2 * 2)
    playing = False

    def callback(outdata, frames, time_info, status):
        nonlocal pending, playing

        needed = frames * 2  # int16 mono = 2 bytes/sample

        # Collect incoming audio chunks
        while True:
            try:
                pending.extend(
                    speaker_queue.get_nowait()
                )
            except queue.Empty:
                break

        # Wait until enough audio is buffered
        if not playing:
            if len(pending) >= buffer_target:
                playing = True
            else:
                outdata[:] = bytes(needed)
                return

        available = min(len(pending), needed)

        outdata[:] = bytes(
            pending[:available]
        ).ljust(needed, b"\x00")

        del pending[:available]

        # Rebuffer if audio runs out
        if len(pending) < needed:
            playing = False

    with sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=callback,
    ):
        print("Speaker ready!")
        stop_event.wait()



async def receive_task(ws):
    async for message in ws:
        event = json.loads(message)
        kind = event.get("type")

        if kind in (
            "response.output_audio.delta",
            "response.audio.delta",
        ):
            audio = base64.b64decode(
                event["delta"]
            )

            if not speaker_queue.full():
                speaker_queue.put_nowait(audio)

        elif kind == "input_audio_buffer.speech_started":
            clear_speaker()
            print("\nYou are speaking...")

        elif kind == "input_audio_buffer.speech_stopped":
            print("Thinking...")

        elif kind in (
            "response.output_audio_transcript.delta",
            "response.audio_transcript.delta",
        ):
            print(event.get("delta", ""), end="", flush=True)

        elif kind == "response.done":
            print("\n")

        elif kind == "error":
            print("\nAPI Error:", event)


async def main():
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    stop_event = threading.Event()

    speaker_thread = threading.Thread(
        target=speaker_task,
        args=(stop_event,),
        daemon=True,
    )

    speaker_thread.start()

    try:
        async with websockets.connect(
            URL,
            additional_headers=headers,
            max_size=None,
        ) as ws:

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": JAPANESE_PROMPT,
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": SAMPLE_RATE
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "interrupt_response": True,
                                "create_response": True
                            }
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": SAMPLE_RATE
                            },
                            "voice": "verse"
                        }
                    }
                }
            }))

            print("--------------------------")
            print("Japanese Voice AI")
            print("Connected to OpenAI")
            print("Speak Japanese to begin")
            print("Press Ctrl+C to exit")
            print("--------------------------")

            await asyncio.gather(
                microphone_task(ws),
                receive_task(ws),
            )

    finally:
        stop_event.set()
        speaker_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")

import os
import gc
import time
import asyncio
import sys
from enum import Enum
from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import torch

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import snapshot_download
from transformers import pipeline as hf_pipeline

load_dotenv()

# Audio utils
import librosa
import soundfile as sf  # librosa depends on this; used for writing clips.

# mlx-whisper for ASR (replace mlx_audio STT)
if os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
import whisper

# MLX-Audio (Apple Silicon) / Fallback for Windows/Linux
try:
    from mlx_audio.tts.utils import load_model as load_tts
    from mlx_audio.tts.generate import generate_audio
    HAS_MLX_AUDIO = True
except (ImportError, ModuleNotFoundError, Exception) as e:
    HAS_MLX_AUDIO = False
    print(f"Notice: mlx-audio is not available on this platform ({e}). TTS will run in simulation/fallback mode.")


class Language(str, Enum):
    ENGLISH = "english"
    HINDI = "hindi"

    @classmethod
    def validate(cls, lang: str):
        if lang.lower() not in [l.value for l in cls]:
            raise ValueError(
                f"Unsupported language: '{lang}'. "
                f"Supported languages are: {[l.value for l in cls]}"
            )


class VoiceTranslationPipeline:
    def __init__(self, use_gpu: bool = False):
        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        print("Initializing Native MLX-Audio Pipeline")
        print(f"Models directory: {self.models_dir}")

        # 1. STT models (English vs Hindi)
        ASR_MODEL_EN = os.getenv("ASR_MODEL_EN", "openai/whisper-tiny")
        ASR_MODEL_HI = os.getenv("ASR_MODEL_HI", "collabora/whisper-tiny-hindi")

        self.asr_path_en = self._ensure_local_model(ASR_MODEL_EN, "whisper-tiny")
        self.asr_path_hi = self._ensure_local_model(ASR_MODEL_HI, "whisper-tiny-hi")

        print(f"Loading English Whisper from: {self.asr_path_en}")
        self.whisper_en = whisper.load_model(
            name="tiny",
            download_root=self.asr_path_en,
            device=self.device,
        )

        print(f"Loading Hindi Whisper from: {self.asr_path_hi}")
        pipe_device = 0 if torch.cuda.is_available() else -1
        self.asr_pipe_hi = hf_pipeline(
            "automatic-speech-recognition",
            model=self.asr_path_hi,
            chunk_length_s=30,
            device=pipe_device,
        )

        # 2. TTS model (Qwen3-TTS 3-second voice clone capable).
        if HAS_MLX_AUDIO:
            self.tts_repo = os.getenv("TTS_MODEL", "chatterbox-4bit")
            self.tts_path = self._ensure_local_model(self.tts_repo, "chatterbox-4bit")
            self.tts_model = load_tts(self.tts_path)
        else:
            self.tts_model = None

        # 3. Local translation LLM
        self.api_base = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
        self.llm_model = os.getenv("LLM_MODEL", "qwen3.5-0.8b")
        print(f"Initializing Local Translation LLM at: {self.api_base} with model: {self.llm_model}")

        self.llm = ChatOpenAI(
            base_url=self.api_base,
            api_key="not-needed",
            model=self.llm_model,
            temperature=0,
        )

        self.translate_prompt = ChatPromptTemplate.from_template(
            "You are a professional translator. Translate the following text from "
            "{src_lang} to {tgt_lang}. Only return the translated text without any "
            "explanations.\n\n"
            "Text: {text}\n"
            "Translation:"
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _ensure_local_model(self, repo_id: str, sub_dir: str) -> str:
        local_path = os.path.join(self.models_dir, sub_dir)
        if not os.path.exists(local_path) or not os.listdir(local_path):
            print(f"Downloading model {repo_id} to {local_path}...")
            snapshot_download(repo_id=repo_id, local_dir=local_path)
        else:
            print(f"Using local model: {local_path}")
        return local_path

    # -------------------------------------------------------------------------
    # Voice clone preparation
    # -------------------------------------------------------------------------
    def prepare_voice_clone(self, src_audio_path: str, work_dir: str) -> str | None:
        """
        If the source audio is > 3 seconds, create a 3-second clip for voice cloning.
        Returns path to the clipped reference audio, or None to use default voice.
        """
        # Fast duration check straight from file.
        try:
            duration = librosa.get_duration(path=src_audio_path)
        except TypeError:
            duration = librosa.get_duration(filename=src_audio_path)

        print(f"Source audio duration: {duration:.2f}s")

        if duration <= 3.0:
            print("Duration <= 3s; using default TTS voice (no clone).")
            return None

        print("Duration > 3s; creating 7-second reference clip for voice cloning.")
        # Load only the first 7 seconds at native sample rate.
        y, sr = librosa.load(src_audio_path, sr=None, offset=0.0, duration=7.0)

        os.makedirs(work_dir, exist_ok=True)
        ref_path = os.path.join(work_dir, "voice_ref.wav")
        sf.write(ref_path, y, sr)
        print(f"Wrote 5s reference clip to: {ref_path}")
        return ref_path

    # -------------------------------------------------------------------------
    # Translation LLM
    # -------------------------------------------------------------------------
    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # Validate languages
        Language.validate(src_lang)
        Language.validate(tgt_lang)

        if src_lang == tgt_lang or not text.strip():
            return text

        print(f"Translating: {src_lang} -> {tgt_lang}")
        try:
            chain = self.translate_prompt | self.llm
            response = chain.invoke({"src_lang": src_lang, "tgt_lang": tgt_lang, "text": text})
            print(f"Local LLM translation result: {response}")
            return response.content.strip()
        except Exception as e:
            print(f"Local LLM server unreachable at {self.api_base} ({e}). Using online fallback translator...")
            try:
                from deep_translator import GoogleTranslator
                lang_code_map = {"english": "en", "hindi": "hi"}
                s_code = lang_code_map.get(src_lang.lower(), "auto")
                t_code = lang_code_map.get(tgt_lang.lower(), "hi")
                translated = GoogleTranslator(source=s_code, target=t_code).translate(text)
                if translated:
                    print(f"Fallback translation result: {translated}")
                    return translated
            except Exception as fallback_err:
                print(f"Fallback translator error: {fallback_err}")

            return f"[LLM Server Offline at {self.api_base} — Start LM Studio or Ollama]"

    # -------------------------------------------------------------------------
    # TTS with optional voice cloning (offloaded to thread)
    # -------------------------------------------------------------------------
    async def text_to_speech(
        self,
        text: str,
        tgt_lang: str,
        output_path: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ):
        """
        Generate speech audio from text.
        If ref_audio is provided and long enough, Qwen3-TTS will clone that voice.
        If ref_text is also provided, MLX-Audio will NOT try to transcribe ref_audio itself.
        `output_path` is treated as an output directory.
        """
        # Validate target language
        Language.validate(tgt_lang)

        print(
            f"Generating TTS for language: {tgt_lang} "
            f"(voice clone: {'ON' if ref_audio else 'OFF'})"
        )
        os.makedirs(output_path, exist_ok=True)

        kwargs = {
            "model": self.tts_model,
            "text": text,
            "output_path": output_path,
            "audio_format": "wav",
            "lang_code": "hi" if tgt_lang == Language.HINDI else "en",
            # "file_prefix": f"{tgt_lang}_",
            # You can add stream=True, streaming_interval, etc. if you want streaming.
        }
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text

        # Run heavy TTS in a worker thread so we don't block the event loop.
        def _run_tts():
            if HAS_MLX_AUDIO:
                generate_audio(**kwargs)
            else:
                print("Notice: MLX audio not available on this OS. Skipping audio generation.")

        await asyncio.to_thread(_run_tts)

    # -------------------------------------------------------------------------
    # STT
    # -------------------------------------------------------------------------
    def transcribe_audio(self, audio_path: str, src_lang: str) -> str:
        """
        Transcribe audio using OpenAI Whisper.
        Selects model based on src_lang.
        """
        # Select model and language code
        if src_lang == Language.ENGLISH:
            print(f"Transcribing with Whisper (EN): {audio_path}")
            # Use MPS acceleration if available, CPU fallback
            result = self.whisper_en.transcribe(
                audio_path,
                language="English",
                fp16=False,
                verbose=False,
                # word_timestamps=False,
            )
            text = result["text"].strip()
            return text

        elif src_lang == Language.HINDI:
            print(f"Transcribing with Transformers Pipeline (HI): {audio_path}")
            prediction = self.asr_pipe_hi(audio_path, return_timestamps=False)
            text = prediction["text"].strip()
            print(f"Transcribed text: {text}")
            return text

        else:
            raise ValueError(f"No STT model configured for language: {src_lang}")

    # -------------------------------------------------------------------------
    # Memory cleanup
    # -------------------------------------------------------------------------
    def clear_memory(self):
        print("Clearing models from memory...")

        if hasattr(self, "whisper_en"):
            del self.whisper_en
        if hasattr(self, "asr_pipe_hi"):
            del self.asr_pipe_hi
        if hasattr(self, "tts_model"):
            del self.tts_model
        if hasattr(self, "llm"):
            del self.llm

        gc.collect()

        if "mps" in self.device:
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

        print("Memory cleanup complete.")

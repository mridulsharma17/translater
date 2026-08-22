import os
import time
import asyncio
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from pipeline import VoiceTranslationPipeline, Language
from document_translator import DocumentParser, DocumentTranslator, PDFExporter

# --- Page Config ---
st.set_page_config(
    page_title="Let's Talk - Voice Agent & Document Translator",
    page_icon="🎙️",
    layout="wide",
)

# --- Cached Pipeline Loader (Renders UI immediately) ---
@st.cache_resource(show_spinner="Loading Whisper & Qwen translation models...")
def get_pipeline():
    return VoiceTranslationPipeline()


def get_active_pipeline():
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        st.session_state.pipeline = get_pipeline()
    return st.session_state.pipeline


if "processing" not in st.session_state:
    st.session_state.processing = False

# Initialize the pipeline once for the run
pipeline = get_active_pipeline()

# --- App Header ---
st.title("🎙️ Let's Talk: Voice Agent & Document Translator")
st.markdown("Real-time voice agent and document translation powered by **Qwen 3.5** and **Whisper**.")

# --- Tabs ---
tab_voice, tab_doc = st.tabs(["🎙️ Real-Time Voice Agent", "📄 Document Translator & PDF Exporter"])

# =============================================================================
# TAB 1: Real-Time Voice Agent
# =============================================================================
with tab_voice:
    st.markdown("""
    Speak into your microphone to transcribe, translate, and synthesize a voiced response in real-time.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Settings")
        
        src_lang = st.selectbox(
            "Source Language",
            options=[l.value for l in Language],
            index=0,
            key="voice_src_lang",
            help="The language you will be speaking."
        )
        
        tgt_lang = st.selectbox(
            "Target Language",
            options=[l.value for l in Language],
            index=1,
            key="voice_tgt_lang",
            help="The language the agent will translate into."
        )
        
        st.write("---")
        st.subheader("Record Audio")
        
        st.markdown("""
        **How to use:**
        1. Click **Start/Stop** to record your voice.
        2. Click **Process Voice Agent** to run translation & speech output.
        """)
        
        audio_bytes = audio_recorder(
            text="Click to Start/Stop",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            key="voice_recorder"
        )
        
        if audio_bytes:
            st.success("✅ Recording Captured!")
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.info("🎤 Ready to record...")
    
    with col2:
        st.header("Agent Output")
        
        if audio_bytes:
            input_dir = os.path.join(os.getcwd(), "inputs")
            os.makedirs(input_dir, exist_ok=True)
            input_path = os.path.join(input_dir, "recorded_audio.wav")
            
            with open(input_path, "wb") as f:
                f.write(audio_bytes)
            
            if st.button("Process Voice Agent", key="btn_process_voice"):
                st.session_state.processing = True
                
                try:
                    with st.status("Processing...", expanded=True) as status:
                        # 1. Transcribe
                        st.write("Transcribing audio...")
                        original_text = pipeline.transcribe_audio(input_path, src_lang)
                        st.write(f"**Original ({src_lang}):** {original_text}")
                        
                        # 2. Prepare voice clone (optional)
                        outputs_dir = os.path.join(os.getcwd(), "outputs")
                        os.makedirs(outputs_dir, exist_ok=True)
                        ref_audio = pipeline.prepare_voice_clone(input_path, outputs_dir)
                        
                        # 3. Translate
                        st.write("Translating text with local Qwen LLM...")
                        translated_text = pipeline.translate_text(
                            original_text, src_lang, tgt_lang
                        )
                        st.write(f"**Translated ({tgt_lang}):** {translated_text}")
                        
                        # 4. TTS
                        st.write("Generating Speech...")
                        asyncio.run(pipeline.text_to_speech(
                            text=translated_text,
                            tgt_lang=tgt_lang,
                            output_path=outputs_dir,
                            ref_audio=ref_audio
                        ))
                        
                        status.update(label="Processing Complete!", state="complete", expanded=False)
    
                    st.session_state.processing = False
                    
                    st.subheader("Results")
                    st.success(f"**Transcription:** {original_text}")
                    st.success(f"**Translation:** {translated_text}")
                    
                    output_files = [f for f in os.listdir(outputs_dir) if f.endswith(".wav") and f != "voice_ref.wav"]
                    if output_files:
                        output_files.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)), reverse=True)
                        latest_audio = os.path.join(outputs_dir, output_files[0])
                        st.audio(latest_audio)
    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.session_state.processing = False
        else:
            st.info("Waiting for audio input...")

# =============================================================================
# TAB 2: Document Translator & PDF Exporter
# =============================================================================
with tab_doc:
    st.header("📄 Document Translator & PDF Exporter")
    st.markdown("""
    Upload your document (**PDF**, **Word .docx**, or **Text .txt/.md**), translate its contents with **Qwen 3.5**, and download the result as a formatted **PDF**.
    """)
    
    doc_col1, doc_col2 = st.columns([1, 1])
    
    with doc_col1:
        st.subheader("1. Upload & Settings")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=["pdf", "docx", "txt", "md"],
            help="Supported formats: PDF, DOCX, TXT, MD"
        )
        
        doc_src_lang = st.selectbox(
            "Document Source Language",
            options=[l.value for l in Language],
            index=0,
            key="doc_src_lang"
        )
        
        doc_tgt_lang = st.selectbox(
            "Document Target Language",
            options=[l.value for l in Language],
            index=1,
            key="doc_tgt_lang"
        )
        
        if uploaded_file is not None:
            st.success(f"📁 **File Uploaded:** `{uploaded_file.name}` ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")
            
            if st.button("🚀 Translate Document", key="btn_translate_doc"):
                try:
                    file_bytes = uploaded_file.getvalue()
                    
                    with st.spinner("Extracting text from document..."):
                        extracted_text = DocumentParser.extract_text(file_bytes, uploaded_file.name)
                    
                    st.info(f"Extracted **{len(extracted_text.split())} words** from document.")
                    
                    # Translation progress
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    translator = DocumentTranslator(pipeline)
                    
                    def update_progress(pct: float, msg: str):
                        progress_bar.progress(pct)
                        status_text.text(msg)
                    
                    translated_doc_text = translator.translate_document(
                        extracted_text,
                        doc_src_lang,
                        doc_tgt_lang,
                        progress_callback=update_progress
                    )
                    
                    st.session_state["extracted_text"] = extracted_text
                    st.session_state["translated_doc_text"] = translated_doc_text
                    st.session_state["doc_filename"] = uploaded_file.name
                    st.session_state["doc_tgt_lang"] = doc_tgt_lang
                    
                    status_text.success("🎉 Document Translation Complete!")
                    
                except Exception as e:
                    st.error(f"Failed to process document: {str(e)}")

    with doc_col2:
        st.subheader("2. Translated Output & Export")
        
        if "translated_doc_text" in st.session_state:
            translated_text = st.session_state["translated_doc_text"]
            orig_filename = st.session_state.get("doc_filename", "Document")
            target_lang = st.session_state.get("doc_tgt_lang", "translated")
            
            st.markdown("### Preview of Translated Text")
            st.text_area(
                "Translated Content",
                value=translated_text,
                height=300
            )
            
            st.write("---")
            st.subheader("📥 Export Options")
            
            # Generate PDF
            with st.spinner("Preparing PDF export..."):
                doc_title = f"Translated: {os.path.splitext(orig_filename)[0]}"
                pdf_bytes = PDFExporter.generate_pdf(
                    translated_text=translated_text,
                    title=doc_title,
                    src_lang=st.session_state.get("doc_src_lang", "english"),
                    tgt_lang=target_lang
                )
            
            out_pdf_filename = f"Translated_{os.path.splitext(orig_filename)[0]}_{target_lang}.pdf"
            
            st.download_button(
                label="📄 Download Translated PDF",
                data=pdf_bytes,
                file_name=out_pdf_filename,
                mime="application/pdf",
                key="btn_download_pdf"
            )
        else:
            st.info("Upload a document on the left and click **Translate Document** to generate your PDF.")

# --- Footer ---
st.write("---")
if st.button("Clear Memory & Refresh", key="btn_clear_memory"):
    if pipeline:
        pipeline.clear_memory()
    if "pipeline" in st.session_state:
        st.session_state.pipeline = None
    for k in ["extracted_text", "translated_doc_text", "doc_filename", "doc_tgt_lang"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

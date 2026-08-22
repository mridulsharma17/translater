import os
import io
from typing import List, Callable, Optional

# Document extraction libraries
import pypdf
import docx

# PDF Generation library
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _register_unicode_font() -> str:
    """
    Detects and registers a real Devanagari-capable TrueType font (Nirmala UI, Mangal, Noto Sans Devanagari).
    """
    font_candidates = [
        ("NirmalaUI", "C:/Windows/Fonts/Nirmala.ttc", 0),
        ("NirmalaUI_B", "C:/Windows/Fonts/NirmalaB.ttf", None),
        ("Mangal", "C:/Windows/Fonts/mangal.ttf", None),
        ("Mangal_B", "C:/Windows/Fonts/MANGAL.TTF", None),
        ("Kokila", "C:/Windows/Fonts/kokila.ttf", None),
        ("Utsaah", "C:/Windows/Fonts/utsaah.ttf", None),
        ("Aparajita", "C:/Windows/Fonts/aparaj.ttf", None),
    ]
    for font_name, font_path, subfont_idx in font_candidates:
        if os.path.exists(font_path):
            try:
                if subfont_idx is not None:
                    pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=subfont_idx))
                else:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                print(f"Registered Devanagari Font: {font_name} ({font_path})")
                return font_name
            except Exception as e:
                print(f"Failed to register {font_path}: {e}")
                continue

    # Fallback: check or download local NotoSansDevanagari font
    font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "fonts")
    os.makedirs(font_dir, exist_ok=True)
    local_noto = os.path.join(font_dir, "NotoSansDevanagari-Regular.ttf")
    if not os.path.exists(local_noto):
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
            urllib.request.urlretrieve(url, local_noto)
        except Exception:
            pass

    if os.path.exists(local_noto):
        try:
            pdfmetrics.registerFont(TTFont("NotoSansDevanagari", local_noto))
            return "NotoSansDevanagari"
        except Exception:
            pass

    return "Helvetica"


UNICODE_FONT = _register_unicode_font()


class DocumentParser:
    """
    Parses text content from PDF, DOCX, and TXT files.
    """
    @staticmethod
    def extract_text(file_bytes: bytes, file_name: str) -> str:
        ext = os.path.splitext(file_name)[1].lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_bytes)
        elif ext in [".docx", ".doc"]:
            return DocumentParser._parse_docx(file_bytes)
        elif ext in [".txt", ".md", ".csv", ".json"]:
            return file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> str:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        extracted_pages = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text.strip())
        return "\n\n".join(extracted_pages)

    @staticmethod
    def _parse_docx(file_bytes: bytes) -> str:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)


class DocumentTranslator:
    """
    Handles chunking text, calling the translation pipeline, and reconstructing translated text.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Splits text into logical paragraphs/chunks suitable for LLM translation.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            if current_len + len(p_clean) > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p_clean]
                current_len = len(p_clean)
            else:
                current_chunk.append(p_clean)
                current_len += len(p_clean)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]

    def translate_document(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Translates a full document text chunk by chunk with optional progress callback.
        """
        chunks = self.chunk_text(text, max_chunk_size=600)
        translated_chunks = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback((i + 1) / total, f"Translating section {i + 1} of {total}...")

            # Call pipeline translation
            translated = self.pipeline.translate_text(chunk, src_lang, tgt_lang)
            translated_chunks.append(translated)

        return "\n\n".join(translated_chunks)


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas for adding page numbers ("Page X of Y") and headers/footers to ReportLab PDFs.
    Uses registered UNICODE_FONT for full multi-language rendering.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont(UNICODE_FONT, 9)
        self.setFillColor(colors.HexColor("#7f8c8d"))

        # Header
        self.drawString(54, 750, "अनुवादित दस्तावेज — Let's Talk Voice Agent")
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)

        # Footer
        self.line(54, 50, 558, 50)
        page_text = f"पृष्ठ {self._pageNumber} / {page_count}"
        self.drawRightString(558, 38, page_text)
        self.drawString(54, 38, "Let's Talk अनुवाद पाइपलाइन द्वारा निर्मित")
        self.restoreState()


class PDFExporter:
    """
    Exports translated text into a clean, formatted PDF document with Devanagari Unicode support.
    """
    @staticmethod
    def _format_paragraph_text(text: str) -> str:
        """
        Escapes XML special characters for Platypus Paragraph while preserving formatting tags and symbols.
        """
        # Escape & < > safely
        safe_text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        # Restore allowed basic inline formatting tags if present
        safe_text = (
            safe_text.replace("&lt;b&gt;", "<b>")
            .replace("&lt;/b&gt;", "</b>")
            .replace("&lt;i&gt;", "<i>")
            .replace("&lt;/i&gt;", "</i>")
            .replace("&lt;br/&gt;", "<br/>")
            .replace("&lt;br&gt;", "<br/>")
        )
        return safe_text

    @staticmethod
    def generate_pdf(
        translated_text: str,
        title: str = "Translated Document",
        src_lang: str = "English",
        tgt_lang: str = "Hindi"
    ) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName=UNICODE_FONT,
            fontSize=18,
            leading=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6
        )

        meta_style = ParagraphStyle(
            'DocMeta',
            parent=styles['Normal'],
            fontName=UNICODE_FONT,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=15
        )

        body_style = ParagraphStyle(
            'DocBody',
            parent=styles['Normal'],
            fontName=UNICODE_FONT,
            fontSize=11,
            leading=17,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10
        )

        bullet_style = ParagraphStyle(
            'DocBullet',
            parent=body_style,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=6
        )

        story = []

        # Title & Meta
        formatted_title = PDFExporter._format_paragraph_text(title)
        story.append(Paragraph(formatted_title, title_style))

        direction_label = f"अनुवाद दिशा: <b>{src_lang.capitalize()} → {tgt_lang.capitalize()}</b>"
        story.append(Paragraph(direction_label, meta_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3498db"), spaceAfter=15))

        # Body Paragraphs & Bullet Points
        raw_paragraphs = translated_text.split("\n\n")
        for p in raw_paragraphs:
            clean_p = p.strip()
            if not clean_p:
                continue

            lines = clean_p.split("\n")
            for line in lines:
                l_strip = line.strip()
                if not l_strip:
                    continue

                formatted_line = PDFExporter._format_paragraph_text(l_strip)

                # Use bullet style if line starts with a bullet symbol, hyphen, asterisk, or arrow
                if l_strip.startswith(("•", "-", "*", "→", "▪", "▸")):
                    story.append(Paragraph(formatted_line, bullet_style))
                else:
                    story.append(Paragraph(formatted_line, body_style))

            story.append(Spacer(1, 4))

        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()

# import array
from pypdf import PdfReader, PdfWriter
from pathlib import Path


pdf_path = (
    Path(__file__).resolve().parent / 
    "A_QUEDA_DO_CEU.pdf"
)

pdf_reader = PdfReader(pdf_path)
pdf_writer = PdfWriter()

print("pdf_path: ", pdf_path)
print("pages: ", len(pdf_reader.pages))

target_pages = range(194, 221)
print("target_pages: ", list(target_pages))
# remove pages
for num_page, page in enumerate(pdf_reader.pages):
    if  num_page in target_pages:
        pdf_writer.add_page(page)

pdf_writer.write("output.pdf")

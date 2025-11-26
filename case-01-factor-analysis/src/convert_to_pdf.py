import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert MD to HTML
    html_text = markdown.markdown(text)

    # Add some basic styling
    html_content = f"""
    <html>
    <head>
    <style>
        @page {{
            size: letter;
            margin: 2cm;
        }}
        body {{ 
            font-family: Helvetica, sans-serif; 
            font-size: 12px; 
            line-height: 1.5;
        }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            margin: 10px 0;
        }}
        h1 {{ 
            color: #2c3e50; 
            font-size: 24px;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 5px;
        }}
        h2 {{ 
            color: #34495e; 
            font-size: 18px;
            border-bottom: 1px solid #eee; 
            padding-bottom: 5px; 
            margin-top: 20px;
        }}
        h3 {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 15px;
        }}
        code {{ 
            background-color: #f8f8f8; 
            padding: 2px 4px; 
            border-radius: 3px; 
            font-family: Courier, monospace; 
            font-size: 10px;
        }}
        pre {{ 
            background-color: #f8f8f8; 
            padding: 10px; 
            border-radius: 5px; 
            border: 1px solid #ddd;
            white-space: pre-wrap;
        }}
        ul {{
            margin-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
    </style>
    </head>
    <body>
    {html_text}
    </body>
    </html>
    """

    # Convert HTML to PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file,
            link_callback=link_callback
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created {output_pdf}")

def link_callback(uri, rel):
    """
    Convert HTML URIs to absolute system paths so xhtml2pdf can access those resources
    """
    # Handle local files
    if not uri.startswith('http'):
        # If the path is relative, make it absolute based on the current working directory
        if not os.path.isabs(uri):
            uri = os.path.join(os.getcwd(), uri)
    return uri

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("reports", exist_ok=True)
    convert_md_to_pdf("reports/TECHNICAL_REPORT.md", "reports/technical_report.pdf")

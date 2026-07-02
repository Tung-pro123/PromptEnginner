import os
import sys
import subprocess

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Đảm bảo có thư viện pypdf hoặc PyPDF2
try:
    install_and_import("pypdf")
    from pypdf import PdfReader
except Exception:
    install_and_import("PyPDF2")
    from PyPDF2 import PdfReader

def main():
    pdf_path = r"d:\Jetson\Jetson\JetRacer Paper\RoughTerrain_Ackerman_HybridDRL___Dhruv_Ameya_AIM2024V5 (1).pdf"
    output_path = r"d:\Jetson\Jetson\docs\aim_paper_text.md"
    
    if not os.path.exists(pdf_path):
        print(f"Không tìm thấy file PDF tại: {pdf_path}")
        return
        
    print(f"Đang đọc file PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text_content = []
    
    text_content.append("# Trích xuất nội dung bài báo: RoughTerrain Ackerman Hybrid DRL")
    text_content.append(f"**Nguồn:** {pdf_path}\n\n---\n")
    
    for i, page in enumerate(reader.pages):
        print(f"Đang trích xuất trang {i+1}/{len(reader.pages)}...")
        text = page.extract_text()
        text_content.append(f"## Trang {i+1}\n")
        text_content.append(text if text else "[Không có text ở trang này]")
        text_content.append("\n\n---\n")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))
        
    print(f"Hoàn thành! Đã lưu văn bản vào: {output_path}")

if __name__ == "__main__":
    main()

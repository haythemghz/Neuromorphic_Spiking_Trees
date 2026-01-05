
import sys
import importlib.util

def check_install(package):
    if importlib.util.find_spec(package) is None:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pypdf
except ImportError:
    try:
        check_install('pypdf')
        import pypdf
    except:
        print("Failed to install pypdf")
        sys.exit(1)

reader = pypdf.PdfReader("c:/Users/Dell/Desktop/Genetic_Decision_Tree__Copy_/Neuromorphic_Paper/Reference Audit.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("audit_content.txt", "w", encoding="utf-8") as f:
    f.write(text)

import re
import sys

def check_latex_balance(filepath):
    print(f"Analyzing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Simple brace balance (ignoring escaped ones)
    open_braces = 0
    for i, char in enumerate(content):
        if char == '{':
            if i > 0 and content[i-1] == '\\': continue
            open_braces += 1
        elif char == '}':
            if i > 0 and content[i-1] == '\\': continue
            open_braces -= 1
        if open_braces < 0:
            print(f"ERROR: Extra closing brace '}}' at index {i} near context: ...{content[max(0, i-20):i+20]}...")
            open_braces = 0
    
    if open_braces > 0:
        print(f"ERROR: {open_braces} unclosed open braces '{{' found.")
    else:
        print("Braces are balanced (basic count).")

    # 2. Environment balance
    stack = []
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        clean_line = re.sub(r'(?<!\\)%.*', '', line)
        for match in re.finditer(r'\\begin\{([\w\*]+)\}|\\end\{([\w\*]+)\}', clean_line):
            if match.group(1): # begin
                stack.append((match.group(1), line_num))
            elif match.group(2): # end
                env_name = match.group(2)
                if not stack:
                    print(f"ERROR: Extra \\end{{{env_name}}} at line {line_num}")
                else:
                    top, start_line = stack.pop()
                    if top != env_name:
                        print(f"ERROR: Mismatch! \\begin{{{top}}} (line {start_line}) closed by \\end{{{env_name}}} at line {line_num}")
    
    for env, line_num in stack:
        print(f"ERROR: Unclosed \\begin{{{env}}} at line {line_num}")

if __name__ == "__main__":
    check_latex_balance('main.tex')

import os

root_dir = r"c:\Data Science\Data Science Projects\rag-mlops-pipeline"

count = 0
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.py') or file.endswith('.yml') or file.endswith('.yaml') or file.startswith('Dockerfile'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                new_lines = [line for line in lines if not line.strip().startswith('=== FILE:')]
                
                if len(lines) != len(new_lines):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    print(f"Cleaned {filepath}")
                    count += 1
            except Exception as e:
                pass

print(f"Cleaned {count} files in total.")

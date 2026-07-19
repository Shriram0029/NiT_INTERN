import io
with io.open('diagnose_err.txt', 'r', encoding='utf-16le') as f1:
    content = f1.read()
with io.open('diagnose_err_utf8.txt', 'w', encoding='utf-8') as f2:
    f2.write(content)
with io.open('full_output.txt', 'r', encoding='utf-16le') as f1:
    content = f1.read()
with io.open('full_output_utf8.txt', 'w', encoding='utf-8') as f2:
    f2.write(content)

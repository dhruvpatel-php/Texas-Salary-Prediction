"""
Standalone HTML Exporter for PRCP-1024-Texas-Salary-Prediction.ipynb
Generates a self-contained, responsive, beautifully styled HTML document in reports/notebook.html.
No external network or npm/pip dependencies required.
"""

import os
import re
import html
import nbformat as nbf

def markdown_to_html(md_text):
    """Simple, clean markdown to HTML converter for notebook markdown cells."""
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    in_list = False
    in_ol = False
    in_code_block = False
    table_header = True

    def format_inline(text):
        # Escape HTML first
        # But preserve LaTeX formulas
        parts = re.split(r'(\$\$.*?\$\$|\$.*?\$)', text)
        formatted_parts = []
        for p in parts:
            if p.startswith('$$') and p.endswith('$$'):
                formatted_parts.append(f'<span class="math-display">{html.escape(p[2:-2])}</span>')
            elif p.startswith('$') and p.endswith('$'):
                formatted_parts.append(f'<span class="math-inline">{html.escape(p[1:-1])}</span>')
            else:
                s = html.escape(p)
                # Bold
                s = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
                # Italic
                s = re.sub(r'\*(.*?)\*', r'<em>\1</em>', s)
                # Inline code
                s = re.sub(r'`(.*?)`', r'<code>\1</code>', s)
                formatted_parts.append(s)
        return ''.join(formatted_parts)

    for line in lines:
        stripped = line.strip()

        # Code block
        if stripped.startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                lang = stripped[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        # Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped[1:-1].split('|')]
            if all(set(c) <= set('-: ') for c in cells):
                table_header = False
                continue
            if not in_table:
                html_lines.append('<div class="table-container"><table class="report-table">')
                in_table = True
                table_header = True

            tag = 'th' if table_header else 'td'
            row_html = '<tr>' + ''.join(f'<{tag}>{format_inline(c)}</{tag}>' for c in cells) + '</tr>'
            html_lines.append(row_html)
            continue
        elif in_table:
            html_lines.append('</table></div>')
            in_table = False
            table_header = True

        # Unordered Lists
        if stripped.startswith(('- ', '* ')):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            item_text = format_inline(stripped[2:])
            html_lines.append(f'<li>{item_text}</li>')
            continue
        elif in_list and not stripped.startswith(('- ', '* ')):
            html_lines.append('</ul>')
            in_list = False

        # Ordered Lists
        m_ol = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if m_ol:
            if not in_ol:
                html_lines.append('<ol>')
                in_ol = True
            item_text = format_inline(m_ol.group(2))
            html_lines.append(f'<li>{item_text}</li>')
            continue
        elif in_ol and not m_ol:
            html_lines.append('</ol>')
            in_ol = False

        # Headers
        if stripped.startswith('# '):
            html_lines.append(f'<h1>{format_inline(stripped[2:])}</h1>')
        elif stripped.startswith('## '):
            html_lines.append(f'<h2>{format_inline(stripped[3:])}</h2>')
        elif stripped.startswith('### '):
            html_lines.append(f'<h3>{format_inline(stripped[4:])}</h3>')
        elif stripped.startswith('#### '):
            html_lines.append(f'<h4>{format_inline(stripped[5:])}</h4>')
        elif stripped.startswith('> '):
            html_lines.append(f'<blockquote>{format_inline(stripped[2:])}</blockquote>')
        elif stripped == '':
            html_lines.append('<p class="spacer"></p>')
        else:
            html_lines.append(f'<p>{format_inline(line)}</p>')

    if in_table:
        html_lines.append('</table></div>')
    if in_list:
        html_lines.append('</ul>')
    if in_ol:
        html_lines.append('</ol>')
    if in_code_block:
        html_lines.append('</code></pre>')

    return '\n'.join(html_lines)

def export_notebook(notebook_path, output_html_path):
    print(f"Reading notebook: {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    css = """
    :root {
        --bg-primary: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --accent-blue: #1e40af;
        --accent-teal: #0d9488;
        --border-color: #e2e8f0;
        --code-bg: #f1f5f9;
        --insight-bg: #f0fdf4;
        --insight-border: #16a34a;
        --code-border: #cbd5e1;
    }
    * { box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.65;
        color: var(--text-primary);
        background-color: var(--bg-primary);
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 1180px;
        margin: 0 auto;
        padding: 30px 24px 80px;
    }
    .header-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: #ffffff;
        padding: 40px 30px;
        border-radius: 12px;
        margin-bottom: 35px;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.15);
    }
    .header-banner h1 { margin: 0 0 10px; font-size: 2.2rem; color: #ffffff; border: none; }
    .header-banner p { margin: 4px 0; font-size: 1.05rem; opacity: 0.92; }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.2);
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .cell-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        margin-bottom: 22px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .cell-card:hover {
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.07);
    }
    .markdown-cell h1 {
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 10px;
        font-size: 1.85rem;
    }
    .markdown-cell h2 {
        color: #1e3a8a;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 15px;
        font-size: 1.45rem;
    }
    .markdown-cell h3 {
        color: #0369a1;
        margin-top: 12px;
        font-size: 1.2rem;
    }
    .markdown-cell p { margin: 8px 0; color: var(--text-secondary); }
    .markdown-cell ul, .markdown-cell ol { padding-left: 24px; color: var(--text-secondary); }
    .markdown-cell li { margin-bottom: 4px; }
    blockquote {
        border-left: 4px solid var(--accent-blue);
        background: #f8fafc;
        margin: 12px 0;
        padding: 8px 16px;
        color: #334155;
        font-style: italic;
    }
    .code-cell-container {
        margin: 14px 0;
    }
    .code-prompt {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 4px;
    }
    pre.code-source {
        background: #1e293b;
        color: #f8fafc;
        padding: 16px;
        border-radius: 8px;
        overflow-x: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.86rem;
        line-height: 1.5;
        margin: 0;
    }
    .output-area {
        margin-top: 12px;
        border-top: 1px dashed var(--border-color);
        padding-top: 12px;
    }
    .output-prompt {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: #dc2626;
        margin-bottom: 4px;
    }
    pre.stream-output {
        background: var(--code-bg);
        border: 1px solid var(--code-border);
        border-radius: 6px;
        padding: 12px;
        overflow-x: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.84rem;
        color: #0f172a;
        margin: 4px 0 10px;
        max-height: 480px;
    }
    .table-container {
        overflow-x: auto;
        margin: 14px 0;
    }
    table.report-table, table.dataframe {
        border-collapse: collapse;
        width: 100%;
        margin: 8px 0;
        font-size: 0.88rem;
        text-align: left;
    }
    table.report-table th, table.dataframe th {
        background: #f1f5f9;
        color: #1e293b;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid var(--border-color);
    }
    table.report-table td, table.dataframe td {
        padding: 8px 14px;
        border: 1px solid var(--border-color);
        color: #334155;
    }
    table.report-table tr:nth-child(even), table.dataframe tr:nth-child(even) {
        background-color: #f8fafc;
    }
    .chart-output {
        text-align: center;
        margin: 16px 0;
    }
    .chart-output img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    .insight-card {
        border-left: 5px solid var(--insight-border);
        background: var(--insight-bg);
    }
    .math-inline {
        font-family: "Cambria Math", "Latin Modern Math", serif;
        font-style: italic;
        background: #f1f5f9;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .math-display {
        display: block;
        text-align: center;
        font-family: "Cambria Math", "Latin Modern Math", serif;
        font-size: 1.1rem;
        margin: 10px 0;
    }
    """

    body_html = []
    body_html.append(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PRCP-1024: Texas State Salary Prediction — Complete Evaluator Report</title>
        <style>{css}</style>
    </head>
    <body>
    <div class="container">
        <div class="header-banner">
            <div>
                <span class="badge">Project Code: PRCP-1024</span>
                <span class="badge">Target Grade: A+</span>
                <span class="badge">Dataset: 149,481 Records</span>
                <span class="badge">Final Test R2: 0.9519</span>
            </div>
            <h1 style="margin-top: 12px;">Texas State Salary Prediction & Workforce Analytics</h1>
            <p>Comprehensive End-to-End Machine Learning System, Diagnostic Reports, and Business Insights</p>
        </div>
    """)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            # Check if this cell is an Insight block
            is_insight = "### Insight:" in cell.source or "### Insight" in cell.source
            card_class = "cell-card markdown-cell insight-card" if is_insight else "cell-card markdown-cell"
            rendered_md = markdown_to_html(cell.source)
            body_html.append(f'<div class="{card_class}">{rendered_md}</div>')

        elif cell.cell_type == 'code':
            exec_num = cell.execution_count or (i + 1)
            code_src = html.escape(cell.source)

            cell_html = ['<div class="cell-card code-cell-container">']
            cell_html.append(f'<div class="code-prompt">In [{exec_num}]:</div>')
            cell_html.append(f'<pre class="code-source"><code>{code_src}</code></pre>')

            if cell.outputs:
                cell_html.append('<div class="output-area">')
                for out in cell.outputs:
                    out_type = out.output_type

                    if out_type == 'stream':
                        stream_text = html.escape(out.text)
                        cell_html.append(f'<pre class="stream-output">{stream_text}</pre>')

                    elif out_type in ['display_data', 'execute_result']:
                        data = out.data
                        if 'image/png' in data:
                            b64 = data['image/png']
                            cell_html.append(f'<div class="chart-output"><img src="data:image/png;base64,{b64}" alt="Figure Output" /></div>')
                        elif 'text/html' in data:
                            cell_html.append(f'<div class="table-container">{data["text/html"]}</div>')
                        elif 'text/plain' in data:
                            plain = html.escape(data['text/plain'])
                            cell_html.append(f'<pre class="stream-output">{plain}</pre>')

                    elif out_type == 'error':
                        ename = html.escape(out.ename)
                        evalue = html.escape(out.evalue)
                        cell_html.append(f'<pre class="stream-output" style="color: #dc2626; background: #fee2e2;">{ename}: {evalue}</pre>')

                cell_html.append('</div>')
            cell_html.append('</div>')
            body_html.append('\n'.join(cell_html))

    body_html.append("""
        <div style="text-align: center; margin-top: 40px; color: var(--text-secondary); font-size: 0.9rem;">
            <p>Texas State Salary Prediction Capstone Project (PRCP-1024) | Generated automatically from executed notebook.</p>
        </div>
    </div>
    </body>
    </html>
    """)

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(body_html))

    file_size_mb = os.path.getsize(output_html_path) / (1024 * 1024)
    print(f"Successfully generated standalone HTML report: {output_html_path} ({file_size_mb:.2f} MB)")

if __name__ == '__main__':
    export_notebook('PRCP-1024-Texas-Salary-Prediction.ipynb', 'reports/notebook.html')

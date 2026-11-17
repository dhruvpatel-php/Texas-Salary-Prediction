"""
In-process notebook runner for PRCP-1024-Texas-Salary-Prediction.ipynb.
Captures all text streams, rich HTML displays, and matplotlib charts,
writing standard nbformat v4 outputs into the notebook.
"""

import sys
import io
import os
import time
import base64
import traceback
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.displaypub import DisplayPublisher

# Set local directory for ipython to avoid permission warning
local_ipython = os.path.abspath('.ipython')
os.makedirs(local_ipython, exist_ok=True)
os.environ['IPYTHONDIR'] = local_ipython

class TeeStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = io.StringIO()

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        self.buffer.write(s)

    def flush(self):
        self.original_stream.flush()

    def getvalue(self):
        return self.buffer.getvalue()

class CustomDisplayPublisher(DisplayPublisher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.published = []

    def publish(self, data, metadata=None, **kwargs):
        self.published.append((data, metadata or {}))

    def clear_output(self, wait=False):
        self.published = []

class NotebookInProcessRunner:
    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        with open(notebook_path, 'r', encoding='utf-8') as f:
            self.nb = nbf.read(f, as_version=4)
            
        self.shell = InteractiveShell.instance()
        self.display_pub = CustomDisplayPublisher()
        self.shell.display_pub = self.display_pub
        
        # Suppress non-interactive show warning
        plt.show = lambda *args, **kwargs: None

    def execute_cell(self, cell, exec_count):
        cell.execution_count = exec_count
        cell.outputs = []
        self.display_pub.published = []
        plt.close('all')
        
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        tee_out = TeeStream(orig_stdout)
        tee_err = TeeStream(orig_stderr)
        
        sys.stdout = tee_out
        sys.stderr = tee_err
        
        res = None
        start_t = time.time()
        try:
            res = self.shell.run_cell(cell.source)
        except Exception as e:
            traceback.print_exc()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            
        elapsed = time.time() - start_t
        
        # Capture stdout
        stdout_text = tee_out.getvalue()
        if stdout_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stdout',
                text=stdout_text
            ))
            
        # Capture stderr (filter minor benign warnings if desired)
        stderr_text = tee_err.getvalue()
        if stderr_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stderr',
                text=stderr_text
            ))
            
        # Capture rich displays (e.g. styled dataframes)
        for data, meta in self.display_pub.published:
            cell.outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data=data,
                metadata=meta
            ))
            
        # Capture return expression result
        if res and res.result is not None:
            val = res.result
            data = {'text/plain': repr(val)}
            if hasattr(val, '_repr_html_'):
                data['text/html'] = val._repr_html_()
            elif hasattr(val, 'to_html'):
                data['text/html'] = val.to_html()
            cell.outputs.append(nbf.v4.new_output(
                output_type='execute_result',
                execution_count=exec_count,
                data=data
            ))
            
        # Capture matplotlib figures
        fig_nums = plt.get_fignums()
        for fignum in fig_nums:
            fig = plt.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode('utf-8')
            cell.outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data={
                    'image/png': b64_str,
                    'text/plain': f'<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>'
                }
            ))
        plt.close('all')
        
        # Check for execution errors
        if res and not res.success:
            if res.error_in_exec:
                err = res.error_in_exec
                tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
                cell.outputs.append(nbf.v4.new_output(
                    output_type='error',
                    ename=type(err).__name__,
                    evalue=str(err),
                    traceback=tb_lines
                ))
            return False, elapsed
            
        return True, elapsed

    def run_all(self, save_path=None):
        if save_path is None:
            save_path = self.notebook_path
            
        code_cells = [c for c in self.nb.cells if c.cell_type == 'code']
        total = len(code_cells)
        print(f"=== Starting In-Process Notebook Execution ({total} Code Cells) ===")
        
        exec_count = 0
        code_idx = 0
        for i, cell in enumerate(self.nb.cells):
            if cell.cell_type == 'code':
                code_idx += 1
                lines = [l.strip() for l in cell.source.splitlines() if l.strip() and not l.strip().startswith('#')]
                desc = lines[0] if lines else "code cell"
                print(f"\n[{code_idx}/{total}] Cell #{i+1}: {desc[:60]}...")
                
                exec_count += 1
                success, elapsed = self.execute_cell(cell, exec_count)
                print(f"[{code_idx}/{total}] Completed in {elapsed:.2f}s | Success: {success} | Outputs: {len(cell.outputs)}")
                
                if not success:
                    print(f"!!! Execution HALTED due to error in cell #{i+1} !!!")
                    break

        print(f"\nWriting executed notebook with outputs to: {save_path}...")
        with open(save_path, 'w', encoding='utf-8') as f:
            nbf.write(self.nb, f)
        print(f"Successfully saved {save_path} ({os.path.getsize(save_path) / 1024:.1f} KB)")

if __name__ == '__main__':
    runner = NotebookInProcessRunner('PRCP-1024-Texas-Salary-Prediction.ipynb')
    runner.run_all()

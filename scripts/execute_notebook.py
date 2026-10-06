"""Execute cells in-process, useful where Jupyter kernel sockets are unavailable."""
import os
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support
p=Path(__file__).resolve().parents[1]/'notebooks/01_customer_churn.ipynb'
os.chdir(p.parent)
nb=nbformat.read(p,as_version=4)
shell=InteractiveShell.instance()
configure_inline_support(shell,'module://matplotlib_inline.backend_inline')
# Explicit image display preserves figure outputs without a socket-based kernel.
import matplotlib.pyplot as plt
from IPython.display import display, Image
from io import BytesIO
def capture_figures(*args, **kwargs):
    for number in plt.get_fignums():
        buffer=BytesIO()
        plt.figure(number).savefig(buffer,format='png',dpi=110,bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close('all')
plt.show=capture_figures
count=0
for cell in nb.cells:
    if cell.cell_type!='code': continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=True)
        shell.events.trigger('post_execute')
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Cell {count} failed: {result.error_before_exec or result.error_in_exec}')
    cell.execution_count=count;cell.outputs=[]
    if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    print('Executed cell',count,flush=True)
nbformat.validate(nb);nbformat.write(nb,p)
print('Completed',count,'code cells')

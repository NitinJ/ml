// This template file converts Jupyter Notebook code cells to plain Python code.
// For each code cell in the notebook, it extracts the source code and applies
// the 'ipython2python' filter to convert IPython syntax to standard Python.
// Only code cells are processed; other cell types are ignored.
{% for cell in nb.cells %}
{% if cell.cell_type == 'code' %}
{{ cell.source | ipython2python }}

{% endif %}
{% endfor %}

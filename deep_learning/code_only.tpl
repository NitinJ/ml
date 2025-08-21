{% for cell in nb.cells %}
{% if cell.cell_type == 'code' %}
{{ cell.source | ipython2python }}

{% endif %}
{% endfor %}

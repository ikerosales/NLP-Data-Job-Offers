import nbformat

# Load the notebook
notebook_path = "first-notebook-mlapp.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

# Check for metadata.widgets and clean them if necessary
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

# Also check and clean each cell if they have metadata.widgets
for cell in nb.cells:
    if "widgets" in cell.get("metadata", {}):
        del cell["metadata"]["widgets"]

# Save the cleaned notebook
cleaned_notebook_path = "first-notebook-mlapp-clean.ipynb"
with open(cleaned_notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
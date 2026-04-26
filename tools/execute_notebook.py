from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/execute_notebook.py <notebook_path>")
        return 2

    path = Path(sys.argv[1])
    print(f"EXECUTING {path}")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)
    print(f"DONE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

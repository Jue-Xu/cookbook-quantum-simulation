# Quantum Simulatoin Cookbook

## About
The cookbook for quantum simulation

## Usage 
### theorem, definition environments
`pip install sphinx-proof`
https://sphinx-proof.readthedocs.io/en/latest/syntax.html

### Building the book

If you'd like to develop and/or build the jupyter book, you should:

1. Clone this repository
2. Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
3. (Optional) Edit the books source files located in the `./` directory
4. Run `jupyter-book clean .` to remove any existing builds
5. Run `jupyter-book build .`

A fully-rendered HTML version of the book will be built in `./_build/html/`.

### deploy on GitHub pages
`ghp-import -n -p -f _build/html`
more details read ...
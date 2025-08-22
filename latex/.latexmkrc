# This file is for use with latexmk
# It tells latexmk how to process the glossaries and citations with biber.
# This ensures that both the bibliography and glossaries are generated correctly.

# Use biber for citations
$bibtex = 'biber %O --output-directory="%R" %B';

# Use makeglossaries for glossaries and acronyms
add_cus_dep('glo', 'gls', 0, 'makeglossaries');
add_cus_dep('acn', 'acr', 0, 'makeglossaries');
add_cus_dep('ist', 'idx', 0, 'makeglossaries');
sub makeglossaries { system("makeglossaries \"$_[0]\"") }
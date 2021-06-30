mkdir -p ~/.streamlit/
cat ~/.streamlit/test.txt

echo "\
[general]\n\
email = \"maria.s.startseva@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
\[theme]\n\
primaryColor = \"#d33682\"\n\
backgroundColor = \"#d1d1e0\"\n\
secondaryBackgroundColor = \"#586e75\"\n\
textColor = \"#fafafa\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml
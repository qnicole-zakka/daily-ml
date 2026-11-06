# MLAI Armory — daily practice v0

A small browser-based practice bank with 35 concept Q&As and 15 Python/PyTorch coding exercises. Topics: basic deep learning, transformers, LLM fine-tuning, and LLM inference.

Repository: https://github.com/qnicole-zakka/daily-ml

Practice website: https://qnicole-zakka.github.io/daily-ml

## Study

Open `index.html` in a browser. Today’s practice selects three concept cards and one coding exercise using your device’s local calendar date. The bank repeats as dates advance. Use **Show answer** after thinking, or **Topic archive** to browse all 50 questions by topic. Coding exercises run in your own Python/PyTorch environment.

The site has no account, server, build step, external fonts, or tracking. It does not record scores or synchronize review history. All study content travels with the page files.

## Publish on GitHub Pages

1. Put `index.html`, `style.css`, `app.js`, `questions.js`, in the root of a GitHub repository. A public repository supports GitHub Pages on GitHub Free.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose your branch (usually `main`) and **/ (root)**, then save.
5. Open the published URL once deployment completes. Bookmark it on your computer and phone.

Relative asset paths support both user sites and project sites. The published site and its question bank are public.

## Edit questions

Edit the objects in `questions.js`. Each question has an ID, type (`qa` or `coding`), topic, title, prompt, answer paragraphs, and a takeaway. Coding questions also have starter code, solution code, and checks. Text is rendered as text, so no HTML markup is needed.

## Local checks

To serve it locally: `python3 -m http.server 8765 --directory .`, then open `http://localhost:8765`. Opening the HTML file directly also works.

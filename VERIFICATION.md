# Current update verification

Verified on November 5, 2026 using JavaScript DOM simulation and Python execution.

- Bank: 100 unique questions, 65 concept Q&As and 35 coding exercises. Added 15 RL-for-LLMs Q&As, 15 LLM fundamentals Q&As, and 20 standard-Python LLM coding exercises.
- Daily selection: 2 basic deep learning Q&As, 2 LLM Q&As, 1 basic deep learning coding exercise, and 1 LLM coding exercise. The LLM pool includes every topic outside basic deep learning. Selection was checked across 140 simulated dates before the Practiced addition.
- Practiced: checked daily questions appear in the new tab; unchecking removes them and updates the daily checkbox. Checked IDs persist across reloads using browser local storage. Verified empty state, tab navigation, malformed storage recovery, and in-session fallback when saving is blocked.
- Syntax: JavaScript syntax checks pass; all Python starter and solution snippets parse. All 20 new solution scripts pass their included assertions. Existing PyTorch solutions were not rerun for this update.
- Browser layout and real-browser interaction have not been verified for this update. Checks above use a minimal simulated DOM.

## Earlier baseline verification (before this update)

# Question-bank verification

Verified on November 4, 2026.

- Bank: 50 unique questions, 35 concept Q&As, and 15 coding exercises. The original 10 question objects are unchanged.
- Topics: basic deep learning (9 Q&As / 4 coding), transformers (10 / 4), LLM fine-tuning (8 / 4), and LLM inference (8 / 3).
- Daily practice: exactly 3 concept cards and 1 coding exercise. Thirty-five simulated local dates covered every question; same-day reloads kept the selection stable.
- Interface: initial answers hidden; keyboard and mouse reveal/hide worked; topic filters returned correct counts; every coding answer revealed its solution.
- Layout: no page overflow at 320, 390, 768, or 1280 pixels. Fresh desktop and phone screenshots inspected. Long code lines scroll inside code blocks.
- Loading: local file access worked with no browser script errors.
- Coding: all 30 starter/solution snippets parsed as Python; all 15 solution scripts and their assertions passed on CPU using PyTorch 2.15.0 in a temporary Python 3.12 environment. The 11 new starter harnesses also passed when filled with the supplied solution functions.

Review history is not stored or synchronized; daily selections use each device’s local calendar date.

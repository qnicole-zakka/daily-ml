(() => {
  'use strict';
  const questions = window.QUESTION_BANK;
  const topics = ['Basic deep learning', 'Transformers', 'LLM fine-tuning', 'LLM inference'];
  const today = new Date();
  const day = Math.floor(Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()) / 86400000);
  const concepts = questions.filter(question => question.type === 'qa');
  const coding = questions.filter(question => question.type === 'coding');
  const daily = [0, 1, 2].map(offset => concepts[(day + offset) % concepts.length]);
  daily.push(coding[day % coding.length]);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function codeBlock(code) {
    const pre = element('pre');
    pre.tabIndex = 0;
    pre.setAttribute('aria-label', 'Python code; scroll horizontally if needed');
    pre.append(element('code', '', code));
    return pre;
  }

  function card(question, index) {
    const article = element('article', 'question-card');
    article.dataset.questionId = question.id;
    const meta = element('div', 'card-meta');
    meta.append(
      element('span', `type-badge ${question.type === 'coding' ? 'coding' : ''}`, question.type === 'coding' ? 'Coding' : 'Concept'),
      element('span', '', question.topic),
      element('span', 'question-number', String(index + 1).padStart(2, '0'))
    );
    article.append(meta, element('h3', '', question.title), element('p', 'prompt', question.prompt));
    if (question.starterCode) {
      article.append(element('p', 'code-label', 'YOUR STARTING POINT · PYTHON / PYTORCH'), codeBlock(question.starterCode));
    }
    const details = element('details');
    const summary = element('summary');
    summary.append(element('span', 'show-label', 'Show answer'), element('span', 'hide-label', 'Hide answer'));
    const answer = element('div', 'answer');
    question.answer.forEach(paragraph => answer.append(element('p', '', paragraph)));
    if (question.solutionCode) {
      answer.append(element('p', 'code-label', 'ONE POSSIBLE SOLUTION'), codeBlock(question.solutionCode));
    }
    if (question.checks && question.checks.length) {
      answer.append(element('p', 'code-label', 'CHECK YOUR UNDERSTANDING'));
      const checks = element('ul');
      question.checks.forEach(check => checks.append(element('li', '', check)));
      answer.append(checks);
    }
    const takeaway = element('p', 'takeaway');
    takeaway.append(element('strong', '', 'Remember'), document.createTextNode(question.takeaway));
    answer.append(takeaway);
    details.append(summary, answer);
    article.append(details);
    return article;
  }

  function render(containerId, bank) {
    document.getElementById(containerId).replaceChildren(...bank.map(card));
  }

  document.getElementById('today-date').textContent = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).format(today).toUpperCase();
  render('today-questions', daily);

  function showView(view) {
    const isToday = view === 'today';
    document.getElementById('today-view').hidden = !isToday;
    document.getElementById('archive-view').hidden = isToday;
    document.getElementById('today-button').setAttribute('aria-pressed', String(isToday));
    document.getElementById('archive-button').setAttribute('aria-pressed', String(!isToday));
  }
  document.getElementById('today-button').addEventListener('click', () => showView('today'));
  document.getElementById('archive-button').addEventListener('click', () => showView('archive'));

  function filterArchive(topic) {
    const selected = topic === 'All topics' ? questions : questions.filter(question => question.topic === topic);
    render('archive-questions', selected);
    document.getElementById('archive-count').textContent = `${selected.length} ${selected.length === 1 ? 'question' : 'questions'}`;
    document.querySelectorAll('#topic-filters button').forEach(button => {
      button.setAttribute('aria-pressed', String(button.textContent === topic));
    });
  }
  const filters = document.getElementById('topic-filters');
  ['All topics', ...topics].forEach(topic => {
    const button = element('button', '', topic);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(topic === 'All topics'));
    button.addEventListener('click', () => filterArchive(topic));
    filters.append(button);
  });
  filterArchive('All topics');
})();

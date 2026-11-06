(() => {
  'use strict';
  const questions = window.QUESTION_BANK;
  const topics = [...new Set(questions.map(question => question.topic))];
  const today = new Date();
  const day = Math.floor(Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()) / 86400000);
  // Everything outside basic deep learning belongs to the LLM rotation,
  // including transformers and reinforcement learning for language models.
  const basic = question => question.topic === 'Basic deep learning';
  const pool = (type, isBasic) => questions.filter(question =>
    question.type === type && basic(question) === isBasic);
  const pick = (bank, count) => Array.from({ length: count }, (_, offset) =>
    bank[(day * count + offset) % bank.length]);
  const daily = [
    ...pick(pool('qa', true), 2),
    ...pick(pool('qa', false), 2),
    ...pick(pool('coding', true), 1),
    ...pick(pool('coding', false), 1)
  ];

  const storageKey = 'daily-ml-practiced-v1';
  let practiced;
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || '[]');
    practiced = new Set(Array.isArray(saved) ? saved.filter(id => questions.some(q => q.id === id)) : []);
  } catch {
    practiced = new Set();
  }
  let storageAvailable = true;

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

  function card(question, index, showCheckbox = false) {
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
    if (showCheckbox) {
      const label = element('label', 'practice-check');
      const checkbox = element('input');
      checkbox.type = 'checkbox';
      checkbox.checked = practiced.has(question.id);
      checkbox.dataset.practicedId = question.id;
      checkbox.setAttribute('aria-label', `Practiced: ${question.title}`);
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) practiced.add(question.id);
        else practiced.delete(question.id);
        try {
          localStorage.setItem(storageKey, JSON.stringify([...practiced]));
          storageAvailable = true;
        } catch {
          storageAvailable = false;
        }
        document.querySelectorAll('input[data-practiced-id]').forEach(input => {
          input.checked = practiced.has(input.dataset.practicedId);
        });
        renderPracticed();
      });
      label.append(checkbox, element('span', '', 'Practiced'));
      article.prepend(label);
    }
    return article;
  }

  function render(containerId, bank) {
    document.getElementById(containerId).replaceChildren(...bank.map((question, index) => card(question, index, true)));
  }

  document.getElementById('today-date').textContent = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).format(today).toUpperCase();
  render('today-questions', daily);

  function showView(view) {
    ['today', 'archive', 'practiced'].forEach(name => {
      document.getElementById(`${name}-view`).hidden = view !== name;
      document.getElementById(`${name}-button`).setAttribute('aria-pressed', String(view === name));
    });
  }
  ['today', 'archive', 'practiced'].forEach(view => {
    document.getElementById(`${view}-button`).addEventListener('click', () => showView(view));
  });

  function renderPracticed() {
    const selected = questions.filter(question => practiced.has(question.id));
    render('practiced-questions', selected);
    document.getElementById('practiced-count').textContent = `${selected.length} ${selected.length === 1 ? 'question' : 'questions'}`;
    document.getElementById('practiced-empty').hidden = selected.length > 0;
    document.getElementById('practiced-storage-note').textContent = storageAvailable
      ? 'Saved in this browser. Uncheck a question to remove it from Practiced.'
      : 'Browser storage is unavailable. These checks will last only while this page stays open.';
  }
  renderPracticed();

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

const API = 'http://localhost:5000';

let jobId = null;
let pollTimer = null;
let lastLogLen = 0;
const wavesurfers = {};

const $ = id => document.getElementById(id);

const runBtn    = $('runBtn');
const termBody  = $('termBody');
const termWrap  = $('termWrap');
const progWrap  = $('progWrap');
const progFill  = $('progFill');
const progStep  = $('progStep');
const progPct   = $('progPct');
const results   = $('results');
const errAlert  = $('errAlert');
const errMsg    = $('errMsg');
const lipImg    = $('lipImg');
const lipSect   = $('lipSection');
const audioGrid = $('audioGrid');
const methodNote = $('methodNote');
const corrCompare = $('corrCompare');

function setupDrop(dzId, inputId, nameId) {
  const dz    = $(dzId);
  const input = $(inputId);
  const name  = $(nameId);

  function pick(f) {
    if (!f) return;
    input._file = f;
    dz.classList.add('filled');
    dz.classList.remove('drag');
    name.textContent = '✓ ' + f.name;
    name.classList.remove('hidden');
    checkReady();
  }

  input.addEventListener('change', () => pick(input.files[0]));
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag');
    if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]);
  });
}

setupDrop('dz1', 'file1', 'f1name');
setupDrop('dz2', 'file2', 'f2name');

function getFile(id) {
  const el = $(id);
  return el._file || (el.files && el.files[0]) || null;
}

function checkReady() {
  runBtn.disabled = !(getFile('file1') && getFile('file2'));
}

const STEP_MAP = { queued: 0, mixing: 1, separating: 2, lip_tracking: 3, matching: 4, done: 5 };

function setStep(s) {
  const idx = STEP_MAP[s] ?? 0;
  for (let i = 0; i <= 5; i++) {
    const el = $('st' + i);
    el.classList.toggle('done', i < idx);
    el.classList.toggle('active', i === idx);
  }
}

function appendLogs(logs) {
  if (!logs) return;
  for (let i = lastLogLen; i < logs.length; i++) {
    const msg = logs[i];
    const line = document.createElement('div');
    const err = msg.includes('ERROR') || msg.includes('CRITICAL');
    line.className = 'tline tactive' + (err ? ' terror' : '');
    line.innerHTML = `<span class="tpfx">▶</span><span class="tmsg">${msg}</span>`;
    termBody.appendChild(line);
  }
  lastLogLen = logs.length;
  termBody.scrollTop = termBody.scrollHeight;
}

const STEP_LABELS = {
  queued:       'Queued...',
  mixing:       'Mixing audio tracks...',
  separating:   'Running Conv-TasNet separation...',
  lip_tracking: 'Tracking lip movement...',
  matching:     'Correlating audio with lips...',
  done:         'Complete',
};

function setProgress(pct, step) {
  progFill.style.width = pct + '%';
  progPct.textContent = pct + '%';
  progStep.textContent = STEP_LABELS[step] || step;
}

function makeAudioCard(opts) {
  const card = document.createElement('div');
  card.className = 'acard ' + (opts.isWinner ? 'winner' : opts.score === undefined ? 'neutral' : 'loser');

  const tag = opts.tag ? `<span class="${opts.tagClass}">${opts.tag}</span>` : '';
  const dur = opts.duration ? `<span class="card-duration">${opts.duration}s</span>` : '';

  const score = opts.score !== undefined
    ? `<div class="score-row">
         <div class="score-badge ${opts.isWinner ? 'high' : 'low'}">r = ${opts.score.toFixed(3)}</div>
         <div class="score-label">${opts.scoreLabel}</div>
       </div>`
    : '';

  const dl = opts.isWinner
    ? `<a class="btn primary" href="${opts.src}" download>⬇ DOWNLOAD</a>` : '';

  card.innerHTML = `
    <div class="card-head">
      <div class="card-label">${opts.label} ${tag}</div>
      ${dur}
    </div>
    <div class="card-sub">${opts.sub}</div>
    ${score}
    <div class="ws-container" id="ws_${opts.id}"></div>
    <div class="card-ctrls">
      <button class="btn" id="play_${opts.id}">▶ PLAY</button>
      ${dl}
    </div>
  `;
  return card;
}

function initWaveSurfer(id, src, type) {
  const container = $('ws_' + id);
  if (!container) return;

  const palette = {
    winner:  { wave: '#00e5a0', progress: '#00ffb5' },
    loser:   { wave: '#1e3554', progress: '#2e4a6a' },
    neutral: { wave: '#38b6ff', progress: '#60caff' },
  };
  const c = palette[type] || palette.neutral;

  const ws = WaveSurfer.create({
    container,
    waveColor: c.wave, progressColor: c.progress,
    barWidth: 2, barGap: 1, barRadius: 2,
    height: 88, url: src,
  });

  const btn = $('play_' + id);
  ws.on('play',   () => { if (btn) btn.textContent = '⏸ PAUSE'; });
  ws.on('pause',  () => { if (btn) btn.textContent = '▶ PLAY'; });
  ws.on('finish', () => { if (btn) btn.textContent = '▶ PLAY'; });
  if (btn) btn.addEventListener('click', () => ws.playPause());

  wavesurfers[id] = ws;
}

function renderCorrBars(c1, c2, winner) {
  corrCompare.classList.remove('hidden');

  const max = Math.max(Math.abs(c1), Math.abs(c2), 0.01);
  const pct1 = Math.max(0, (c1 / max) * 100);
  const pct2 = Math.max(0, (c2 / max) * 100);

  const bar1 = $('corrBar1');
  const bar2 = $('corrBar2');
  bar1.className = 'corr-bar-fill ' + (winner === 'voice1' ? 'winner-bar' : 'loser-bar');
  bar2.className = 'corr-bar-fill ' + (winner === 'voice2' ? 'winner-bar' : 'loser-bar');

  requestAnimationFrame(() => {
    bar1.style.width = pct1 + '%';
    bar2.style.width = pct2 + '%';
  });

  $('corrVal1').textContent = c1.toFixed(3);
  $('corrVal2').textContent = c2.toFixed(3);
}

function renderMethodNote(method) {
  methodNote.classList.remove('hidden');

  if (method === 'pearson') {
    methodNote.innerHTML =
      'Speaker matched using <strong>lip-audio correlation</strong>. ' +
      'The system tracked lip movement and compared it against each separated track\'s energy pattern. ' +
      'The track whose rhythm best matches the lip movement is identified as the target speaker.';
  } else {
    methodNote.innerHTML =
      'Face could not be detected in the target video. ' +
      'Speaker matched using <strong>vocal energy analysis</strong> instead. ' +
      'The track with the stronger vocal presence was selected as the target speaker.';
  }
}

function renderResults(job) {
  results.classList.remove('hidden');

  renderMethodNote(job.method);
  renderCorrBars(job.corr1, job.corr2, job.matched);

  if (job.graph_url) {
    lipSect.classList.remove('hidden');
    lipImg.src = API + job.graph_url + '?t=' + Date.now();
  } else {
    lipSect.classList.add('hidden');
  }

  const v1win = job.matched === 'voice1';
  const dur = job.duration ? job.duration + '' : null;
  audioGrid.innerHTML = '';

  const mixSrc = API + '/api/audio/' + job.mixed;
  audioGrid.appendChild(makeAudioCard({
    id: 'mixed', label: 'Mixed Input',
    sub: 'Combined cocktail signal before separation',
    tag: 'INPUT', tagClass: 'mix-tag', src: mixSrc,
    duration: dur,
  }));
  initWaveSurfer('mixed', mixSrc, 'neutral');

  const winFile  = v1win ? job.voice1 : job.voice2;
  const winScore = v1win ? job.corr1  : job.corr2;
  const winSrc   = API + '/api/audio/' + winFile;
  audioGrid.appendChild(makeAudioCard({
    id: 'winner', label: 'Isolated Target',
    sub: 'Best match with the target speaker\'s lip movement',
    tag: 'TARGET ✓', tagClass: 'win-tag', src: winSrc,
    score: winScore,
    scoreLabel: 'Highest correlation\nIdentified as target speaker',
    isWinner: true, duration: dur,
  }));
  initWaveSurfer('winner', winSrc, 'winner');

  const loseFile  = v1win ? job.voice2 : job.voice1;
  const loseScore = v1win ? job.corr2  : job.corr1;
  const loseSrc   = API + '/api/audio/' + loseFile;
  audioGrid.appendChild(makeAudioCard({
    id: 'loser', label: 'Other Speaker',
    sub: 'Lower correlation with lip movement',
    tag: '', tagClass: '', src: loseSrc,
    score: loseScore,
    scoreLabel: 'Lower correlation\nDiscarded as non-target',
    isWinner: false, duration: dur,
  }));
  initWaveSurfer('loser', loseSrc, 'loser');
}

function startPolling(id) {
  lastLogLen = 0;
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(API + '/api/status/' + id);
      if (!res.ok) return;
      const job = await res.json();

      setStep(job.step || 'queued');
      setProgress(job.progress || 0, job.step || 'queued');
      appendLogs(job.logs);

      if (job.status === 'error') {
        clearInterval(pollTimer);
        errMsg.textContent = 'Pipeline error: ' + job.error;
        errAlert.classList.remove('hidden');
        runBtn.disabled = false;
        runBtn.textContent = 'RETRY';
      }

      if (job.status === 'complete') {
        clearInterval(pollTimer);
        runBtn.disabled = false;
        runBtn.textContent = 'RUN AGAIN';
        renderResults(job);
      }
    } catch {
      // keep polling through transient failures
    }
  }, 800);
}

runBtn.addEventListener('click', async () => {
  const f1 = getFile('file1');
  const f2 = getFile('file2');
  if (!f1 || !f2) return;

  results.classList.add('hidden');
  errAlert.classList.add('hidden');
  methodNote.classList.add('hidden');
  corrCompare.classList.add('hidden');
  audioGrid.innerHTML = '';
  Object.values(wavesurfers).forEach(ws => ws.destroy());
  for (const k in wavesurfers) delete wavesurfers[k];
  termBody.innerHTML = '';
  lastLogLen = 0;
  progFill.style.width = '0%';

  runBtn.disabled = true;
  runBtn.textContent = 'PROCESSING...';
  termWrap.classList.remove('hidden');
  progWrap.classList.remove('hidden');
  setStep('queued');

  try {
    const fd = new FormData();
    fd.append('video1', f1);
    fd.append('video2', f2);

    const res = await fetch(API + '/api/process', { method: 'POST', body: fd });

    if (res.status === 413) {
      throw new Error('File too large. Maximum size is 100MB per video.');
    }
    if (res.status === 415) {
      const data = await res.json();
      throw new Error(data.error || 'Invalid file type. Please upload video files.');
    }
    if (!res.ok) throw new Error('Server returned HTTP ' + res.status);

    const data = await res.json();
    jobId = data.job_id;
    startPolling(jobId);
  } catch (e) {
    errMsg.textContent = e.message || 'Could not reach the backend. Make sure python app.py is running.';
    errAlert.classList.remove('hidden');
    runBtn.disabled = false;
    runBtn.textContent = 'RETRY';
    progWrap.classList.add('hidden');
    termWrap.classList.add('hidden');
  }
});

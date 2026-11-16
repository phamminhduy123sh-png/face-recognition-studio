'use strict';
const $ = (id) => document.getElementById(id);
const names = { dung: 'Dũng', duy: 'Duy', hung: 'Hưng' };
const pageNames = { recognize: 'Nhận diện', dataset: 'Dữ liệu', training: 'Huấn luyện', reports: 'Báo cáo', guide: 'Hướng dẫn' };
const state = { page: 'recognize', image: null, imageUrl: null, faces: [], streams: {}, job: null, toastTimer: null, status: null, reports: {} };
const escapeHTML = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const number = (v) => Number.isFinite(Number(v)) ? Number(v) : null;
function toast(message, error = false) {
  $('toast').textContent = message; $('toast').className = `toast${error ? ' error' : ''}`;
  clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => $('toast').classList.add('hidden'), 6500);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data; try { data = await response.json(); } catch { throw new Error('Không đọc được phản hồi từ ứng dụng.'); }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(x => x.msg).join('; ') : 'Thao tác chưa hoàn tất. Vui lòng thử lại.');
  }
  return data;
}
async function busy(button, operation) {
  const previous = button.disabled; button.disabled = true; button.classList.add('spinning');
  try { return await operation(); } catch (error) { toast(error.message, true); return null; }
  finally { button.disabled = previous || (!!state.job && ['queued', 'running'].includes(state.job.status) && ['startTraining', 'importLegacy', 'evaluateModel', 'calibrateModel'].includes(button.id)); button.classList.remove('spinning'); }
}
function stopCamera(key) { state.streams[key]?.getTracks().forEach(track => track.stop()); delete state.streams[key]; }
function showPage(page) {
  if (!(page in pageNames)) page = 'recognize';
  if (state.page !== page) { stopCamera('recognize'); stopCamera('dataset'); $('dataCameraStage').classList.add('hidden'); }
  state.page = page;
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
  $('breadcrumb').textContent = pageNames[page]; document.title = `${pageNames[page]} · Face Studio`;
  history.replaceState(null, '', `#${page}`);
  if (page === 'dataset') loadDataset(); if (page === 'reports') loadReports();
}
document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click', event => { event.preventDefault(); showPage(button.dataset.page); }));
window.addEventListener('hashchange', () => showPage(location.hash.slice(1)));
async function loadStatus() {
  try {
    const status = await api('/api/status'); state.status = status;
    const ready = status.model_ready;
    for (const id of ['modelStatus', 'trainingModelStatus']) {
      $(id).classList.toggle('ready', ready);
      $(id).innerHTML = `<span></span>${ready ? 'Mô hình sẵn sàng' : 'Chưa có mô hình'}`;
    }
    $('modelSummary').textContent = ready ? 'Đã có trọng số' : 'Chưa có mô hình';
    $('modelSummaryNote').textContent = ready ? status.checkpoint : 'Bắt đầu từ dữ liệu của bạn';
    $('thresholdSummary').textContent = status.threshold_ready ? 'Đã hiệu chỉnh' : 'Chưa hiệu chỉnh';
    applySplits(status.splits); if (status.job) renderJob(status.job);
  } catch (error) { $('modelStatus').innerHTML = '<span></span>Chưa kết nối được ứng dụng'; }
}
function applySplits(splits) {
  if (!splits) return;
  const labels = {train:'Train',val:'Validation',test:'Test'};
  const owner = session => Object.keys(labels).find(key => (splits[key] || []).includes(session));
  document.querySelectorAll('.session-table thead th:not(:first-child)').forEach((th, index) => { const tag=th.querySelector('span'); if(tag)tag.textContent=(labels[owner(index+1)]||'Chưa gán').toUpperCase(); });
  [...$('captureSession').options].forEach(option => { option.textContent=`Phiên ${String(option.value).padStart(2,'0')} · ${labels[owner(Number(option.value))]||'Chưa gán'}`; });
  document.querySelectorAll('.split-strip span').forEach((span,index)=>{const key=['train','val','test'][index];span.innerHTML=`<i></i>${labels[key]} · phiên ${(splits[key]||[]).map(Number).join(', ')}`;});
  const stat=document.querySelector('#page-dataset .stat-card:nth-child(2) strong');
  if(stat)stat.innerHTML=`${(splits.train||[]).length} <i>/</i> ${(splits.val||[]).length} <i>/</i> ${(splits.test||[]).length}`;
}
function issueMessage(issue) {
  if(typeof issue==='string')return issue;
  const messages={missing_root:'Chưa có bộ dữ liệu. Tạo cấu trúc thư mục rồi thêm ảnh vào 18 phiên.',unexpected_person:'Thư mục dữ liệu chỉ dành cho dung, duy và hung.',misplaced_image:'Ảnh đang đặt sai cấp. Chuyển ảnh vào thư mục người/phien_XX tương ứng.',missing_session:'Thiếu thư mục phiên chụp. Chọn “Tạo cấu trúc thư mục” để bổ sung.',empty_session:'Phiên chưa có ảnh hợp lệ. Bổ sung ảnh chụp riêng cho người và phiên này.',unexpected_session:'Tên thư mục phiên chưa hợp lệ. Dùng phien_01 đến phien_06.',duplicate_leakage:'Ảnh trùng giữa nhiều danh tính hoặc nhiều tập dữ liệu. Cần loại trùng trước khi huấn luyện.',duplicate_image:'Có ảnh trùng trong cùng người và tập dữ liệu.',corrupt_image:'Ảnh không đọc được hoặc có định dạng không hợp lệ.',symlink_person:'Thư mục người phải là thư mục thật, không dùng liên kết.',symlink_session:'Thư mục phiên phải là thư mục thật, không dùng liên kết.',symlink_image:'Ảnh phải là file gốc trong bộ dữ liệu.',face_validation:'Ảnh chưa đạt kiểm tra khuôn mặt. Cần đúng một khuôn mặt rõ ràng.'};
  const path=issue.path?String(issue.path).split(/[\\/]/).slice(-3).join('/') : '';
  return (messages[issue.code]||issue.message||issue.code)+(path?' · '+path:'');
}
function setSource(source) {
  const camera = source === 'camera';
  $('uploadTab').classList.toggle('selected', !camera); $('cameraTab').classList.toggle('selected', camera);
  $('uploadTab').setAttribute('aria-selected', !camera); $('cameraTab').setAttribute('aria-selected', camera);
  $('cameraStage').classList.toggle('hidden', !camera);
  $('uploadZone').classList.toggle('hidden', camera || !!state.image);
  $('previewStage').classList.toggle('hidden', camera || !state.image);
  if (!camera) stopCamera('recognize');
  if (camera && !state.streams.recognize) { $('cameraPlaceholder').classList.remove('hidden'); $('takeSnapshot').classList.add('hidden'); }
}
$('uploadTab').onclick = () => setSource('upload'); $('cameraTab').onclick = () => setSource('camera');
function resetResults() { state.faces = []; $('faceCount').textContent = '—'; $('resultList').classList.add('hidden'); $('resultEmpty').classList.remove('hidden'); drawFaces(); }
async function selectImage(file) {
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { toast('Ảnh vượt quá 10 MB. Hãy chọn ảnh nhỏ hơn.', true); return; }
  if (!file.type.startsWith('image/')) { toast('Vui lòng chọn một file ảnh.', true); return; }
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.image = file; state.imageUrl = URL.createObjectURL(file); $('inputPreview').src = state.imageUrl;
  $('runRecognition').disabled = false; resetResults(); setSource('upload');
  try { await $('inputPreview').decode(); drawFaces(); } catch { toast('Không đọc được ảnh đã chọn.', true); state.image = null; $('runRecognition').disabled = true; setSource('upload'); }
}
$('chooseImage').onclick = event => { event.stopPropagation(); $('imageInput').click(); };
$('uploadZone').onclick = () => $('imageInput').click();
$('uploadZone').onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('imageInput').click(); } };
$('replaceImage').onclick = () => $('imageInput').click();
$('imageInput').onchange = event => { selectImage(event.target.files[0]); event.target.value = ''; };
['dragenter', 'dragover'].forEach(name => $('uploadZone').addEventListener(name, event => { event.preventDefault(); $('uploadZone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach(name => $('uploadZone').addEventListener(name, event => { event.preventDefault(); $('uploadZone').classList.remove('dragging'); }));
$('uploadZone').addEventListener('drop', event => selectImage(event.dataTransfer.files[0]));
function drawFaces() {
  const stage = $('previewStage'), image = $('inputPreview'), canvas = $('faceOverlay');
  if (!image.naturalWidth || stage.classList.contains('hidden')) return;
  const w = stage.clientWidth, h = stage.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
  const scale = Math.min(w / image.naturalWidth, h / image.naturalHeight);
  const ox = (w - image.naturalWidth * scale) / 2, oy = (h - image.naturalHeight * scale) / 2;
  state.faces.forEach((face, index) => {
    if (!Array.isArray(face.bbox) || face.bbox.length !== 4) return;
    const [x, y, bw, bh] = face.bbox.map(Number);
    if (![x, y, bw, bh].every(Number.isFinite)) return;
    const color = face.accepted ? '#cde5a9' : face.status === 'uncalibrated' ? '#d6c5ea' : '#f0c891';
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(ox + x * scale, oy + y * scale, bw * scale, bh * scale);
    const person = face.candidate_display_name || face.display_name || 'Khuôn mặt';
    const caption = face.status === 'uncalibrated' ? `Ứng viên: ${person}` : face.status === 'manual_match' ? `Thủ công: ${face.display_name || person}` : face.accepted ? face.display_name : 'Không xác định';
    const label = `${index + 1} · ${caption}`;
    ctx.font = '12px Segoe UI, sans-serif'; const tw = ctx.measureText(label).width + 14;
    const ty = Math.max(0, oy + y * scale - 25), tx = Math.min(w - tw, Math.max(0, ox + x * scale));
    ctx.fillStyle = color; ctx.fillRect(tx, ty, tw, 23); ctx.fillStyle = '#29442c'; ctx.fillText(label, tx + 7, ty + 16);
  });
}
new ResizeObserver(drawFaces).observe($('previewStage'));
function renderResults(result) {
  state.faces = Array.isArray(result.faces) ? result.faces : [];
  $('faceCount').textContent = state.faces.length;
  if (!state.faces.length) {
    $('resultList').classList.add('hidden'); $('resultEmpty').classList.remove('hidden');
    $('resultEmpty').querySelector('h3').textContent = 'Chưa tìm thấy khuôn mặt';
    $('resultEmpty').querySelector('p').textContent = 'Thử ảnh rõ hơn, hướng mặt về camera và có đủ ánh sáng.';
    drawFaces(); return;
  }
  $('resultEmpty').classList.add('hidden'); $('resultList').classList.remove('hidden');
  $('resultList').innerHTML = state.faces.map((face, index) => {
    const uncalibrated = face.status === 'uncalibrated';
    const candidate = face.candidate_display_name || names[face.candidate] || face.display_name || 'Chưa có ứng viên';
    const title = face.accepted ? face.display_name : uncalibrated ? candidate : 'Không xác định';
    const label = uncalibrated ? 'Ứng viên · chưa hiệu chỉnh' : face.status === 'manual_match' ? 'Khớp ngưỡng thủ công' : face.accepted ? 'Đạt ngưỡng nhận diện' : `Gần nhất: ${candidate}`;
    const score = number(face.score);
    return `<article class="face-result ${uncalibrated ? 'uncalibrated' : face.accepted ? '' : 'unknown'}"><div class="face-result-top"><span class="face-avatar">${index + 1}</span><div><h3>${escapeHTML(title)}</h3><span class="face-label">${escapeHTML(label)}</span></div></div><div class="score-row"><span>Độ tương đồng cosine</span><strong>${score === null ? '—' : score.toFixed(4)}</strong></div>${uncalibrated ? '<div class="face-result-note">Chưa xác nhận danh tính khi chưa có ngưỡng hiệu chỉnh.</div>' : ''}</article>`;
  }).join(''); drawFaces();
}
async function recognize() {
  if (!state.image) return;
  await busy($('runRecognition'), async () => {
    const form = new FormData(); form.append('image', state.image, state.image.name || 'camera.jpg');
    $('faceCount').textContent = '…';
    const result = await api('/api/predict', { method: 'POST', body: form }); renderResults(result);
  });
}
$('runRecognition').onclick = recognize;
async function openCamera(key, video) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera cần trình duyệt hỗ trợ và kết nối localhost hoặc HTTPS.');
  stopCamera(key);
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    state.streams[key] = stream; video.srcObject = stream; await video.play();
  } catch (error) {
    throw new Error(error.name === 'NotAllowedError' ? 'Chưa được cấp quyền camera. Hãy cho phép trong cài đặt trình duyệt.' : error.name === 'NotFoundError' ? 'Không tìm thấy camera trên thiết bị này.' : 'Không mở được camera. Hãy kiểm tra thiết bị và quyền truy cập.');
  }
}
$('startCamera').onclick = () => busy($('startCamera'), async () => { await openCamera('recognize', $('cameraVideo')); $('cameraPlaceholder').classList.add('hidden'); $('takeSnapshot').classList.remove('hidden'); });
async function snapshot(video) {
  if (!video.videoWidth) throw new Error('Camera chưa sẵn sàng. Vui lòng thử lại.');
  const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Không chụp được ảnh.')), 'image/jpeg', .94));
}
$('takeSnapshot').onclick = () => busy($('takeSnapshot'), async () => { const blob = await snapshot($('cameraVideo')); await selectImage(blob); await recognize(); });
function sessionCount(counts, person, session) {
  const value = counts?.by_person_session?.[person]?.[`phien_${String(session).padStart(2, '0')}`] ?? counts?.by_person_session?.[person]?.[String(session)] ?? 0;
  return typeof value === 'number' ? value : Number(value?.count || 0);
}
async function loadDataset() {
  try {
    const report = await api('/api/dataset'), counts = report.counts || {};
    $('totalImages').textContent = counts.total ?? 0;
    $('auditStatus').textContent = report.valid ? 'Sẵn sàng' : 'Cần bổ sung';
    $('auditNote').textContent = report.valid ? 'Cấu trúc và ảnh đã qua kiểm tra nhanh' : 'Xem các mục cần hoàn thiện bên dưới';
    $('sessionTable').innerHTML = Object.entries(names).map(([person, name]) => `<tr><td><span class="person-avatar ${person}">${name[0]}</span>${name}</td>${[1,2,3,4,5,6].map(session => { const count = sessionCount(counts, person, session); return `<td><button class="session-cell ${count > 0 ? 'has-images' : ''}" data-person="${person}" data-session="${session}" title="Thêm ảnh ${name}, phiên ${session}">${count} ảnh</button></td>`; }).join('')}</tr>`).join('');
    document.querySelectorAll('.session-cell').forEach(cell => cell.onclick = () => { $('capturePerson').value = cell.dataset.person; $('captureSession').value = cell.dataset.session; $('capturePerson').scrollIntoView({ behavior: 'smooth', block: 'center' }); $('capturePerson').focus(); });
    const issues = Array.isArray(report.issues) ? report.issues : [];
    $('auditIssues').innerHTML = issues.length ? issues.slice(0, 30).map(issue => `<div class="audit-item"><span></span><div>${escapeHTML(issueMessage(issue))}</div></div>`).join('') : '<div class="audit-item success"><span></span><div>Kiểm tra nhanh không phát hiện vấn đề. Khi huấn luyện, từng ảnh sẽ được kiểm tra khuôn mặt.</div></div>';
  } catch (error) { $('auditStatus').textContent = 'Chưa đọc được'; $('auditIssues').innerHTML = `<p class="error-inline">${escapeHTML(error.message)}</p>`; renderEmptySessions(); }
}
function renderEmptySessions() { if (!$('sessionTable').children.length) $('sessionTable').innerHTML = Object.entries(names).map(([person, name]) => `<tr><td><span class="person-avatar ${person}">${name[0]}</span>${name}</td>${[1,2,3,4,5,6].map(() => '<td><span class="session-cell">—</span></td>').join('')}</tr>`).join(''); }
$('refreshDataset').onclick = () => busy($('refreshDataset'), loadDataset);
$('initDataset').onclick = () => busy($('initDataset'), async () => { const data = await api('/api/dataset/init', { method: 'POST' }); toast(data.message); await loadDataset(); });
async function saveCapture(blob, options = {}) {
  const { refresh = true, notify = true } = options;

  const form = new FormData();
  form.append('person', $('capturePerson').value);
  form.append('session', $('captureSession').value);
  form.append('image', blob, blob.name || 'camera.jpg');

  const result = await api('/api/capture', {
    method: 'POST',
    body: form
  });

  if (notify) toast(result.message);
  if (refresh) await loadDataset();

  return result;
}
$('uploadCapture').onclick = () => $('captureInput').click();
$('captureInput').onchange = async event => {
  const files = Array.from(event.target.files || []);
  event.target.value = '';

  if (!files.length) return;

  await busy($('uploadCapture'), async () => {
    let saved = 0;
    const errors = [];

    for (const file of files) {
      try {
        await saveCapture(file, {
          refresh: false,
          notify: false
        });

        saved++;
      } catch (error) {
        errors.push(`${file.name}: ${error.message}`);
      }
    }

    // Cập nhật giao diện sau khi xử lý tất cả ảnh
    await loadDataset();

    if (errors.length > 0) {
      toast(
        `Đã lưu ${saved}/${files.length} ảnh. Có ${errors.length} ảnh lỗi.`,
        true
      );

      alert(
        'Những ảnh chưa lưu được:\n\n' + errors.join('\n')
      );
    } else {
      toast(`Đã lưu thành công ${saved} ảnh!`);
    }
  });
};
$('openDataCamera').onclick = () => busy($('openDataCamera'), async () => { $('dataCameraStage').classList.remove('hidden'); try { await openCamera('dataset', $('dataCameraVideo')); } catch (error) { $('dataCameraStage').classList.add('hidden'); throw error; } });
$('closeDataCamera').onclick = () => { stopCamera('dataset'); $('dataCameraStage').classList.add('hidden'); };
$('saveDataSnapshot').onclick = () => busy($('saveDataSnapshot'), async () => saveCapture(await snapshot($('dataCameraVideo'))));
function renderJob(job) {
  state.job = job; if (!job) return;
  const active = ['queued', 'running'].includes(job.status), done = job.status === 'completed', failed = job.status === 'failed';
  $('startTraining').disabled = active; $('importLegacy').disabled = active; $('evaluateModel').disabled = active; if ($('calibrateModel')) $('calibrateModel').disabled = active;
  $('jobStatus').textContent = active ? 'ĐANG CHẠY' : done ? 'HOÀN TẤT' : failed ? 'CẦN KIỂM TRA' : 'SẴN SÀNG';
  $('jobStatus').className = `status-pill${active ? ' running' : failed ? ' failed' : ''}`;
  const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
  $('progressRing').style.setProperty('--progress', `${progress}%`); $('progressValue').textContent = `${Math.round(progress)}%`;
  const kind = { train: 'Huấn luyện', evaluate: 'Đánh giá tập test', import: 'Nhập trọng số', calibrate: 'Hiệu chỉnh ngưỡng' }[job.kind] || 'Tác vụ';
  $('jobTitle').textContent = failed ? `${kind} chưa hoàn tất` : done ? `${kind} đã hoàn tất` : `${kind} đang diễn ra`;
  $('jobSubtitle').textContent = job.epoch ? `Epoch ${job.epoch}/${job.epochs || '…'}` : active ? 'Bạn có thể tiếp tục xem các mục khác.' : done ? 'Kết quả đã được lưu trên máy.' : 'Xem nhật ký để biết chi tiết.';
  $('jobDescription').textContent = failed ? job.error : job.epoch ? `Loss: ${number(job.loss)?.toFixed(4) ?? '—'} · Validation: ${number(job.val_accuracy) === null ? '—' : (job.val_accuracy * 100).toFixed(1) + '%'}` : done ? 'Mở phần báo cáo để xem kết quả thực tế.' : 'Tiến trình và nhật ký sẽ được cập nhật tự động.';
  $('jobLogs').textContent = (job.logs || []).join('\n') || (active ? 'Đang chuẩn bị dữ liệu và mô hình…' : 'Không có nhật ký bổ sung.'); $('jobLogs').scrollTop = $('jobLogs').scrollHeight;
}
$('trainingForm').onsubmit = event => { event.preventDefault(); busy($('startTraining'), async () => { const job = await api('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: Number($('epochs').value), batch_size: Number($('batchSize').value), backbone: $('backbone').value }) }); renderJob(job); $('startTraining').disabled = true; toast('Đã bắt đầu tác vụ huấn luyện.'); }); };
$('importLegacy').onclick = () => busy($('importLegacy'), async () => { renderJob(await api('/api/legacy/import', { method: 'POST' })); showPage('training'); toast('Đang kiểm tra và nhập trọng số legacy.'); });
$('evaluateModel').onclick = () => busy($('evaluateModel'), async () => { renderJob(await api('/api/evaluate', { method: 'POST' })); showPage('training'); toast('Đã bắt đầu đánh giá phiên test.'); });
function renderHistory(history) {
  const rows = Array.isArray(history) ? history : history?.history || [];
  const points = rows.map((row, index) => ({ x: number(row.epoch) ?? index + 1, y: number(row.train_loss ?? row.loss) })).filter(p => p.y !== null);
  if (!points.length) { $('trainingChart').innerHTML = '<p class="chart-placeholder">Chưa có lịch sử loss theo epoch.</p>'; return; }
  const W=560,H=240,L=40,R=20,T=20,B=35,maxY=Math.max(...points.map(p=>p.y),.01)*1.1,maxX=Math.max(...points.map(p=>p.x),1);
  const x = value => L+(value-1)/Math.max(maxX-1,1)*(W-L-R), y = value => H-B-value/maxY*(H-T-B);
  const line = points.map((p,i)=>`${i?'L':'M'}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ');
  const area = `${line} L${x(points.at(-1).x)},${H-B} L${x(points[0].x)},${H-B} Z`;
  $('trainingChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Biểu đồ train loss theo epoch"><defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#a3bc7d" stop-opacity=".25"/><stop offset="1" stop-color="#a3bc7d" stop-opacity=".02"/></linearGradient></defs>${[0,.25,.5,.75,1].map(f=>`<line x1="${L}" y1="${y(maxY*f)}" x2="${W-R}" y2="${y(maxY*f)}" stroke="#edf1e4"/><text x="${L-9}" y="${y(maxY*f)+3}" text-anchor="end" fill="#a6b295" font-size="9">${(maxY*f).toFixed(2)}</text>`).join('')}<path d="${area}" fill="url(#chartFill)"/><path d="${line}" stroke="#7e9e59" stroke-width="2" fill="none"/>${points.map(p=>`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3" fill="#779853"/>`).join('')}<text x="${L}" y="${H-10}" font-size="9" fill="#9fad8b">Epoch 1</text><text x="${W-R}" y="${H-10}" text-anchor="end" font-size="9" fill="#9fad8b">Epoch ${maxX}</text></svg>`;
}
async function loadReports() {
  try {
    const { reports, available } = await api('/api/reports'); state.reports = reports;
    $('reportEmpty').classList.toggle('hidden', available); $('reportContent').classList.toggle('hidden', !available); if (!available) return;
    const metrics = reports.test || reports.validation || reports.summary?.metrics?.val || {};
    const accuracy = number(metrics.accuracy_known_only ?? metrics.accuracy), f1 = number(metrics.macro_f1);
    const count = number(metrics.total ?? metrics.count ?? metrics.n_samples ?? metrics.support);
    $('reportMetrics').innerHTML = [['ĐỘ CHÍNH XÁC · NGƯỜI ĐÃ BIẾT', accuracy === null ? '—' : (accuracy*100).toFixed(1)+'%', reports.test ? 'Trên tập test' : 'Trên tập validation'], ['MACRO F1', f1 === null ? '—' : f1.toFixed(3), 'Trung bình trên ba danh tính'], ['SỐ ẢNH ĐÁNH GIÁ', count ?? '—', 'Theo báo cáo đã lưu']].map(([title,value,note])=>`<div class="stat-card"><span>${title}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
    renderHistory(reports.history);
    const matrix = metrics.confusion_matrix;
    $('confusionMatrix').innerHTML = Array.isArray(matrix) && matrix.length === 3 ? `<table class="matrix-table"><thead><tr><th>Thật ↓ / Dự đoán →</th>${Object.values(names).map(name=>`<th>${name}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,i)=>`<tr><th>${Object.values(names)[i]}</th>${row.map((value,j)=>`<td class="${i===j?'diag':''}">${Number(value)||0}</td>`).join('')}</tr>`).join('')}</tbody></table>` : '<p class="chart-placeholder">Chưa có ma trận nhầm lẫn trong báo cáo.</p>';
    const threshold = number(reports.calibration?.threshold);
    $('calibrationTitle').textContent = threshold === null ? 'Chưa hiệu chỉnh ngưỡng' : `Ngưỡng cosine: ${threshold.toFixed(4)}`;
    $('calibrationDescription').textContent = threshold === null ? 'Cần bộ ảnh người lạ dùng riêng cho validation để đánh giá khả năng từ chối danh tính ngoài bộ dữ liệu.' : `Ngưỡng được chọn từ dữ liệu validation. FAR người lạ thực nghiệm: ${number(reports.calibration.empirical_unknown_far) === null ? '—' : (reports.calibration.empirical_unknown_far*100).toFixed(1)+'%'}. Điểm cosine không phải xác suất.`;
  } catch (error) { toast(error.message,true); }
}
if ($('calibrateModel')) $('calibrateModel').onclick = () => busy($('calibrateModel'), async () => { renderJob(await api('/api/calibrate', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({max_far:.05})})); showPage('training'); toast('Đã bắt đầu hiệu chỉnh ngưỡng trên validation.'); });
setInterval(async () => {
  if (state.job && ['queued','running'].includes(state.job.status)) {
    try { const {job} = await api('/api/jobs/current'); const wasActive = ['queued','running'].includes(state.job.status); renderJob(job); if(wasActive && ['completed','failed'].includes(job?.status)) { await loadStatus(); await loadReports(); toast(job.status==='completed'?'Tác vụ đã hoàn tất.':job.error||'Tác vụ chưa hoàn tất.',job.status==='failed'); } } catch {}
  }
}, 1500);
window.addEventListener('beforeunload',()=>{stopCamera('recognize');stopCamera('dataset');if(state.imageUrl)URL.revokeObjectURL(state.imageUrl);});
showPage(location.hash.slice(1) || 'recognize'); loadStatus(); loadDataset(); loadReports();

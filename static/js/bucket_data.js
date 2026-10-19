// ── ОБНОВЛЕНИЕ ЧАСОВ ────────────────────────────────────────────────
    function updateClock() {
      const now = new Date();
      const months = ['ЯНВАРЯ','ФЕВРАЛЯ','МАРТА','АПРЕЛЯ','МАЯ','ИЮНЯ','ИЮЛЯ','АВГУСТА','СЕНТЯБРЯ','ОКТЯБРЯ','НОЯБРЯ','ДЕКАБРЯ'];
      const h = String(now.getHours()).padStart(2,'0');
      const m = String(now.getMinutes()).padStart(2,'0');
      const s = String(now.getSeconds()).padStart(2,'0');
      const d = now.getDate();
      const mon = months[now.getMonth()];
      const y = now.getFullYear();
      document.getElementById('clock').textContent = `${h}:${m}:${s} | ${d} ${mon} ${y} г.`;
    }
    updateClock();
    setInterval(updateClock, 1000);

    // ── ОБНОВЛЕНИЕ FRAME ID ─────────────────────────────────────────────
    function updateFrameId() {
      const ts = Date.now() / 1000;
      const now = new Date();
      const dateStr = now.toISOString().slice(0,10).replace(/-/g,'');
      document.getElementById('frame-id').textContent = `Frame ID: ${dateStr}${String(now.getHours()).padStart(2,'0')}-${ts.toFixed(6)}`;
    }
    setInterval(updateFrameId, 1000);
    updateFrameId();

    // ── ЗАГРУЗКА ДАННЫХ КОВША ИЗ ФАЙЛА static/bucket.json ──────────────
    async function loadBucketData() {
      try {
        const response = await fetch('/static/bucket.json?t=' + Date.now());

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const bucket = data.bucket || data;  // поддержка обоих форматов

        document.getElementById('bucket-name').textContent = bucket.name || '—';

        const current = Number(bucket.current_buckets) || 0;
        const max     = Number(bucket.max_allowed_buckets) || 0;
        const volume  = Number(bucket.volume_m3 ?? 0).toFixed(1).replace('.', ',');

        document.getElementById('current-buckets').textContent = current;
        document.getElementById('max-buckets').textContent     = max || '—';
        document.getElementById('bucket-volume').textContent   = volume;

        let percent = 0;
        if (max > 0) percent = Math.min(100, Math.round((current / max) * 100));

        const fill = document.getElementById('progress-fill');
        fill.style.width = percent + '%';
        document.getElementById('progress-percent').textContent = percent;

        fill.style.background = 
          percent >= 95 ? '#e63946' :
          percent >= 80 ? '#f4a261' : 
                          '#2a9d8f';

      } catch (err) {
        console.error('[BUCKET] Ошибка:', err);
        document.getElementById('bucket-name').textContent = 'Ошибка загрузки';
      }
    }

    // ── ЗАПУСК ──────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
      loadBucketData();
      setInterval(loadBucketData, 2000);
      console.log('[APP] Инициализация завершена');
    });

    // ── КНОПКИ ──────────────────────────────────────────────────────────
    document.querySelectorAll('.btn-action:not(:first-child)').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        btn.classList.toggle('secondary');
      });
    });
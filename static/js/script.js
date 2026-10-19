    // Live clock
    function updateClock() {
      const now = new Date();
      const months = ['ЯНВАРЯ','ФЕВРАЛЯ','МАРТА','АПРЕЛЯ','МАЯ','ИЮНЯ',
                      'ИЮЛЯ','АВГУСТА','СЕНТЯБРЯ','ОКТЯБРЯ','НОЯБРЯ','ДЕКАБРЯ'];
      const h = String(now.getHours()).padStart(2,'0');
      const m = String(now.getMinutes()).padStart(2,'0');
      const s = String(now.getSeconds()).padStart(2,'0');
      const d = now.getDate();
      const mon = months[now.getMonth()];
      const y = now.getFullYear();
      document.getElementById('clock').textContent =
        `${h}:${m}:${s} | ${d} ${mon} ${y} г.`;
    }
    updateClock();
    setInterval(updateClock, 1000);

    // Frame ID update (will be populated from backend via SSE or polling if needed)
    function updateFrameId() {
      const ts = Date.now() / 1000;
      const now = new Date();
      const dateStr = now.toISOString().slice(0,10).replace(/-/g,'');
      document.getElementById('frame-id').textContent =
        `Frame ID: ${dateStr}${String(now.getHours()).padStart(2,'0')}-${ts.toFixed(6)}`;
    }
    setInterval(updateFrameId, 1000);
    updateFrameId();

    // Button toggle
    document.querySelectorAll('.btn-action:not(:first-child)').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        btn.classList.toggle('secondary');
      });
    });

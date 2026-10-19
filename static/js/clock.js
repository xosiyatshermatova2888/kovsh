    // Динамические часы
    function updateClock() {
      const now = new Date();
      const timeStr = now.toLocaleTimeString('ru-RU', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      });
      const dateStr = now.toLocaleDateString('ru-RU', { 
        day: '2-digit', 
        month: 'long', 
        year: 'numeric' 
      }).toUpperCase();

      document.getElementById('clock').textContent = `${timeStr} | ${dateStr}`;
    }

    setInterval(updateClock, 1000);
    updateClock();

    // Показать placeholder при ошибке загрузки видео
    function showPlaceholder() {
      document.getElementById('placeholder').style.display = 'flex';
      document.getElementById('video-feed').style.display = 'none';
    }
/* ════════════════════════════════════════════════════════════════════
   confetti.js — micro celebration on completion
   ════════════════════════════════════════════════════════════════════ */
(function () {
  const COLORS = ['#1b3a6b', '#2563eb', '#16a34a', '#d97706', '#7c3aed', '#dc2626'];

  window.fireConfetti = function(opts = {}) {
    const cv = document.getElementById('confettiCanvas');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    cv.width  = innerWidth;
    cv.height = innerHeight;
    const N = opts.count || 120;
    const parts = [];
    for (let i = 0; i < N; i++) {
      parts.push({
        x: innerWidth/2 + (Math.random()-.5)*40,
        y: innerHeight/2 + (Math.random()-.5)*40,
        vx: (Math.random()-.5)*14,
        vy: (Math.random()-1.2)*16,
        size: 4 + Math.random()*6,
        color: COLORS[Math.floor(Math.random()*COLORS.length)],
        rot: Math.random()*Math.PI*2,
        vr: (Math.random()-.5)*.3,
        life: 1,
      });
    }
    let t = 0;
    function frame() {
      ctx.clearRect(0,0,cv.width, cv.height);
      let alive = false;
      for (const p of parts) {
        p.vy += 0.4;                 // gravity
        p.x += p.vx; p.y += p.vy;
        p.rot += p.vr;
        p.life -= 0.012;
        if (p.life > 0) alive = true;
        ctx.save();
        ctx.globalAlpha = Math.max(0, p.life);
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size/2, -p.size/2, p.size, p.size*0.6);
        ctx.restore();
      }
      t++;
      if (alive && t < 200) requestAnimationFrame(frame);
      else ctx.clearRect(0,0,cv.width,cv.height);
    }
    frame();
  };
})();

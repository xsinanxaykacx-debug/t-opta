(() => {
  const escapeHtml = value => String(value == null ? "" : value).replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
  function ensurePanel() {
    if (document.getElementById("similarHistoryPanel")) return;
    const status = document.getElementById("status");
    if (!status) return;
    const panel = document.createElement("section");
    panel.id = "similarHistoryPanel";
    panel.className = "hidden mb-6 rounded-xl border border-slate-700 bg-slate-950/80 p-4";
    panel.innerHTML = '<div class="flex items-center justify-between gap-3 mb-3"><div><h3 id="similarHistoryTitle" class="font-bold text-sm">Benzer geçmiş maçlar</h3><p class="text-[11px] text-slate-400 mt-1">Marjdan arındırılmış 1X2 olasılık yakınlığına göre sıralanır.</p></div><button type="button" id="similarHistoryClose" class="text-xs text-slate-400 underline">Kapat</button></div><div id="similarHistoryStatus" class="text-xs text-slate-300 mb-3"></div><div id="similarHistoryList" class="space-y-2"></div><p id="similarHistoryNote" class="text-[10px] text-slate-500 mt-3"></p>';
    status.insertAdjacentElement("afterend", panel);
    document.getElementById("similarHistoryClose").addEventListener("click", () => panel.classList.add("hidden"));
  }
  async function showSimilar(match) {
    ensurePanel();
    const panel = document.getElementById("similarHistoryPanel");
    const status = document.getElementById("similarHistoryStatus");
    const list = document.getElementById("similarHistoryList");
    const note = document.getElementById("similarHistoryNote");
    panel.classList.remove("hidden");
    status.textContent = "Geçmiş arşiv taranıyor...";
    list.innerHTML = "";
    note.textContent = "";
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    try {
      const response = await fetch("/api/matches/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date: match.date, odds: match.odds, limit: 20 })
      });
      const data = await response.json();
      if (!response.ok || data.status !== "success") throw new Error(data.message || "Benzer maç analizi başarısız.");
      status.textContent = data.count + " benzer geçmiş maç bulundu.";
      note.textContent = data.note || "";
      if (!data.matches.length) {
        list.innerHTML = '<div class="text-xs text-slate-400">Bu tarih ve oran yapısı için uygun geçmiş maç bulunamadı.</div>';
        return;
      }
      list.innerHTML = data.matches.map(item =>
        '<article class="rounded-lg border border-slate-800 bg-slate-900/70 p-3">' +
          '<div class="flex flex-wrap justify-between gap-2 text-xs"><span class="text-slate-300">' + escapeHtml(item.date) + ' · ' + escapeHtml(item.league) + '</span>' +
          '<span class="text-cyan-300">' + Number(item.distance_pp).toFixed(3) + ' yüzde puan uzaklık</span></div>' +
          '<div class="text-sm font-semibold mt-1">' + escapeHtml(item.home) + ' ' + escapeHtml(item.score) + ' ' + escapeHtml(item.away) + '</div>' +
          '<div class="text-xs text-slate-400 mt-2">MS1 ' + Number(item.odds["1"]).toFixed(2) + ' · MSX ' + Number(item.odds.X).toFixed(2) + ' · MS2 ' + Number(item.odds["2"]).toFixed(2) +
          ' <span class="ml-2 text-emerald-300">' + escapeHtml(item.result) + '</span></div></article>'
      ).join("");
    } catch (error) {
      status.textContent = error.message;
    }
  }
  async function showOddsMovement(match) {
    ensurePanel();
    const panel = document.getElementById("similarHistoryPanel");
    const status = document.getElementById("similarHistoryStatus");
    const list = document.getElementById("similarHistoryList");
    const note = document.getElementById("similarHistoryNote");
    panel.classList.remove("hidden");
    document.getElementById("similarHistoryTitle").textContent = "Oran hareketi · " + match.home + " - " + match.away;
    status.textContent = "Kaydedilmiş oran gözlemleri yükleniyor...";
    list.innerHTML = "";
    note.textContent = "";
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    try {
      const response = await fetch("/api/matches/odds-history/" + encodeURIComponent(match.match_key));
      const data = await response.json();
      if (!response.ok || data.status !== "success") throw new Error(data.message || "Oran geçmişi alınamadı.");
      status.textContent = data.count + " farklı oran gözlemi kaydedilmiş.";
      note.textContent = data.note || "";
      if (!data.snapshots.length) {
        list.innerHTML = '<div class="text-xs text-slate-400">Henüz oran değişimi kaydedilmedi. T-OPTA açık kaldıkça farklı oranlar kaydedilecek.</div>';
        return;
      }
      list.innerHTML = data.snapshots.map((item, index) =>
        '<article class="rounded-lg border border-slate-800 bg-slate-900/70 p-3">' +
          '<div class="text-xs text-slate-400">' + escapeHtml(new Date(item.captured_at).toLocaleString("tr-TR")) + '</div>' +
          '<div class="text-sm mt-1">MS1 <b>' + Number(item.odds["1"]).toFixed(2) + '</b> · MSX <b>' + Number(item.odds.X).toFixed(2) + '</b> · MS2 <b>' + Number(item.odds["2"]).toFixed(2) + '</b></div>' +
        '</article>'
      ).join("");
      if (data.movement_from_first) {
        const delta = data.movement_from_first;
        list.insertAdjacentHTML("beforeend", '<div class="text-xs text-cyan-300 pt-2">İlk gözleme göre değişim: MS1 ' + (delta["1"] > 0 ? "+" : "") + delta["1"].toFixed(3) + ' · MSX ' + (delta.X > 0 ? "+" : "") + delta.X.toFixed(3) + ' · MS2 ' + (delta["2"] > 0 ? "+" : "") + delta["2"].toFixed(3) + '</div>');
      }
    } catch (error) {
      status.textContent = error.message;
    }
  }
  window.showTOptaSimilar = showSimilar;
  function attachButtons() {
    const matches = window.__tOptaMatchRows || [];
    if (!matches.length) return;
    document.querySelectorAll(".rank-pick, .compact-match").forEach(card => {
      if (card.dataset.similarBound === "1") return;
      const match = matches.find(item => card.textContent.includes(item.home) && card.textContent.includes(item.away));
      if (!match) return;
      card.dataset.similarBound = "1";
      const similarButton = document.createElement("button");
      similarButton.type = "button";
      similarButton.className = "mt-2 mr-4 text-xs text-cyan-300 underline";
      similarButton.textContent = "Benzer maçlar";
      similarButton.addEventListener("click", () => showSimilar(match));
      card.appendChild(similarButton);
      if (match.match_key) {
        const movementButton = document.createElement("button");
        movementButton.type = "button";
        movementButton.className = "mt-2 text-xs text-amber-300 underline";
        movementButton.textContent = "Oran hareketi";
        movementButton.addEventListener("click", () => showOddsMovement(match));
        card.appendChild(movementButton);
      }
    });
  }
  const observer = new MutationObserver(attachButtons);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", () => { ensurePanel(); attachButtons(); });
  ensurePanel();
  attachButtons();
})();
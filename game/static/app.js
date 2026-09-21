// 이 파일은 상태·서버 요청·화면 전환·로그인·로비를 맡는다. 테이블은 table.js, 전적·순위는 pages.js.
// 입력: 사용자의 클릭과 서버 응답. 출력: 화면 갱신.
"use strict";

const 상태 = { access: null, refresh: null, 사용자: null, 게임: null, 상대이름: null, 구글ID: null };
const $ = (id) => document.getElementById(id);

function 저장() {
  // 왜 try인가: 사생활 보호 모드에서는 localStorage 가 예외를 던진다. 저장 못 해도 플레이는 된다.
  try { localStorage.setItem("bj_tokens", JSON.stringify({ access: 상태.access, refresh: 상태.refresh })); } catch (e) { /* 무시 */ }
}
function 불러오기() {
  try {
    const t = JSON.parse(localStorage.getItem("bj_tokens") || "null");
    if (t) { 상태.access = t.access; 상태.refresh = t.refresh; }
  } catch (e) { /* 무시 */ }
}
function 지우기() {
  상태.access = null; 상태.refresh = null; 상태.사용자 = null; 상태.게임 = null;
  try { localStorage.removeItem("bj_tokens"); } catch (e) { /* 무시 */ }
}

function 알림(글) { $("알림").textContent = 글 || ""; }

function 화면(이름) {
  for (const s of document.querySelectorAll("section[data-화면]")) s.hidden = s.dataset.화면 !== 이름;
  $("nav").hidden = 이름 === "로그인";
  알림("");
}

let 갱신중 = null;

async function 갱신한번() {
  // 왜 localStorage 를 먼저 다시 읽는가: 다른 탭이 방금 회전했으면 새 토큰이 거기 있다.
  //   메모리의 옛 토큰으로 보내면 401 이고, 서버는 그것을 겹친 요청으로 봐 넘어가지만
  //   이 탭은 로그아웃돼 버린다.
  불러오기();
  if (!상태.refresh) return false;
  const 보낸토큰 = 상태.refresh;
  const r = await fetch("/api/auth/refresh", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: 보낸토큰 }) });
  if (r.ok) {
    const t = await r.json();
    상태.access = t.access_token; 상태.refresh = t.refresh_token; 저장();
    return true;
  }
  불러오기();
  if (상태.refresh && 상태.refresh !== 보낸토큰) return 갱신한번();   // 다른 탭이 이미 회전했다
  지우기(); 화면("로그인");
  return false;
}

function 토큰갱신() {
  // 왜 약속을 하나로 합치는가: 전적 화면은 요청 두 개를 동시에 보낸다. 둘 다 401 이면
  //   리프레시가 두 번 나가 같은 토큰이 두 번 회전을 시도한다. 한 번만 보낸다.
  if (!갱신중) 갱신중 = 갱신한번().finally(() => { 갱신중 = null; });
  return 갱신중;
}

async function 요청(경로, { method = "GET", body = null, 인증 = true, 재시도 = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (인증 && 상태.access) headers.Authorization = `Bearer ${상태.access}`;
  const r = await fetch(경로, { method, headers, body: body === null ? null : JSON.stringify(body) });
  // 왜 한 번만 다시 하는가: 리프레시 뒤에도 401이면 진짜로 로그인이 풀린 것이다.
  if (r.status === 401 && 인증 && 재시도 && (await 토큰갱신())) {
    return 요청(경로, { method, body, 인증, 재시도: false });
  }
  let 본문 = null;
  if (r.status !== 204) { try { 본문 = await r.json(); } catch (e) { 본문 = null; } }
  return { ok: r.ok, status: r.status, 본문 };
}

function 폼값(폼) { return Object.fromEntries(new FormData(폼).entries()); }

async function 토큰받기(경로, body, 폼) {
  const r = await 요청(경로, { method: "POST", body, 인증: false });
  if (!r.ok) { 알림(logic.오류글(r.본문, "실패했습니다")); return; }
  상태.access = r.본문.access_token; 상태.refresh = r.본문.refresh_token; 저장();
  if (폼) 폼.reset();
  await 로비열기();
}

async function 로비열기() {
  const 나 = await 요청("/api/auth/me");
  if (!나.ok) { 지우기(); 화면("로그인"); return; }
  상태.사용자 = 나.본문;
  const 모델들 = await 요청("/api/models", { 인증: false });
  const 선택 = $("선택-상대");
  선택.replaceChildren();
  for (const m of (모델들.본문 || [])) {
    const o = document.createElement("option");
    o.value = String(m.id);
    o.textContent = `${m.name} (${m.family}, 기대값 ${logic.부호붙이기(Number(m.exact_ev).toFixed(4))})`;
    선택.appendChild(o);
  }
  const 최근 = await 요청("/api/me/games?limit=1");
  const 열린게임 = (최근.본문 || []).find((g) => g.ended_at === null) || null;
  $("버튼-이어하기").hidden = 열린게임 === null;
  $("버튼-이어하기").dataset.gameId = 열린게임 ? String(열린게임.game_id) : "";
  $("로비-요약").textContent = `${상태.사용자.display_name}님, 상대를 고르고 시작하세요.`;
  화면("로비");
}

async function 게임열기(gameId) {
  const r = await 요청(`/api/games/${gameId}`);
  if (!r.ok) { 알림(logic.오류글(r.본문, "게임을 열 수 없습니다")); return 로비열기(); }
  상태.게임 = r.본문;
  상태.상대이름 = r.본문.opponent ? r.본문.opponent.name : null;
  테이블.게임그리기();
  화면("테이블");
}

async function 새게임() {
  const 상대 = Number($("선택-상대").value) || null;
  const r = await 요청("/api/games", { method: "POST", body: { opponent_model_id: 상대 } });
  if (r.status === 409 && r.본문 && r.본문.detail && r.본문.detail.game_id) {
    // 왜: 열린 라운드가 있으면 서버가 그 게임 번호를 준다. 이어 두는 것이 기록을 지키는 길이다.
    return 게임열기(r.본문.detail.game_id);
  }
  if (!r.ok) { 알림(logic.오류글(r.본문, "게임을 만들 수 없습니다")); return; }
  상태.게임 = r.본문;
  상태.상대이름 = r.본문.opponent ? r.본문.opponent.name : null;
  테이블.게임그리기();
  화면("테이블");
}

async function 로그아웃() {
  if (상태.refresh) await 요청("/api/auth/logout", { method: "POST", body: { refresh_token: 상태.refresh }, 인증: false });
  지우기();
  화면("로그인");
}

function 구글준비() {
  // 왜 설정을 먼저 묻는가: 클라이언트 ID 가 없는 배포(개발 PC)에서는 버튼을 그리지 않는다.
  요청("/api/auth/config", { 인증: false }).then((r) => {
    const id = r.본문 && r.본문.google_client_id;
    if (!id || !window.google) return;
    상태.구글ID = id;
    google.accounts.id.initialize({ client_id: id, callback: (resp) => 토큰받기("/api/auth/google", { credential: resp.credential }, null) });
    google.accounts.id.renderButton($("google-button"), { theme: "outline", size: "large", text: "signin_with" });
  });
}

window.addEventListener("DOMContentLoaded", () => {
  $("폼-로그인").addEventListener("submit", (e) => { e.preventDefault(); 토큰받기("/api/auth/login", 폼값(e.target), e.target); });
  $("폼-가입").addEventListener("submit", (e) => { e.preventDefault(); 토큰받기("/api/auth/signup", 폼값(e.target), e.target); });
  $("버튼-새게임").addEventListener("click", 새게임);
  $("버튼-이어하기").addEventListener("click", (e) => 게임열기(Number(e.target.dataset.gameId)));
  $("버튼-로그아웃").addEventListener("click", 로그아웃);
  for (const b of document.querySelectorAll("#nav [data-이동]")) {
    b.addEventListener("click", () => {
      if (b.dataset.이동 === "로비") 로비열기();
      else if (b.dataset.이동 === "전적") 페이지.전적열기();
      else 페이지.순위열기();
    });
  }
  불러오기();
  if (상태.access) 로비열기(); else 화면("로그인");
});

window.앱 = { 상태, 요청, 화면, 알림, 로비열기, 게임열기 };
window.구글준비 = 구글준비;

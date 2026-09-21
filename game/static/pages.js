// 이 파일은 내 전적·순위표·계정(비밀번호 변경, 전체 로그아웃) 화면을 그린다.
// 입력: /api/me/stats, /api/me/games, /api/leaderboard 응답. 출력: 표 갱신.
"use strict";

function 줄(셀들) {
  const tr = document.createElement("tr");
  for (const 글 of 셀들) { const td = document.createElement("td"); td.textContent = 글; tr.appendChild(td); }
  return tr;
}

function 항목(dl, 이름, 값) {
  const dt = document.createElement("dt"); dt.textContent = 이름;
  const dd = document.createElement("dd"); dd.textContent = 값;
  dl.append(dt, dd);
}

async function 전적열기() {
  const [통계, 게임들] = await Promise.all([앱.요청("/api/me/stats"), 앱.요청("/api/me/games?limit=20")]);
  if (!통계.ok) { 앱.알림(logic.오류글(통계.본문, "전적을 읽을 수 없습니다")); return; }
  const s = 통계.본문;
  const dl = $("전적표");
  dl.replaceChildren();
  항목(dl, "게임 / 라운드 / 결정", `${s.games} / ${s.rounds} / ${s.decisions}`);
  항목(dl, "승 / 패 / 무", `${s.wins} / ${s.losses} / ${s.pushes} (승률 ${logic.퍼센트(s.win_rate)})`);
  항목(dl, "기본전략 일치율", logic.퍼센트(s.agreement));
  항목(dl, "누적 기대값 손실", `${Number(s.ev_loss_total).toFixed(3)} (결정당 ${Number(s.ev_loss_per_decision).toFixed(4)})`);
  항목(dl, "순손익", logic.부호붙이기(s.net_result));
  항목(dl, "순위표 자격", s.rank_eligible ? "있음" : `아직 없음 (200결정 필요, 지금 ${s.decisions})`);
  const 몸 = $("게임목록-몸");
  몸.replaceChildren(...(게임들.본문 || []).map((g) => 줄([
    `#${g.game_id}${g.ended_at ? "" : " (진행 중)"}`, g.opponent_name || "없음", String(g.rounds),
    logic.부호붙이기(g.net_result), logic.퍼센트(g.agreement)])));
  $("폼-비밀번호").hidden = 앱.상태.사용자.auth_provider !== "local";
  앱.화면("전적");
}

async function 순위열기() {
  const r = await 앱.요청("/api/leaderboard?limit=50", { 인증: false });
  if (!r.ok) { 앱.알림(logic.오류글(r.본문, "순위표를 읽을 수 없습니다")); return; }
  const 몸 = $("순위-몸");
  몸.replaceChildren(...r.본문.map((x) => {
    const tr = 줄([String(x.rank), x.display_name, String(x.games), String(x.decisions),
                  logic.퍼센트(x.agreement), Number(x.ev_loss_per_decision).toFixed(4)]);
    if (앱.상태.사용자 && x.user_id === 앱.상태.사용자.id) tr.className = "나";
    return tr;
  }));
  앱.화면("순위");
}

async function 비밀번호바꾸기(e) {
  e.preventDefault();
  const r = await 앱.요청("/api/auth/password", { method: "POST", body: Object.fromEntries(new FormData(e.target).entries()) });
  if (!r.ok) { 앱.알림(logic.오류글(r.본문, "바꾸지 못했습니다")); return; }
  // 왜 토큰을 바꿔 끼우는가: 서버가 옛 토큰을 전부 죽였다. 새 쌍을 받아 그대로 이어 쓴다.
  앱.상태.access = r.본문.access_token; 앱.상태.refresh = r.본문.refresh_token;
  try { localStorage.setItem("bj_tokens", JSON.stringify({ access: 앱.상태.access, refresh: 앱.상태.refresh })); } catch (err) { /* 무시 */ }
  e.target.reset();
  앱.알림("비밀번호를 바꿨습니다. 다른 기기는 모두 로그아웃되었습니다.");
}

async function 전체로그아웃() {
  await 앱.요청("/api/auth/logout-all", { method: "POST" });
  앱.상태.access = null; 앱.상태.refresh = null; 앱.상태.사용자 = null; 앱.상태.게임 = null;
  try { localStorage.removeItem("bj_tokens"); } catch (err) { /* 무시 */ }
  앱.화면("로그인");
}

window.addEventListener("DOMContentLoaded", () => {
  $("폼-비밀번호").addEventListener("submit", 비밀번호바꾸기);
  $("버튼-전체로그아웃").addEventListener("click", 전체로그아웃);
});

window.페이지 = { 전적열기, 순위열기 };

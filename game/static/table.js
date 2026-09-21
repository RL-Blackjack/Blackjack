// 이 파일은 테이블 화면을 그리고 행동을 서버에 보낸다. 상태는 앱.상태.게임 하나뿐이다.
// 입력: GameOut / ActOut / RoundOut 응답. 출력: 카드·버튼·피드백 갱신.
"use strict";

function 카드요소(값) {
  const s = document.createElement("span");
  s.className = "카드";
  s.textContent = logic.카드표시(값);
  return s;
}

function 카드들(컨테이너, 값들) {
  컨테이너.replaceChildren(...값들.map(카드요소));
}

function 손요소(제목, 카드값들, 설명, 강조) {
  const d = document.createElement("div");
  d.className = 강조 ? "손 지금" : "손";
  const h = document.createElement("h3");
  h.textContent = 제목;
  const c = document.createElement("div");
  c.className = "카드들";
  카드들(c, 카드값들);
  const p = document.createElement("p");
  p.textContent = 설명;
  d.append(h, c, p);
  return d;
}

function 게임그리기() {
  const 게임 = 앱.상태.게임;
  const 라운드 = 게임.round;
  $("상대이름").textContent = 앱.상태.상대이름 ? `상대: ${앱.상태.상대이름}` : "상대 없음";
  $("회차").textContent = `${게임.round_no}회차`;
  const 손들 = $("손들");
  손들.replaceChildren();
  if (라운드.status === "pending") {
    카드들($("딜러카드"), [라운드.dealer_up]);
    // 왜 손 하나만 그리는가: 서버는 지금 결정을 기다리는 손만 준다(다른 손의 카드는
    //   재생 없이는 모른다). 스플릿이면 "2번째 손 / 총 2손"으로 자리를 알린다.
    const 제목 = 라운드.n_hands > 1 ? `${라운드.hand_index + 1}번째 손 / 총 ${라운드.n_hands}손` : "내 손";
    const 합 = `${라운드.is_soft ? "소프트 " : ""}${라운드.total} · 베팅 ${라운드.bet}`;
    손들.appendChild(손요소(제목, 라운드.player_cards, 합, true));
    $("버튼-다음라운드").hidden = true;
  } else {
    카드들($("딜러카드"), 라운드.dealer_cards);
    라운드.hands.forEach((h, i) => {
      const 제목 = 라운드.hands.length > 1 ? `${i + 1}번째 손` : "내 손";
      손들.appendChild(손요소(제목, h.cards, `${h.total} · ${logic.손결과문장(h.result)} (${logic.부호붙이기(h.result)})`, false));
    });
    const 합계 = document.createElement("p");
    합계.className = "순손익";
    합계.textContent = `이번 라운드 ${logic.부호붙이기(라운드.net)}`;
    손들.appendChild(합계);
    $("버튼-다음라운드").hidden = false;
  }
  logic.버튼가능(라운드).forEach((가능, i) => { $(`행동-${i}`).disabled = !가능; });
}

async function 다시읽기() {
  const r = await 앱.요청(`/api/games/${앱.상태.게임.game_id}`);
  if (r.ok) { 앱.상태.게임 = r.본문; 게임그리기(); }
}

async function 충돌처리(본문) {
  const code = 본문 && 본문.detail ? 본문.detail.code : "";
  if (code === "stale_seq" || code === "round_finished") return 다시읽기();
  if (code === "no_open_round") return 다음라운드();
  if (code === "rules_changed") {
    앱.알림("규칙이 바뀌어 이 게임은 이어 둘 수 없습니다. 게임을 끝냅니다.");
    return 게임끝내기();
  }
  if (code === "game_ended" || code === "open_round") return 앱.로비열기();
  앱.알림(logic.오류글(본문, "요청이 거절되었습니다"));
}

async function 행동하기(action) {
  const 게임 = 앱.상태.게임;
  if (게임.round.status !== "pending") return;
  for (let i = 0; i < 4; i += 1) $(`행동-${i}`).disabled = true;   // 왜: 두 번 눌림 방지
  const r = await 앱.요청(`/api/games/${게임.game_id}/act`, { method: "POST", body: { seq: 게임.round.seq, action } });
  if (r.status === 409) return 충돌처리(r.본문);
  if (!r.ok) { 앱.알림(logic.오류글(r.본문, "행동을 보낼 수 없습니다")); return 게임그리기(); }
  $("피드백").textContent = logic.피드백문장(r.본문.feedback, 앱.상태.상대이름);
  게임.round = r.본문.round;
  게임그리기();
}

async function 다음라운드() {
  const 게임 = 앱.상태.게임;
  const r = await 앱.요청(`/api/games/${게임.game_id}/rounds`, { method: "POST" });
  if (r.status === 409) return 충돌처리(r.본문);
  if (!r.ok) { 앱.알림(logic.오류글(r.본문, "라운드를 열 수 없습니다")); return; }
  게임.round_no = r.본문.round_no;
  게임.round = r.본문.round;
  $("피드백").textContent = "";
  게임그리기();
}

async function 게임끝내기() {
  const r = await 앱.요청(`/api/games/${앱.상태.게임.game_id}/finish`, { method: "POST" });
  if (r.status === 409) return 충돌처리(r.본문);
  앱.상태.게임 = null;
  return 앱.로비열기();
}

window.addEventListener("DOMContentLoaded", () => {
  for (let i = 0; i < 4; i += 1) $(`행동-${i}`).addEventListener("click", () => 행동하기(i));
  $("버튼-다음라운드").addEventListener("click", 다음라운드);
  $("버튼-게임끝").addEventListener("click", () => {
    // 왜 묻는가: 열린 라운드가 있으면 서버가 409 open_round 로 막는다. 먼저 알려 준다.
    if (앱.상태.게임 && 앱.상태.게임.round.status === "pending") { 앱.알림("이 라운드를 마친 뒤 끝낼 수 있습니다."); return; }
    게임끝내기();
  });
});

window.테이블 = { 게임그리기 };

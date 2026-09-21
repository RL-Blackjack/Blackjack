// 이 파일은 화면 계산과 문장 만들기만 한다. DOM도 fetch도 없다(그래서 Node로 테스트한다).
// 입력: 서버 응답 객체. 출력: 문자열과 불리언.
"use strict";

const 행동이름 = { 0: "스탠드", 1: "히트", 2: "더블", 3: "스플릿" };

// 왜 11만 바꾸는가: 엔진은 카드를 값(2~11)으로만 준다. 11이 에이스다. 무늬는 없다.
function 카드표시(값) { return 값 === 11 ? "A" : String(값); }

function 버튼가능(라운드) {
  if (!라운드 || 라운드.status !== "pending") return [false, false, false, false];
  return 라운드.legal.map((x) => x === 1);
}

function 소수(값, 자리) { return Number(값).toFixed(자리); }

function 피드백문장(피드백, 상대이름) {
  const 정답 = 행동이름[피드백.dp_optimal_action];
  const ai = 피드백.ai_action === null || 피드백.ai_action === undefined || !상대이름
    ? "" : ` ${상대이름}${피드백.was_optimal ? "도" : "는"} ${행동이름[피드백.ai_action]}를 골랐습니다.`;
  if (피드백.was_optimal) return `정답! ${정답}가 가장 좋은 선택이었습니다.${ai}`;
  return `정답은 ${정답}였습니다(기대값 손실 ${소수(피드백.dp_ev_loss, 3)}).${ai}`;
}

function 손결과문장(result) {
  if (result > 0) return "이김";
  if (result < 0) return "짐";
  return "무승부";
}

function 부호붙이기(값) {
  const n = Number(값);
  if (n > 0) return `+${n}`;
  if (n < 0) return `−${Math.abs(n)}`;   // 왜 U+2212인가: 하이픈보다 숫자 옆에서 또렷하다.
  return "0";
}

function 퍼센트(x) { return `${소수(x * 100, 1)}%`; }

function 오류글(본문, 기본) {
  if (!본문 || 본문.detail === undefined) return 기본;
  const d = 본문.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((e) => `${(e.loc || []).filter((x) => x !== "body").join(".")}: ${e.msg}`).join(", ");
  }
  return d.message || 기본;
}

const logic = { 카드표시, 행동이름, 버튼가능, 피드백문장, 손결과문장, 부호붙이기, 퍼센트, 오류글 };
if (typeof module !== "undefined") module.exports = logic;
if (typeof window !== "undefined") window.logic = logic;

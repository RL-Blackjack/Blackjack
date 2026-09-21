// 이 파일은 logic.js 의 순수 함수(카드 표시·버튼 가능 여부·피드백 문장)를 확인한다.
// 입력: 서버 응답 모양의 자바스크립트 객체. 출력: node --test 결과.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const logic = require("../../game/static/logic.js");

test("11은 에이스로 보인다", () => {
  assert.equal(logic.카드표시(11), "A");
  assert.equal(logic.카드표시(10), "10");
  assert.equal(logic.카드표시(2), "2");
});

test("버튼은 legal 그대로 켜진다", () => {
  assert.deepEqual(logic.버튼가능({ status: "pending", legal: [1, 1, 0, 0] }),
                   [true, true, false, false]);
  assert.deepEqual(logic.버튼가능({ status: "finished" }), [false, false, false, false]);
});

test("정답이면 칭찬, 아니면 정답과 손실을 말한다", () => {
  const 좋음 = { dp_optimal_action: 0, dp_ev_loss: 0, was_optimal: true, ai_action: 0 };
  assert.equal(logic.피드백문장(좋음, "랜덤포레스트"),
               "정답! 스탠드가 가장 좋은 선택이었습니다. 랜덤포레스트도 스탠드를 골랐습니다.");
  const 나쁨 = { dp_optimal_action: 1, dp_ev_loss: 0.0213, was_optimal: false, ai_action: 0 };
  assert.equal(logic.피드백문장(나쁨, "랜덤포레스트"),
               "정답은 히트였습니다(기대값 손실 0.021). 랜덤포레스트는 스탠드를 골랐습니다.");
  assert.equal(logic.피드백문장({ ...나쁨, ai_action: null }, null),
               "정답은 히트였습니다(기대값 손실 0.021).");
});

test("손 결과와 순손익 표기", () => {
  assert.equal(logic.손결과문장(1.5), "이김");
  assert.equal(logic.손결과문장(-1), "짐");
  assert.equal(logic.손결과문장(0), "무승부");
  assert.equal(logic.부호붙이기(1.5), "+1.5");
  assert.equal(logic.부호붙이기(-1), "−1");
  assert.equal(logic.부호붙이기(0), "0");
  assert.equal(logic.퍼센트(0.9181), "91.8%");
});

test("서버 오류 본문에서 사람이 읽을 글을 꺼낸다", () => {
  assert.equal(logic.오류글({ detail: "로그인이 필요하다" }, "실패"), "로그인이 필요하다");
  assert.equal(logic.오류글({ detail: { code: "open_round", message: "아직" } }, "실패"), "아직");
  assert.equal(logic.오류글({ detail: [{ loc: ["body", "email"], msg: "invalid" }] }, "실패"),
               "email: invalid");
  assert.equal(logic.오류글(null, "실패"), "실패");
});

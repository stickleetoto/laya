"""Small Korean baseline harness for Laya.

This is intentionally a smoke benchmark, not a publishable benchmark. Its job is
to expose obvious Korean failure modes before we build a larger held-out set.

Examples:
    python research/korean/baseline.py --device cpu
    python research/korean/baseline.py --device cuda
"""

import argparse
import json
import time

from laya import KoreanDecisionEngine


DEPARTMENT_CRITERIA = {
    "billing": "결제, 청구서, 환불, 중복 결제",
    "technical": "버그, 장애, 설치, 연결 또는 시스템 오류",
    "sales": "가격, 견적, 구매 또는 계약 문의",
    "account": "로그인, 계정, 비밀번호, 권한",
    "other": "그 외 요청",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "이 요청을 담당해야 할 부서를 고르십시오.",
        "criteria": DEPARTMENT_CRITERIA,
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "사용자가 명시적으로 환불을 요청했습니까?",
    },
    "urgency": {
        "type": "score",
        "instructions": "요청의 긴급도를 평가하십시오.",
        "criteria": [
            "긴급하지 않음",
            "빠른 처리가 필요함",
            "즉시 처리해야 하는 중요한 문제",
        ],
    },
}

CASES = [
    {
        "name": "billing_duplicate_refund",
        "state": {"message": "카드 결제가 두 번 됐어요. 중복 결제된 금액을 환불해 주세요."},
        "department": "billing",
        "refund": True,
    },
    {
        "name": "billing_invoice",
        "state": {"message": "지난달 청구서 금액이 계약한 금액과 다릅니다. 확인 부탁드립니다."},
        "department": "billing",
        "refund": False,
    },
    {
        "name": "technical_outage",
        "state": {"message": "서비스가 계속 500 오류를 내고 접속이 안 됩니다."},
        "department": "technical",
        "refund": False,
    },
    {
        "name": "technical_install",
        "state": {"message": "업데이트 후 프로그램이 실행되지 않고 바로 종료됩니다."},
        "department": "technical",
        "refund": False,
    },
    {
        "name": "sales_quote",
        "state": {"message": "기업용 50석 라이선스 견적과 연간 계약 가격을 알고 싶습니다."},
        "department": "sales",
        "refund": False,
    },
    {
        "name": "account_login",
        "state": {"message": "비밀번호를 바꾼 뒤 계정에 로그인할 수 없습니다."},
        "department": "account",
        "refund": False,
    },
    {
        "name": "casual_spacing_noise",
        "state": {"message": "결제두번됨 환불좀 해주세요"},
        "department": "billing",
        "refund": True,
    },
    {
        "name": "mixed_korean_english",
        "state": {"message": "API 연결하면 timeout 나고 login도 풀립니다. 버그 확인해주세요."},
        "department": "technical",
        "refund": False,
    },
    {
        "name": "other_feedback",
        "state": {"message": "새 디자인이 전보다 보기 편해졌습니다. 의견 남깁니다."},
        "department": "other",
        "refund": False,
    },
    {
        "name": "urgent_refund",
        "state": {"message": "오늘 결제 마감인데 잘못 청구됐습니다. 지금 바로 취소하고 환불해 주세요."},
        "department": "billing",
        "refund": True,
    },
]


def run(device: str, model: str):
    engine = KoreanDecisionEngine(device=device, preload=True, model=model)

    choice_hits = 0
    noul_hits = 0
    rows = []

    started = time.perf_counter()
    for case in CASES:
        t0 = time.perf_counter()
        result = engine.predict(case["state"], QUESTIONS)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        answers = result["answers"]

        got_department = answers["department"]["choice"]
        refund_p = float(answers["refund_requested"]["noul"])
        got_refund = refund_p >= 0.5

        choice_ok = got_department == case["department"]
        noul_ok = got_refund == case["refund"]
        choice_hits += int(choice_ok)
        noul_hits += int(noul_ok)

        rows.append(
            {
                "name": case["name"],
                "expected_department": case["department"],
                "department": got_department,
                "department_confidence": answers["department"].get("confidence"),
                "department_ok": choice_ok,
                "expected_refund": case["refund"],
                "refund_probability": refund_p,
                "refund_ok": noul_ok,
                "urgency_score": answers["urgency"]["score"],
                "latency_ms": round(elapsed_ms, 2),
                "routing": result.get("routing"),
            }
        )

    total_ms = (time.perf_counter() - started) * 1000.0
    summary = {
        "model": model,
        "device": device,
        "cases": len(CASES),
        "choice_accuracy": choice_hits / len(CASES),
        "noul_accuracy": noul_hits / len(CASES),
        "total_ms": round(total_ms, 2),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="multilingual")
    args = parser.parse_args()
    run(args.device, args.model)

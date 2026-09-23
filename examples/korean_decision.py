"""Minimal Korean Laya example.

Run:
    python examples/korean_decision.py
"""

from pprint import pprint

from laya import KoreanDecisionEngine


engine = KoreanDecisionEngine(device="cuda", preload=True)

state = {
    "request": "결제가 두 번 됐습니다. 오늘 안에 중복 결제를 환불해 주세요.",
    "account_tier": "standard",
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "어느 부서가 이 요청을 처리해야 합니까?",
        "criteria": {
            "billing": "결제, 청구서, 환불, 중복 결제",
            "technical": "버그, 장애, 연결 오류",
            "sales": "가격, 구매, 계약 문의",
            "other": "그 외 요청",
        },
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

result = engine.predict(state, questions)
pprint(result)

from laya.korean import KoreanDecisionEngine


def test_korean_engine_pins_multilingual(monkeypatch):
    engine = KoreanDecisionEngine(model="multilingual")
    seen = {}

    def fake_predict(state, questions, **kwargs):
        seen.update(kwargs)
        return {"answers": {}, "routing": {"model": kwargs["model"]}}

    monkeypatch.setattr(engine.router, "predict", fake_predict)

    result = engine.predict({"message": "환불해 주세요"}, {})

    assert seen["model"] == "multilingual"
    assert seen["lang"] == "ko"
    assert result["routing"]["model"] == "multilingual"


def test_korean_engine_route_is_weight_free(monkeypatch):
    engine = KoreanDecisionEngine(model="multilingual")
    seen = {}

    def fake_route(state, questions, **kwargs):
        seen.update(kwargs)
        return {"model": kwargs["model"], "reason": "test"}

    monkeypatch.setattr(engine.router, "route", fake_route)

    decision = engine.route("로그인이 안 됩니다.")

    assert seen["model"] == "multilingual"
    assert seen["lang"] == "ko"
    assert decision["model"] == "multilingual"

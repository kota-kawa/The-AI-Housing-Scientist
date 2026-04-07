from app.stages.risk_check import run_risk_check


def test_risk_check_detects_major_items():
    text = "更新料1ヶ月、短期解約違約金あり、解約予告2か月前、保証会社加入必須"

    result = run_risk_check(source_text=text)

    risk_types = {item["risk_type"] for item in result["risk_items"]}

    assert "renewal_fee" in risk_types
    assert "early_termination" in risk_types
    assert "notice_period" in risk_types
    assert "guarantor" in risk_types

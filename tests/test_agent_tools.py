from neuroglyph_agent.mcp_server import call_tool, list_tools


def test_agent_tools_list():
    tools = list_tools()
    names = {t["name"] for t in tools}
    assert "train_decoder" in names
    assert "send_prediction_to_unreal" in names


def test_send_prediction_to_unreal_high_conf():
    out = call_tool("send_prediction_to_unreal", {"prediction": "left", "confidence": 0.9})
    assert out["status"] == "ok"
    assert "action" in out


def test_send_prediction_to_unreal_low_conf():
    out = call_tool("send_prediction_to_unreal", {"prediction": "left", "confidence": 0.2})
    assert out["status"] == "ignored"
import llm


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeClient:
    def __init__(self, message):
        self._message = message

        class _Completions:
            def create(inner_self, **kwargs):
                self.kwargs = kwargs
                return _FakeResponse(self._message)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_chat_normalizes_tool_calls(monkeypatch):
    msg = _FakeMessage(None, [_FakeToolCall("call_1", "retrieve", '{"query": "x"}')])
    fake = _FakeClient(msg)
    monkeypatch.setattr(llm, "_get_client", lambda: fake)

    out = llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert out["content"] is None
    assert out["tool_calls"] == [
        {"id": "call_1", "name": "retrieve", "arguments": '{"query": "x"}'}
    ]


def test_chat_normalizes_plain_content(monkeypatch):
    msg = _FakeMessage("hello", None)
    fake = _FakeClient(msg)
    monkeypatch.setattr(llm, "_get_client", lambda: fake)

    out = llm.chat([{"role": "user", "content": "hi"}])

    assert out["content"] == "hello"
    assert out["tool_calls"] == []

from openai import AssistantEventHandler
from typing_extensions import override


class EventHandler(AssistantEventHandler):    
    def __init__(self):
        super().__init__()
        self.response_data = {
            "text_created": "",
            "text_delta": "",
            "tool_call_type": "",
            "code_interpreter_logs": []
        }

    @override
    def on_text_created(self, text):
        self.response_data["text_created"] += f"\nassistant > {text}"
        yield self.response_data  # Yield updated data after text is created

    @override
    def on_text_delta(self, delta, snapshot):
        self.response_data["text_delta"] += delta.value
        yield self.response_data  # Yield updated data after each text delta

    def on_tool_call_created(self, tool_call):
        self.response_data["tool_call_type"] = tool_call.type
        yield self.response_data  # Yield updated data after tool call is created

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                self.response_data["text_delta"] += delta.code_interpreter.input
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        self.response_data["code_interpreter_logs"].append(output.logs)
            yield self.response_data  # Yield updated data after tool call delta


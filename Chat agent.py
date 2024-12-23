import os
import warnings
from dotenv import load_dotenv
from typing import List, Dict, Any
from haystack import Pipeline, component
from haystack.utils.auth import Secret
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.joiners import BranchJoiner
import json

warnings.filterwarnings("ignore")
load_dotenv()

openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

@component
class SimpleFunctionCaller:
    def __init__(self, available_functions: Dict[str, Any]):
        self.available_functions = available_functions

    def run(self, messages: List[ChatMessage]) -> Dict[str, List[ChatMessage]]:
        function_replies = []
        for message in messages:
            if message.role == "assistant" and message.content:
                try:
                    data = json.loads(message.content)
                    name = data["function"]["name"]
                    args = data["function"]["arguments"]
                    if name in self.available_functions:
                        result = self.available_functions[name](**args)
                        function_replies.append(
                            ChatMessage(content=result, role="function", name="function_caller")
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    function_replies.append(
                        ChatMessage(
                            content=f"Error processing function call: {str(e)}",
                            role="function",
                            name="function_caller"
                        )
                    )
        return {"function_replies": function_replies}

@component
class ChatMessagePromptBuilder:
    def run(self, messages: List[ChatMessage]) -> Dict[str, str]:
        lines = []
        for m in messages:
            if m.role == "system":
                lines.append(f"System: {m.content}")
            elif m.role == "user":
                lines.append(f"User: {m.content}")
            elif m.role == "assistant":
                lines.append(f"Assistant: {m.content}")
            else:
                lines.append(f"{m.role.capitalize()}: {m.content}")
        prompt = "\n".join(lines)
        return {"prompt": prompt}

def rag_pipeline_func(query: str) -> str:
    return f"RAG pipeline result for query: {query}"

def get_current_weather(location: str) -> str:
    return f"Current weather in {location} is sunny."

available_functions = {
    "rag_pipeline_func": rag_pipeline_func,
    "get_current_weather": get_current_weather
}

function_caller = SimpleFunctionCaller(available_functions=available_functions)

replies = {
    "replies": [
        ChatMessage(
            content='{"function": {"name": "rag_pipeline_func", "arguments": {"query": "Where does Mark live?"}}}',
            role="assistant",
            name="assistant"
        )
    ]
}

results = function_caller.run(messages=replies["replies"])
print(results["function_replies"])

message_collector = BranchJoiner(type_=List[ChatMessage])
prompt_builder = ChatMessagePromptBuilder()
chat_generator = OpenAIGenerator(model="gpt-3.5-turbo", api_key=openai_api_key)
function_caller = SimpleFunctionCaller(available_functions=available_functions)

chat_agent = Pipeline()
chat_agent.add_component("message_collector", message_collector)
chat_agent.add_component("prompt_builder", prompt_builder)
chat_agent.add_component("generator", chat_generator)
chat_agent.add_component("function_caller", function_caller)

# Correctly connecting components with a connection dictionary
chat_agent.connect(
    "message_collector",
    "prompt_builder",
    {"messages": "messages"}
)
chat_agent.connect(
    "prompt_builder",
    "generator",
    {"prompt": "prompt"}
)
chat_agent.connect(
    "generator",
    "function_caller",
    {"replies": "messages"}
)
chat_agent.connect(
    "function_caller",
    "message_collector",
    {"function_replies": "messages"}
)

chat_agent.draw("chat_agent_pipeline.png")

messages = [
    ChatMessage.from_system(
        """If needed, break down the user's question into simpler questions and follow-up questions that you can use with your tools.
Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.""",
        name="system"
    )
]

while True:
    user_input = input("User: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting chat.")
        break
    messages.append(ChatMessage.from_user(user_input, name="user"))
    result = chat_agent.run({"message_collector": {"messages": messages}})
    
    # Handle cases where generator might not return any replies
    generator_output = result.get("generator", {})
    replies = generator_output.get("replies", [])
    if not replies:
        print("Assistant: I'm sorry, I couldn't process that.")
        continue
    
    reply = replies[0]
    print(f"Assistant: {reply.content}")
    messages.append(ChatMessage(content=reply.content, role="assistant", name="assistant"))

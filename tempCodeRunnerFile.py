from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import TOOLS, run_tool
from Utilities.utilities import log   # changed: import the function, not the module
from types import SimpleNamespace

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []
messages.append({
    "role": "system",
    "content": "You are a helpful assistant and always answer in a concise manner."
})

def stream_chat(history):
    stream = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=history,
            tools=TOOLS,
            stream=True,
        )

    full_response = ""
    calls = {}
    label_printed = False

    #Streaming started
    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            #1. Content Streaming : Do we have message (content) from the LLM
            if delta.content:
                if not label_printed:
                    print("Assistant: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            #2. Tool Calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments

    except KeyboardInterrupt:
        print("\n {Stopped By You}")
        calls = {}
    
    if label_printed:
        print()  # Move to the next line after the assistant's response

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])
    
    return full_response, calls_list


def build_tool_calls_payload(tool_calls):
    """
    Converts our collected tool calls into the dict format
    that the API expects inside an assistant message.
    """

    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        payload.append(tool_call_dict)

    return payload
    



while True:
    user_input = input("\nUser: ")

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        for _ in range(10):
            full_reponse, tool_calls = stream_chat(messages)

            # No tool calls -> final answer (already streamed to the screen)
            if not tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": full_reponse
                })
                break

            log(f"  Step {_+1}: Model called tool '{tool_calls[0].function.name}' with arguments: {tool_calls[0].function.arguments}")
            
            messages.append({
                "role": "assistant",
                "content": full_reponse,
                "tool_calls": build_tool_calls_payload(tool_calls)  
            })

            for tool_call in tool_calls:
                result = run_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        else:
            print("Assistant: Maximum tool-call iterations reached.")

    except Exception as e:
        print(f"Error: {str(e)}")
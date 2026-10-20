from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import FUNCTION_NAME_TO_FUNCTION_MAP, TOOLS
import json
from Utilities.utilities import log

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
            response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=messages,
                tools=TOOLS
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
                print("Assistant: " + (msg.content or ""))
                break

            log(f"step {_+1}: model called tool '{msg.tool_calls[0].function.name}' with arguments: {msg.tool_calls[0].function.arguments}")
            
            messages.append({
                "role": "assistant",
                "content": msg.content,
            })

            for tool_call in msg.tool_calls:
                try:
                    func_name = tool_call.function.name
                    func_args = json.loads(
                        tool_call.function.arguments or "{}"
                    )
                    log(f"  Calling function '{func_name}' with arguments: {func_args}")
                    fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
                    result = fn(**func_args)

                except Exception as e:
                    result = f"Error: {str(e)}"
                log(f"  -> result: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        else:
            print("Assistant: Maximum tool-call iterations reached.")

    except Exception as e:
        print(f"Error: {str(e)}")
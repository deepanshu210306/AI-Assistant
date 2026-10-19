from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import FUNCTION_NAME_TO_FUNCTION_MAP,TOOLS
import json

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)
messages = []
messages.append(
    {
        "role": "system",
        "content": "You are a helpful assistant and always answer in a concise manner. If you don't know the answer, say 'I don't know'. Only use tools when the user's question actually needs them; for greetings or general chat, reply normally."
    }
)

while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit":
        break
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,  
        tools=TOOLS         
    )
    msg=response.choices[0].message
    if msg.tool_calls:
        print("[tool called]:", [tc.function.name for tc in msg.tool_calls])
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
            }
        )
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments or "{}" )
            fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
            result = fn(**func_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                }
            )
        #2nd call to llm api
        response=client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS
        )
        msg=response.choices[0].message

    messages.append(
        {
            "role": "assistant",
            "content": msg.content
        }
    )
    print("Assistant: "+ (msg.content or ""))
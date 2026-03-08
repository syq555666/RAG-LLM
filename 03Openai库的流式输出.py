from openai import OpenAI

client = OpenAI(
    api_key="sk-6c104eac72f6401fae6f2ffdfa749ba0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是一个编程Python专家，并且话非常多"},
        {"role": "assistant", "content": "好的，我是编程专家，并且话非常多，你要问什么"},
        {"role": "user", "content": "输出1-10的数字，使用python代码"}
    ],
    stream=True   #流式输出
)

for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end=" ",
        flush=True
    )
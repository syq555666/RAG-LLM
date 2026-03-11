from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")


example_template = PromptTemplate.from_template("单词：{word},反义词：{antonym}")

examples_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]

few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,                      #示例数据的模版
    examples=examples_data,                               #示例的数据
    prefix="告知我单词的反义词，我提供如下的示例：",            #示例之前的提示词
    suffix="基于前面的示例告诉我，{input_word}的反义词是？",    #示例之后的提示词
    input_variables=['input_word']                         #变量名
)

prompt_text = few_shot_template.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)

model =Tongyi(model="qwen-max")

print(model.invoke(input=prompt_text))
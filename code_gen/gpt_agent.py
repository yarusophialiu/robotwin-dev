from openai import OpenAI

kimi_api = "Your key"
openai_api = "Your key"
deep_seek_api = "sk-8523b7c37d14408aa3454107fd46c75a"

# Configure the API and key (using DeepSeek as an example)
def generate(message, gpt="pangu", temperature=0):

    if gpt == "deepseek":
        MODEL = "deepseek-chat"
        OPENAI_API_BASE = "https://api.deepseek.com"
        # Set your API key here
        OPENAI_API_KEY = deep_seek_api
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    elif gpt == "openai":
        MODEL = "gpt-4o"
        OPENAI_API_BASE = "https://api.gptapi.us/v1"
        OPENAI_API_KEY = openai_api
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    
    elif gpt == "pangu":
        print('Using OpenPangu model')
        MODEL="openPangu-Embedded-7B"
        OPENAI_API_BASE = "https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-3365c7ce-af12-4ea8-b5c3-5826e504190c/user-aa58f4ef-3b12-48e4-90c9-a34b3453cadb/vscode/89da0703-815d-497b-8621-08ec51c45b4e/0aa4160f-d5ac-41aa-bba4-001ef2d17433/proxy/8818/v1"
        OPENAI_API_KEY = "EMPTY"
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        
    elif gpt =="qwen":
        MODEL="Qwen3-8B"
        OPENAI_API_BASE = "https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-3365c7ce-af12-4ea8-b5c3-5826e504190c/user-aa58f4ef-3b12-48e4-90c9-a34b3453cadb/vscode/51a800ae-cc25-4947-8c4a-e239c23c8018/d93b6d70-607c-4443-8009-e3abdb676412/proxy/8000/v1"
        OPENAI_API_KEY = "EMPTY"
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    else:
        raise ValueError(f"Unsupported API provider: {gpt}")

    print('start generating')
    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
        stream=False,
        temperature=temperature,
    )
    print('end generating')

    return response.choices[0].message.content



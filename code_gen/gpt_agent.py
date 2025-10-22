from openai import OpenAI

kimi_api = "sk-s3JarjxSMEMRWxk1zW3bHdmDOXbDJkOymkaa33OYqu0J3eIQ"
openai_api = "Your key"
deep_seek_api = "sk-2775925c7b3842d8ac982a7ead8d3539"

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
        OPENAI_API_BASE = "https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-3365c7ce-af12-4ea8-b5c3-5826e504190c/user-2af699bc-efc5-4399-a3be-ae961f37a7bb/vscode/b3f65e89-f416-429a-808e-253c01c15faf/aefded10-dc68-42d1-a8c4-99c0399c5671/proxy/8818/v1"
        OPENAI_API_KEY = "EMPTY"
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        
    elif gpt =="qwen":
        MODEL="qwen/qwen3-8b:free"
        OPENAI_API_BASE = "https://openrouter.ai/api/v1"
        OPENAI_API_KEY = "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"
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



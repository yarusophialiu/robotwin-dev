import sys, os, json, importlib, inspect, re

# Add the project root directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_agent import *
from prompt_general_template import *
from task_info import *
from test_gen_code import * 
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# 定义所有需要替换的参数键
REQUIRED_KEYS = [
    'tgt_object',
    'pre_grasp_dis',
    'grasp_dis',
    'contract_point_id_grasp',
    'move_by_displacement_z',
    'target',
    'target_func_point_id',
    'functional_point_id',
    'place_pre_dis',
    'place_dis'
]

def generate_pick_and_place_code(json_string: str) -> str:    
    # 1. 尝试解析JSON
    try:
        if "```json" in json_string:
            parts = json_string.split("```json")
            # 取分割后的第二部分，再去除结尾的```标记及可能的空白字符
            json_string = parts[1].rsplit("```", 1)[0].strip()
        params = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(json_string)
        raise ValueError(f"Invalid JSON format provided. Error: {e}")

    # 2. 检查是否为字典
    if not isinstance(params, dict):
        raise ValueError(f"Invalid JSON type. Expected a dictionary (object), but got {type(params).__name__}.")

    # 3. 检查所有必需的键
    missing_keys = []
    for key in REQUIRED_KEYS:
        if key not in params:
            missing_keys.append(key)
            
    if missing_keys:
        raise ValueError(f"Missing required parameters in JSON: {', '.join(missing_keys)}")

    # 4. 生成代码
    code = PICK_AND_PLACE_CODE_EXAMPLE
    
    for key in REQUIRED_KEYS:
        value = params[key]
                
        placeholder_line = f"{key}=PLACEHOLDER"
        replacement_line = f"{key}={value}"
        
        code = code.replace(placeholder_line, replacement_line)

    return code,json_string


def generate_gpt_task_code(task_name: str):
    module = importlib.import_module(f"envs.{task_name}")
    task_class = getattr(module, task_name)
    play_once_code = inspect.getsource(task_class.play_once)
    # 让源码缩进正确（删掉前导空格）
    play_once_code = "\n".join(["    " + line for line in play_once_code.splitlines()])
    
    res = f'''
from envs._base_task import Base_Task
from envs.{task_name} import {task_name}
from envs.utils import *
import sapien

class gpt_{task_name}({task_name}):
{play_once_code}
'''
    return res


def generate_current_code(task_name: str, current_code: str):
    module = importlib.import_module(f"envs.{task_name}")
    task_class = getattr(module, task_name)
    play_once_src = inspect.getsource(task_class.play_once)  # starts with "def play_once(self):"

    # 2) Find the indentation level of the placeholder in current_code
    #    We match a line like: "<indent>def play_once(self): pass"
    pattern = re.compile(r'(^[ \t]*)def\s+play_once\s*\(\s*self\s*\)\s*:\s*pass\s*$', re.MULTILINE)
    m = pattern.search(current_code)
    if not m:
        raise ValueError("Could not find 'def play_once(self): pass' placeholder in current_code.")
    indent = m.group(1)
    reduced_indent = indent[:-4] if len(indent) >= 4 else indent
    # 3) Re-indent the original play_once to match the placeholder's indentation
    #    (prefix every line in the source with the same indent)
    reindented_play_once = "\n".join(
        (reduced_indent + line) if line.strip() else line
        for line in play_once_src.splitlines()
    )
    result = pattern.sub(reindented_play_once, current_code)
    return result


def generate_code(task_info, las_error=None, message=None):
    """
    Generate code for robot task based on task info and previous errors.
    task_info
    """
    if message is None:
        message = []
        
    # Extract task information
    task_name = task_info['task_name']
    task_description = task_info['task_description']
    current_code = task_info['current_code']
    current_code = generate_current_code(task_name, current_code)

    # Get the enriched actor_list
    original_actor_list = task_info['actor_list']
    # e.g. 把model_name 信息放进去，e.g. displaystand 的功能点
    actor_list = enrich_actors(original_actor_list)

    # print("actor_list: ", actor_list)
    
    available_env_function = str(AVAILABLE_ENV_FUNCTION)
    function_example = str(FUNCTION_EXAMPLE)

    # Generate code based on error status
    if las_error is not None:
        # Handle error case - provide error info to improve generation
        Prompt = (
            f"The code is unsuccessful, \n# Last Error Message: \n{las_error}\n\n"
            f"# Task description: \n{task_description}\n\n"
            f"# Actor List: \n{actor_list}\n\n"
        )
    else:
#         # First attempt case - create initial code file
#         res = f'''
# from envs._base_task import Base_Task
# from envs.{task_name} import {task_name}
# from envs.utils import *
# import sapien

# class gpt_{task_name}({task_name}):
#     def play_once(self):
#         pass
#         '''
        res = generate_gpt_task_code(task_name)

        file_name = f"envs_gen/gpt_{task_name}.py"
        with open(file_name, 'w',encoding='utf-8') as file:
            file.write(res)

        # Construct full prompt with all necessary information
        # Prompt = (
        #     f"{BASIC_INFO}\n\n"
        #     f"# Task description: \n{task_description}\n\n"
        #     f"# Actor List: \n{actor_list}\n\n"
        #     f"# Available API: \n{available_env_function}\n\n"
        #     f"# Function Example: \n{function_example}\n\n"
        #     f"# Code Template:\n{PICK_AND_PLACE_CODE_EXAMPLE}"
        #     f"# Current Json:\n{current_json}"
        # )
        Prompt = (
            f"{BASIC_INFO}\n\n"
            f"# Task description: \n{task_description}\n\n"
            f"# Actor List: \n{actor_list}\n\n"
            f"# Available API: \n{available_env_function}\n\n"
            f"# Function Example: \n{function_example}\n\n"
            f"# Current Code:\n{current_code}"
        )

    # Add prompt to message history
    message.append({"role": "user", "content": Prompt})

    # Generate code using the model
    res = generate(message, gpt="deepseek", temperature=0)
    
    # Extract the relevant portion of the generated code
    res = f'''
from envs._base_task import Base_Task
from envs.{task_name} import {task_name}
from envs.utils import *
import sapien

class gpt_{task_name}({task_name}):
    ''' + res[res.find('def play_once'):res.rfind("```")]
    
    # Save generated code to file
    file_name = f"envs_gen/gpt_{task_name}.py"
    with open(file_name, 'w') as file:
        file.write(res)
    
    print("Task Name: ", task_name)
    print("Task Description: ", task_description)
    
    task, args = setup_task_config(task_name)

    try:
        # Update this section to match the new return values of the run function
        # 仿真环境里运行任务，进行数据采集或策略评测时
        # implicitly call load_actors
        success_rate, error_message, error_count, run_records = run(task, args)
        # sys.exit()
        
        return res, success_rate, error_message, error_count, run_records
    except KeyboardInterrupt:
        print("Test interrupted by user")
        return res, 0, "Test interrupted by user", 20
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error occurred during testing: {e}\n{error_trace}")
        return res, 0, f"Error occurred during testing: {e}", 20,"fail"


def main(task_info_dic):
    """Main function to generate and test code for robot tasks."""
    # Initialize variables
    task_info = now_task_info = task_info_dic
    messages = [{"role": "system", "content": "You need to generate relevant code for some robot tasks in a robot simulation environment based on the provided API."}]
    generate_num = 10
    success_threshold = 0.5
    las_error_message = None
    suc_list = []
    task_name = task_info['task_name']
    task_description = task_info['task_description']
    
    # Store the best code and its success rate
    best_code = None
    best_success_rate = 0
    best_run_records = None
    
    # Create log file name with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "envs_gen/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/{task_info['task_name']}_{timestamp}.log"
    
    # Store all attempt records
    all_attempts = []
    
    # Try multiple generations until success or limit reached
    for id in range(generate_num):
        print(f"Generate code for task: {task_info['task_name']} ({id+1}/{generate_num})")
        
        # Generate and test code
        res_code, success_rate, las_error_message, error_count, run_records = generate_code(
            now_task_info, las_error_message, messages
        )

        # Track success rates
        suc_list.append(success_rate)
        
        # Record this attempt
        attempt_record = {
            "attempt_id": id + 1,
            "success_rate": success_rate,
            "error_message": las_error_message,
            "error_count": error_count,
            "code": res_code,
            "run_records": run_records
        }
        all_attempts.append(attempt_record)
        
        # Save best code
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_code = res_code
            best_run_records = run_records
            print(f"New best code found, success rate: {best_success_rate}")
        
        # Check if generation was successful
        if success_rate >= success_threshold:
            print(f"Successfully generated code for task: {task_info['task_name']}")
            break
            
        # Handle failure case
        print(f"Failed to generate code for task: {task_name} (attempt {id+1})\nError message: \n{las_error_message}")
        
        # Update task description and code for the next attempt
        print(f"Failed to generate code for task: {task_info['task_name']} {id}\nError massage: \n{las_error_message}")
        change_info = """The error may be caused by: 
1. you select the wrong target which not in actor list
1. pre_dis_axis is not set correctly in the place_actor function; 
2. the functional point is not set correctly in the place_actor function; 
3. The pre_dis or dis is not set correctly in the place_actor function;
4. The constrain is not set correctly in the place_actor function, free or align is not constantly fixed, if the code did not have above error, please try to set the constrain to another value.
5. The code didn't take into account the note given in the example function.
The task can be accomplished only through the existing API and example function, please do not use any other API that is not listed in the available API list and examples.\n"""
        now_task_info["task_description"] = f"{task_description}\nFailed to generate code, error message: {las_error_message}, error count: {str(error_count)}\n" + change_info
        now_task_info["current_code"] = res_code
    
    # Ensure the final saved code is the best one
    if best_code is not None:
        task_name = task_info['task_name']
        file_name = f"envs_gen/gpt_{task_name}.py"
        print(f"Saving best code, success rate: {best_success_rate}")
        with open(file_name, 'w',encoding='utf-8') as file:
            file.write(best_code)
    
    print(f"Best success rate: {best_success_rate}")
    print(f"All success rates: {suc_list}")
    
    # Save log data to file
    with open(log_filename, 'w',encoding='utf-8') as log_file:
        log_data = {
            "task_name": task_info['task_name'],
            "task_description": task_info['task_description'],
            "best_success_rate": best_success_rate,
            "success_rates": suc_list,
            "best_code": best_code,
            "best_run_records": best_run_records,
            "all_attempts": all_attempts
        }
        json.dump(log_data, log_file, indent=2)
        
    print(f"Log has been saved to: {log_filename}")
    
    return best_success_rate, suc_list, best_code, best_run_records


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    now_task = None
    
    # Get task information based on task name from task_info_template.py, e.g. PLACE_OBJECT_STAND = {}
    try:
        task_name = parser.parse_args().task_name.upper()
        exec(f'now_task = {task_name}')
    except:
        raise ValueError("The task name is wrong.")

    # Run main function with task information
    main(now_task)



"""
Usage:
python code_gen/task_generation_general_template.py place_object_stand
CUDA_VISIBLE_DEVICES=0 python code_gen/task_generation_template.py place_object_stand
PLACE_BURGER_FRIES PLACE_OBJECT_SCALE MOVE_STAPLER_PAD
place_object_stand 0.9
PLACE_OBJECT_SCALE 0.7
move_stapler_pad 0.7
"""

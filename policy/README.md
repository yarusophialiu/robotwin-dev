# Deploy Your Policy

你需要修改的文件：`eval.sh`, `deploy_policy.yml`, `deploy_polidy.py`

请完善`deploy_policy.py`以及`deploy_policy.yml`
你可以在`deploy_policy.yml`中添加任何你需要的参数（用于指定模型），也可以在`eval.sh`的`[TODO]`后接上可输入的参数，用来覆盖`deploy_policy.yml`中的默认参数。接下来整个`deploy_policy.yml`的内容会作为`usr_args`传入`deploy_policy.py`中的`get_model`，以支持自定义的模型加载。

```
#!/bin/bash

policy_name=Your_Policy
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
# [TODO] add parameters here

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} 
    # [TODO] add parameters here
```

在`deploy_policy.py`中，你需要完善:
1. `encode_obs`: 对环境原观测的操作，比如反转颜色通道等等，这个看你自己，也可以不改
2. `get_model`: 通过usr_args获得模型
3. `eval`: 评测函数，基于一个observation来获得动作并执行
4. `update_obs`, `get_action`: 更新模型的观测窗口以及获得动作，这个看个人实现
5. instruction为语言指令(一个str文本)，怎么用看你自己
6. `reset_model`: 每次评测前都会调用，用于刷新观测窗口等，看自己实现
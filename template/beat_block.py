
from envs._base_task import Base_Task
from envs.beat_block_hammer import beat_block_hammer
from envs.utils import *
import sapien

class gpt_beat_block_hammer(beat_block_hammer):
    def play_once(self):
        block_pose = self.block.get_functional_point(0, "pose").p
       
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        # Move the hammer upwards
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        # Place the hammer on the block's functional point (position 1)
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=self.block.get_functional_point(1, "pose"),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
            ))

        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info
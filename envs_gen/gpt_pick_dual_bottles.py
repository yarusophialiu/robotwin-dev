
from envs._base_task import Base_Task
from envs.pick_dual_bottles import pick_dual_bottles
from envs.utils import *
import sapien

class gpt_pick_dual_bottles(pick_dual_bottles):
    def play_once(self):
        # grasping both bottles
        left_arm = ArmTag("left")
        right_arm = ArmTag("right")
        
        # Grasp bottle1 with left arm
        self.move(
            self.grasp_actor(
                actor=self.bottle1,
                arm_tag=left_arm,
                pre_grasp_dis=0.1,
                grasp_dis=0
            )
        )
        
        # Grasp bottle2 with right arm
        self.move(
            self.grasp_actor(
                actor=self.bottle2,
                arm_tag=right_arm,
                pre_grasp_dis=0.1,
                grasp_dis=0
            )
        )
        
        # placing both bottles at their targets with grippers closed
        # Place bottle1 at left target
        self.move(
            self.place_actor(
                actor=self.bottle1,
                arm_tag=left_arm,
                target_pose=self.left_target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0,
                is_open=False
            )
        )
        
        # Place bottle2 at right target
        self.move(
            self.place_actor(
                actor=self.bottle2,
                arm_tag=right_arm,
                target_pose=self.right_target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0,
                is_open=False
            )
        )

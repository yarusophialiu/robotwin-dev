
from envs._base_task import Base_Task
from envs.pick_dual_bottles import pick_dual_bottles
from envs.utils import *
import sapien

class gpt_pick_dual_bottles(pick_dual_bottles):
    def play_once(self):
        # Get bottle positions to determine which arm to use for each
        bottle1_pose = self.bottle1.get_pose()
        bottle2_pose = self.bottle2.get_pose()
        bottle1_position = bottle1_pose.p
        bottle2_position = bottle2_pose.p
        
        # Assign arms based on bottle positions (left arm for left bottle, right arm for right bottle)
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")
        
        # Simultaneously grasp both bottles
        self.move(
            self.grasp_actor(actor=self.bottle1, arm_tag=left_arm_tag, pre_grasp_dis=0.1, grasp_dis=0),
            self.grasp_actor(actor=self.bottle2, arm_tag=right_arm_tag, pre_grasp_dis=0.1, grasp_dis=0)
        )
        
        # Lift both bottles up to avoid collisions
        self.move(
            self.move_by_displacement(arm_tag=left_arm_tag, z=0.07, move_axis='world'),
            self.move_by_displacement(arm_tag=right_arm_tag, z=0.07, move_axis='world')
        )
        
        # Simultaneously move both bottles to their target locations
        # bottle1 goes to left_target_pose, bottle2 goes to right_target_pose
        self.move(
            self.place_actor(
                actor=self.bottle1,
                arm_tag=left_arm_tag,
                target_pose=self.left_target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0,  # Set to 0 since we don't want to open gripper
                is_open=False,  # Keep gripper closed to maintain hold
                constrain="free",
                pre_dis_axis='fp'
            ),
            self.place_actor(
                actor=self.bottle2,
                arm_tag=right_arm_tag,
                target_pose=self.right_target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0,  # Set to 0 since we don't want to open gripper
                is_open=False,  # Keep gripper closed to maintain hold
                constrain="free",
                pre_dis_axis='fp'
            )
        )

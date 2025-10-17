
from envs._base_task import Base_Task
from envs.beat_block_hammer import beat_block_hammer
from envs.utils import *
import sapien

class gpt_beat_block_hammer(beat_block_hammer):
    def play_once(self):
        # Get the block's position to determine which arm to use
        block_pose = self.block.get_pose()
        block_position = block_pose.p
        
        # Select arm based on block's x coordinate
        if block_position[0] > 0:
            arm_tag = ArmTag("right")
        else:
            arm_tag = ArmTag("left")
        
        # Grasp the hammer
        self.move(
            self.grasp_actor(
                actor=self.hammer,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0
            )
        )
        
        # Lift the hammer to avoid collision
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.07,
                move_axis='world'
            )
        )
        
        # Get the block's functional point for beating (use top functional point)
        target_pose = self.block.get_functional_point(1, "pose")
        
        # Place the hammer on the block's functional point to beat it
        # Use functional point 0 of the hammer (on the hammer head) to align with block's top
        self.move(
            self.place_actor(
                actor=self.hammer,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0.02,
                is_open=False,  # Keep gripper closed to maintain hold of hammer
                constrain="free",
                pre_dis_axis='fp'
            )
        )
        
        # Note: As per task description, we don't need to:
        # - Lift the hammer after beating
        # - Open the gripper after beating  
        # - Return the arm to origin position

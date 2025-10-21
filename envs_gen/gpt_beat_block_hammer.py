
from envs._base_task import Base_Task
from envs.beat_block_hammer import beat_block_hammer
from envs.utils import *
import sapien

class gpt_beat_block_hammer(beat_block_hammer):
    def play_once(self):
        # Determine which arm to use based on block's x-coordinate
        block_pose = self.block.get_pose()
        block_x = block_pose.p[0]  # Get x-coordinate from block's pose
        arm_tag = ArmTag("right" if block_x > 0 else "left")
        
        # Grasp the hammer with selected arm
        self.move(
            self.grasp_actor(
                actor=self.hammer,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0
            )
        )

        # Place hammer on block's bottom functional point
        self.move(
            self.place_actor(
                actor=self.hammer,
                arm_tag=arm_tag,
                target_pose=self.block.get_functional_point(0, "pose"),  # Bottom functional point
                functional_point_id=0,  # Hammer's contact point (head)
                pre_dis=0.1,
                dis=0.02,
                pre_dis_axis="fp",
                constrain="free"  # Default placement strategy
            )
        )


from envs._base_task import Base_Task
from envs.place_shoe import place_shoe
from envs.utils import *
import sapien

class gpt_place_shoe(place_shoe):
    def play_once(self):
        # Initialize actors from the actor_list
        self.shoe = self.actor_list['self.shoe']
        self.target_block = self.actor_list['self.target_block']
        
        # Grasp parameters
        pre_grasp_dis = 0.1
        grasp_dis = 0
        contract_point_id_grasp = [0, 1]  # Use shoe's contact points
        move_by_displacement_z = 0.07  # Lift height
        target_func_point_id = 0  # Target block's bottom functional point
        functional_point_id = 0  # Shoe's bottom functional point
        place_pre_dis = 0.1
        place_dis = 0.02

        # Get shoe's position to determine arm
        object_pose = self.shoe.get_pose().p
        self.arm_tag = ArmTag("right" if object_pose[0] > 0 else "left")

        # Grasp shoe using contact points
        self.move(
            self.grasp_actor(
                actor=self.shoe,
                arm_tag=self.arm_tag,
                contact_point_id=contract_point_id_grasp,
                pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis,
            )
        )

        # Lift the grasped object
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=move_by_displacement_z, move_axis="arm"))

        # Place shoe on target block's bottom
        self.move(
            self.place_actor(
                actor=self.shoe,
                arm_tag=self.arm_tag,
                target_pose=self.target_block.get_functional_point(target_func_point_id),
                functional_point_id=functional_point_id,
                pre_dis=place_pre_dis,
                dis=place_dis,
                constrain="align",  # Ensure full pose alignment
                pre_dis_axis="fp",  # Use functional point direction
            )
        )

        # Clear arms to origin
        self.move(self.back_to_origin(arm_tag=self.arm_tag))

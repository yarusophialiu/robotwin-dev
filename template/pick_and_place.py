
from envs._base_task import Base_Task
from envs.pick_and_place import pick_and_place
from envs.utils import *
import sapien

class gpt_pick_and_place(pick_and_place):    
    def play_once(self):
        PLACEHOLDER= None
        #object to pick and place, you need change PLACEHOLDER to the actual object variable in the task
        object= PLACEHOLDER
        pre_grasp_dis=PLACEHOLDER
        grasp_dis=PLACEHOLDER
        contract_point_id_grasp=PLACEHOLDER
        move_by_displacement_z=PLACEHOLDER
        target_pose=PLACEHOLDER
        functional_point_id=PLACEHOLDER
        place_pre_dis=PLACEHOLDER
        place_dis=PLACEHOLDER
        
        object_pose = object.get_pose().p
        # Select arm based on object's x position (right if positive, left if negative)
        arm_tag = ArmTag("right" if object_pose[0] > 0 else "left")

        # Grasp the object using selected arm with specific contact point
        self.move(
            self.grasp_actor(
                object,
                arm_tag=arm_tag,
                contact_point_id=contract_point_id_grasp,
                pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis,
            ))
        # Lift the object up along z-axis
        self.move(self.move_by_displacement(arm_tag, z=move_by_displacement_z, move_axis="arm"))

        # Place the object onto the functional point
        self.move(
            self.place_actor(
                object,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=functional_point_id,
                pre_dis=place_pre_dis,
                dis=place_dis,
            ))
        # Move the arm up by 0.1m after placing
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))


from envs._base_task import Base_Task
from envs.place_shoe import place_shoe
from envs.utils import *
import sapien

class gpt_place_shoe(place_shoe):
    
    def play_once(self):
        PLACEHOLDER= None
        #object to pick and place, you need change PLACEHOLDER to the actual object variable in the task
        tgt_object=self.shoe
        pre_grasp_dis=0.1
        grasp_dis=0
        contract_point_id_grasp=[0, 1]
        move_by_displacement_z=0.07
        target=self.target_block
        target_func_point_id=1
        functional_point_id=0
        place_pre_dis=0.1
        place_dis=0.02

        # don't change any code below this line,only need to change PLACEHOLDER above
                        
        object_pose = tgt_object.get_pose().p
        target_pose = target.get_functional_point(target_func_point_id)
        # Select arm based on object's x position (right if positive, left if negative)
        self.arm_tag = ArmTag("right" if object_pose[0] > 0 else "left")

        # Grasp the object using selected arm with specific contact point
        self.move(
            self.grasp_actor(
                tgt_object,
                arm_tag=self.arm_tag,
                contact_point_id=contract_point_id_grasp,
                pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis,
            ))
        # Lift the object up along z-axis
        self.move(self.move_by_displacement(self.arm_tag, z=move_by_displacement_z, move_axis="arm"))

        # Place the object onto the functional point
        self.move(
            self.place_actor(
                tgt_object,
                target_pose=target_pose,
                arm_tag=self.arm_tag,
                pre_dis=place_pre_dis,
                dis=place_dis,
                functional_point_id=functional_point_id,
            ))
        # Move the arm up by 0.1m after placing
        self.move(self.move_by_displacement(self.arm_tag, z=0.1, move_axis="arm"))

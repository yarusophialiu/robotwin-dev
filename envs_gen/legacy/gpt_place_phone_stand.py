
from envs._base_task import Base_Task
from envs.place_phone_stand import place_phone_stand
from envs.utils import *
import sapien

class gpt_place_phone_stand(place_phone_stand):
    def play_once(self):
        PLACEHOLDER=None
        object= self.phone
        pre_grasp_dis=0.1
        grasp_dis=0
        contract_point_id_grasp=0
        move_by_displacement_z=0.07
        target= self.stand
        target_func_point_id=0
        functional_point_id=0
        place_pre_dis=0.1
        place_dis=0.02

        object_pose = object.get_pose().p
        target_pose = target.get_functional_point(target_func_point_id)
        self.arm_tag = ArmTag("right" if object_pose[0] > 0 else "left")

        self.move(
            self.grasp_actor(
                object,
                arm_tag=self.arm_tag,
                contact_point_id=contract_point_id_grasp,
                pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis,
            ))
        self.move(self.move_by_displacement(self.arm_tag, z=move_by_displacement_z, move_axis="arm"))

        self.move(
            self.place_actor(
                object,
                arm_tag=self.arm_tag,
                target_pose=target_pose,
                pre_dis=place_pre_dis,
                dis=place_dis,
                functional_point_id=functional_point_id,
            ))
        self.move(self.move_by_displacement(self.arm_tag, z=0.1, move_axis="arm"))

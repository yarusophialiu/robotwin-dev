from ._base_task import Base_Task
from .utils import *
import math
import sapien


class _tmp_test_robot(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)
        self.create_table_and_wall()
        self.load_robot(**kwags)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()
        self.pre_move()
        self.robot.set_origin_endpose()

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0

        self.together_close_gripper(save_freq=None)
        # self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self):
        # self.together_close_gripper()
        pass

    def play_once(self):
        _180, _120, _90, _60, _30 = np.pi, np.pi / 1.5, np.pi / 2, np.pi / 3, np.pi / 6
        limits_dict = {"left": {}, "right": {}}
        init_dict = {"left": {}, "right": {}}
        for joint in self.robot.left_arm_joints:
            joint: sapien.physx.PhysxArticulationJoint = joint
            if limits_dict["left"].get(joint.name):
                joint.set_limits(limits_dict["left"][joint.name])
            if init_dict["left"].get(joint.name):
                joint.set_drive_target(init_dict["left"][joint.name])
        for joint in self.robot.right_arm_joints:
            joint: sapien.physx.PhysxArticulationJoint = joint
            if limits_dict["right"].get(joint.name):
                joint.set_limits(limits_dict["right"][joint.name])
            elif limits_dict["left"].get(joint.name):
                limits = limits_dict["left"][joint.name]
                limits = [-limits[1], -limits[0]]
                joint.set_limits(limits)
            if init_dict["right"].get(joint.name):
                joint.set_drive_target(init_dict["right"][joint.name])
            elif init_dict["left"].get(joint.name):
                joint.set_drive_target(init_dict["left"][joint.name])

        for link in self.robot.left_entity.get_links():
            link: sapien.physx.PhysxArticulationLinkComponent = link
            link.set_mass(1)
        for link in self.robot.right_entity.get_links():
            link: sapien.physx.PhysxArticulationLinkComponent = link
            link.set_mass(1)

        def pause(till_close=False):
            nonlocal self
            self.viewer.paused = True
            while self.viewer.paused or (till_close and not self.viewer.closed):
                if till_close:
                    self.scene.step()
                self.viewer.render()

        def run(step=1):
            while step > 0:
                step -= 1
                self.scene.step()
                self.viewer.render()

        # def find(name, left=True) \
        #     -> sapien.physx.PhysxArticulationJoint:
        #     if left:
        #         for joint in self.robot.left_active_joints:
        #             if joint.get_name() == name:
        #                 return joint
        #     else:
        #         for joint in self.robot.right_active_joints:
        #             if joint.get_name() == name:
        #                 return joint
        #     return None

        # ur5
        # ll_gripper_joints = [{
        #     'joint': find('robotiq_85_left_knuckle_joint'),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_left_inner_knuckle_joint'),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_left_finger_tip_joint'),
        #     'mul': -1.
        # }]
        # lr_gripper_joints = [{
        #     'joint': find('robotiq_85_right_knuckle_joint'),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_right_inner_knuckle_joint'),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_right_finger_tip_joint'),
        #     'mul': -1.
        # }]
        # rl_gripper_joints = [{
        #     'joint': find('robotiq_85_left_knuckle_joint', False),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_left_inner_knuckle_joint', False),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_left_finger_tip_joint', False),
        #     'mul': -1.
        # }]
        # rr_gripper_joints = [{
        #     'joint': find('robotiq_85_right_knuckle_joint', False),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_right_inner_knuckle_joint', False),
        #     'mul': 1.
        # }, {
        #     'joint': find('robotiq_85_right_finger_tip_joint', False),
        #     'mul': -1.
        # }]

        joint_dict = {"left": {}, "right": {}}

        def find(name, left=True) -> sapien.physx.PhysxArticulationJoint:
            if left:
                if joint_dict["left"].get(name) is None:
                    for joint in self.robot.left_active_joints:
                        if joint.get_name() == name:
                            joint_dict["left"][name] = joint
                            break
                return joint_dict["left"].get(name)
            else:
                if joint_dict["right"].get(name) is None:
                    for joint in self.robot.right_active_joints:
                        if joint.get_name() == name:
                            joint_dict["right"][name] = joint
                            break
                return joint_dict["right"].get(name)

        left_gripper = [{
            "base":
            "l_gripper_finger1_joint",
            "mimic": [
                ("l_gripper_finger2_joint", 1.0, 0.0),
                ("l_gripper_finger1_finger_joint", 0.4563942, 0.0),
                ("l_gripper_finger2_finger_joint", 0.4563942, 0.0),
                ("l_gripper_finger1_inner_knuckle_joint", 1.49462955, 0.0),
                ("l_gripper_finger2_inner_knuckle_joint", 1.49462955, 0.0),
                ("l_gripper_finger1_finger_tip_joint", 1.49462955, 0.0),
                ("l_gripper_finger2_finger_tip_joint", 1.49462955, 0.0),
            ],
        }]
        right_gripper = [{
            "base":
            "r_gripper_finger1_joint",
            "mimic": [
                ("r_gripper_finger2_joint", 1.0, 0.0),
                ("r_gripper_finger1_finger_joint", 0.4563942, 0.0),
                ("r_gripper_finger2_finger_joint", 0.4563942, 0.0),
                ("r_gripper_finger1_inner_knuckle_joint", 1.49462955, 0.0),
                ("r_gripper_finger2_inner_knuckle_joint", 1.49462955, 0.0),
                ("r_gripper_finger1_finger_tip_joint", 1.49462955, 0.0),
                ("r_gripper_finger2_finger_tip_joint", 1.49462955, 0.0),
            ],
        }]

        def left_set_pos(pos):
            for group in left_gripper:
                find(group["base"]).set_drive_target(pos)
                for mimic in group["mimic"]:
                    find(mimic[0]).set_drive_target(mimic[1] * pos + mimic[2])

        def right_set_pos(pos):
            for group in right_gripper:
                find(group["base"], False).set_drive_target(pos)
                for mimic in group["mimic"]:
                    find(mimic[0], False).set_drive_target(mimic[1] * pos + mimic[2])

        pause()
        self.together_close_gripper()
        self.together_open_gripper()
        self.robot.left_original_pose = self.robot.get_left_ee_pose()
        self.right_original_pose = self.robot.get_right_ee_pose()
        print("left_original_pose ", self.robot.left_original_pose)
        print("right_original_pose ", self.right_original_pose)
        tag = True
        num = 1
        self.left_move_to_pose([
            -0.21300586168239985,
            -0.12699768572656814,
            1.1214351467031964,
            0.8408430736957326,
            -0.20097299224652249,
            0.11846535909524447,
            0.4884247541841289,
        ])
        self.together_close_gripper()
        self.together_open_gripper()
        while 1:
            for _ in range(4):  # render every 4 steps
                qf = self.robot.left_entity.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                self.robot.left_entity.set_qf(qf)
                qf = self.robot.right_entity.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                self.robot.right_entity.set_qf(qf)
            if tag:
                print(
                    "left global: ",
                    self.robot.left_ee_name,
                    self.robot.left_ee.global_pose,
                )
                print(
                    "right global: ",
                    self.robot.right_ee_name,
                    self.robot.right_ee.global_pose,
                )
                print(self.robot.get_left_ee_pose())
                tag = False
            self.scene.step()  # run a physical step
            self.scene.update_render()  # sync pose from SAPIEN to renderer
            self.viewer.render()

    def check_success(self):
        return True

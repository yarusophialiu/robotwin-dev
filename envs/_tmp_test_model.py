from ._base_task import Base_Task
from .utils import *
import sapien


class _tmp_test_model(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)
        self.create_table_and_wall()
        self.load_robot(**kwags)
        self.load_camera(**kwags)
        self.load_actors()
        self.robot.move_to_homestate()
        self.pre_move()
        self.robot.set_origin_endpose()

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        self.bowl1, self.bowl1_data = rand_create_actor(
            self.scene,
            xlim=[0.25, 0.25],
            ylim=[0.0, 0],
            zlim=[0.9],
            modelname="test",
            scale=(1, 1, 1),
            convex=True,
        )
        self.bowl2, self.bowl2_data = rand_create_actor(
            self.scene,
            xlim=[-0.25, -0.25],
            ylim=[0.0, 0],
            zlim=[0.9],
            modelname="test",
            scale=(1, 1, 1),
            convex=True,
        )
        # self.bowl1, self.bowl1_data = rand_create_actor(
        #     self.scene,
        #     xlim=[0.25,0.25],
        #     ylim=[0.0,0],
        #     zlim=[0.9],
        #     modelname="002_container_test",
        #     qpos=[0.5,0.5,0.5,0.5],
        #     model_id = 4,
        #     scale=(1,1,1),
        #     convex=True
        # )
        # self.bowl2, self.bowl2_data = rand_create_actor(
        #     self.scene,
        #     xlim=[-0.25,-0.25],
        #     ylim=[0.0,0],
        #     zlim=[0.9],
        #     modelname="002_container_test",
        #     qpos=[0.5,0.5,0.5,0.5],
        #     model_id = 4,
        #     scale=(1,1,1),
        #     convex=True
        # )
        self.bowl1.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.bowl2.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

    def play_once(self):
        while 1:
            self.close_left_gripper()

    def check_success(self):
        container_pose = self.get_actor_goal_pose(self.container, self.container_data)
        target_pose = np.array([0, -0.05, 0.74])
        eps = np.array([0.02, 0.02, 0.01])
        # left_gripper = self.robot.get_left_gripper_real_val()
        # right_gripper = self.robot.get_right_gripper_real_val()
        # endpose_z = max(self.robot.get_right_endpose()[2], self.robot.get_left_endpose()[2])
        endpose_z = max(self.robot.get_right_ee_pose()[2], self.robot.get_left_ee_pose()[2])
        # return np.all(abs(container_pose - target_pose) < eps) and left_gripper > 0.04 and right_gripper > 0.04 and endpose_z > 0.98 and self.is_left_gripper_open() and self.is_right_gripper_open()
        return (np.all(abs(container_pose - target_pose) < eps) and self.is_left_gripper_open()
                and self.is_right_gripper_open() and endpose_z > 0.98 and self.is_left_gripper_open()
                and self.is_right_gripper_open())

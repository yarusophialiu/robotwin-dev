from ._base_task import Base_Task
from .utils import *
import sapien
import math


class _test_random(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)
        self.create_table_and_wall()
        # self.load_robot(**kwags)
        # self.load_camera(**kwags)
        # self.robot.move_to_homestate()
        # self.pre_move()
        # self.robot.set_origin_endpose()
        if is_test:
            self.id_list = [0, 1]
        else:
            self.id_list = [0, 1]
        self.load_actors()
        if self.cluttered_table:
            self.get_cluttered_table()
        self.step_lim = 400

        self.right_p = []
        self.left_p = []

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        pass

    def play_once(self):
        pause(self)

        info = dict()
        info["cluttered_table_info"] = self.record_messy_objects
        info["texture_info"] = {
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
        }
        info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            "{Arm}": self.lr_tag,
        }
        return info

    def check_success(self, target=0.4):
        return True

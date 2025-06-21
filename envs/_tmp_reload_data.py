from ._base_task import Base_Task
from .utils import *
import math
import sapien
from .place_shoe import place_shoe
import pickle


class _tmp_reload_data(place_shoe):

    def play_once(self):

        def count_files_in_directory(directory_path):
            try:
                items = os.listdir(directory_path)
                file_count = sum(1 for item in items if os.path.isfile(os.path.join(directory_path, item)))
                return file_count
            except FileNotFoundError:
                print(f"目录 {directory_path} 不存在")
                return None
            except PermissionError:
                print(f"没有权限访问目录 {directory_path}")
                return None

        def read_pkl(path):
            with open(path, "rb") as file:
                data = pickle.load(file)
            return data

        def get_obs_shoufa(task_name, head_camera_type):
            folder_path = (f"data/dual_shoes_place_aloha-agilex_D435_mt1_rt1_pkl/episode0/")
            obs_list, action_list = [], []
            obs_num = count_files_in_directory(folder_path)
            for i in range(obs_num):
                file_path = folder_path + f"{i}.pkl"
                obs = read_pkl(file_path)
                obs_list.append(obs)
                action_list.append(obs["joint_action"])
            return obs_list, action_list

        task_name = "place_shoe"
        head_camera_type = "D435"
        gt_obs, gt_actions = get_obs_shoufa(task_name, head_camera_type)
        gt_actions = np.array(gt_actions)

        success_flag = True
        actions = gt_actions
        cnt = 0

        action_chunk = 15

        for action_id in range(0, gt_actions.shape[0], action_chunk):
            actions = gt_actions[action_id:action_id + action_chunk]
            left_arm_actions, left_gripper, left_current_qpos, left_path = (
                [],
                [],
                [],
                [],
            )
            right_arm_actions, right_gripper, right_current_qpos, right_path = (
                [],
                [],
                [],
                [],
            )
            real_observation = self.get_obs()
            real_obs = dict()
            real_obs["agent_pos"] = real_observation["joint_action"]
            if self.dual_arm:
                left_arm_actions, left_gripper = actions[:, :6], actions[:, 6]
                right_arm_actions, right_gripper = actions[:, 7:13], actions[:, 13]
                left_current_qpos, right_current_qpos = (
                    real_obs["agent_pos"][:6],
                    real_obs["agent_pos"][7:13],
                )
            else:
                right_arm_actions, right_gripper = actions[:, :6], actions[:, 6]
                right_current_qpos = real_obs["agent_pos"][:6]

            if self.dual_arm:
                left_path = np.vstack((left_current_qpos, left_arm_actions))
            right_path = np.vstack((right_current_qpos, right_arm_actions))

            topp_left_flag, topp_right_flag = True, True
            try:
                times, left_pos, left_vel, acc, duration = self.robot.left_planner.TOPP(left_path,
                                                                                        1 / 250,
                                                                                        verbose=True)
                left_result = dict()
                left_result["position"], left_result["velocity"] = left_pos, left_vel
                left_n_step = left_result["position"].shape[0]
                left_gripper = np.linspace(left_gripper[0], left_gripper[-1], left_n_step)
            except:
                topp_left_flag = False
                left_n_step = 1

            if left_n_step == 0 or (not self.dual_arm):
                topp_left_flag = False
                left_n_step = 1

            try:
                times, right_pos, right_vel, acc, duration = (self.robot.right_planner.TOPP(right_path,
                                                                                            1 / 250,
                                                                                            verbose=True))
                right_result = dict()
                right_result["position"], right_result["velocity"] = (
                    right_pos,
                    right_vel,
                )
                right_n_step = right_result["position"].shape[0]
                right_gripper = np.linspace(right_gripper[0], right_gripper[-1], right_n_step)
            except:
                topp_right_flag = False
                right_n_step = 1

            if right_n_step == 0:
                topp_right_flag = False
                right_n_step = 1

            cnt += actions.shape[0]

            n_step = max(left_n_step, right_n_step)

            obs_update_freq = n_step // actions.shape[0]

            now_left_id = 0 if topp_left_flag else 1e9
            now_right_id = 0 if topp_right_flag else 1e9
            i = 0

            while now_left_id < left_n_step or now_right_id < right_n_step:
                qf = self.robot.entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                self.robot.entity.set_qf(qf)
                if (topp_left_flag and now_left_id < left_n_step
                        and now_left_id / left_n_step <= now_right_id / right_n_step):
                    self.robot.set_arm_joints(
                        left_result["position"][now_left_id],
                        left_result["velocity"][now_left_id],
                        "left",
                    )
                    if not self.fix_gripper:
                        self.robot.set_gripper(left_gripper[now_left_id], "left")
                    now_left_id += 1

                if (topp_right_flag and now_right_id < right_n_step
                        and now_right_id / right_n_step <= now_left_id / left_n_step):
                    self.robot.set_arm_joints(
                        right_result["position"][now_right_id],
                        right_result["velocity"][now_right_id],
                        "right",
                    )
                    if not self.fix_gripper:
                        self.robot.set_gripper(right_gripper[now_right_id], "right")
                    now_right_id += 1

                self.scene.step()
                self._update_render()

                if i != 0 and i % 100 == 0:
                    observation = self.get_obs()
                    obs = self.get_cam_obs(observation)
                    obs["agent_pos"] = observation["joint_action"]

                if self.render_freq and i % self.render_freq == 0:
                    self._update_render()
                    self.viewer.render()

                i += 1

            self._update_render()
            if self.render_freq:
                self.viewer.render()

            print(f"step: {cnt} / {self.step_lim}", end="\r")

        if success_flag:
            print("\nsuccess!")
            self.suc += 1
        else:
            print("\nfail!")

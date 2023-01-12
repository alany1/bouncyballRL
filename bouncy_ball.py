# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

class BouncyBall(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.x_bound = (self.cfg["env"]["xlow"], self.cfg["env"]["xhigh"])
        self.y_bound = (self.cfg["env"]["ylow"], self.cfg["env"]["yhigh"])
        
        self.max_episode_length = 5000
        self.cfg["env"]["numObservations"] = 6
        
        self.goal_x = self.cfg["env"]["xgoal"]
        self.goal_y = self.cfg["env"]["ygoal"]

        # How many actions per env? 2 for this. 1 for Cartpole.
        self.cfg["env"]["numActions"] = 2
        
        # This calls create_sim. Environments are all created which allows us to acquire dof state tensors.
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # (num_dofs) x 2 (position, velocity)

        # DOFs are specified in the URDF file joint tags by the type argument. 
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(root_state_tensor)
        
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rigid_body_tensor)
        self.num_bodies = 3
        self.rb_positions = self.rb_states[:self.num_envs*self.num_bodies, 0:3].view(self.num_envs, self.num_bodies, 3)
        
        # Can split the tensor for easier viewing
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0
    
    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        # Base sim stuff
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        self._create_ground_plane()

        # envSpacing specifies the distance between environments; more details in _create_envs.
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))
            
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized

        # These are local bounds for the environment? They will be merged together as if they are these rectangles, although they can extend out if they want.
        # This is fine because of collision groups; they will not interfere. 
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/bouncy_ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()

        # For bouncy ball, the base link is the ball. 
        asset_options.fix_base_link = False
        ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ball_asset)
        
        # (1) Ball to pole 1, (2) Ball to pole 2.

        # Pose is interpreted relative to the local coordinates. 
        self.ball_handles = []
        self.envs = []

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.fix_base_link = True       
        box_asset = self.gym.create_box(self.sim, 4, 4, 4, box_asset_options)

        for i in range(self.num_envs):
            
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            pose = gymapi.Transform()

            if self.up_axis == 'z':
                pose.p.z = 2
                pose.p.y = 20*np.random.rand()-10
                pose.p.x = 20*np.random.rand()-10
            else:
                pose.p.y = 2.0
                pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

            ball_handle = self.gym.create_actor(env_ptr, ball_asset, pose, "ball", 0, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, ball_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0

            c = 0.5 + 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])
            self.gym.set_actor_dof_properties(env_ptr, ball_handle, dof_props)
            self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            self.envs.append(env_ptr)
            self.ball_handles.append(ball_handle)

        goal_pose = gymapi.Transform()
        goal_pose.p.x = self.goal_x
        goal_pose.p.y = self.goal_y
        goal_pose.p.z = 10
        env_ptr = self.gym.create_env(self.sim, lower, upper, 1)
        goal_handle = self.gym.create_actor(env_ptr, box_asset, goal_pose, "goal", 0, 1, 0)

        c = 0.5 + 0.5 * np.random.random(3)
        color = gymapi.Vec3(c[0], c[1], c[2])
        self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    def compute_reward(self):
        # retrieve environment observations from buffer
        x = self.obs_buf[:, 0]
        y = self.obs_buf[:, 1]
        z = self.obs_buf[:, 2]

        vx = self.obs_buf[:, 3]
        vy = self.obs_buf[:, 4]
        vz = self.obs_buf[:, 5]

        # progress_buf stores how many iterations we have gone through. This is so that we can cap the max episode length.
        self.rew_buf[:], self.reset_buf[:] = compute_bouncyball_reward(
            x, y, z, vx, vy, vz,
            self.x_bound, self.y_bound, self.reset_buf, self.progress_buf, self.max_episode_length, (self.goal_x, self.goal_y)
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        # You specify the number of observations you want in your buffer. Then, in this method you compute these observations for each 
        # physics step I guess. 

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        without_box = self.root_state[:self.num_envs]
        positions = without_box[env_ids, :3]
        velocities = without_box[env_ids, 7:10]
        self.obs_buf[env_ids, :3] = positions # root XYZ positions
        self.obs_buf[env_ids, 3:] = velocities # root XYZ velocities

        return self.obs_buf

    def reset_idx(self, env_ids):   
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Mark as reset.
        self.reset_buf[env_ids] = 0

        # Reset the total time steps.
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # Convert action to force

        # In Cartpole, we only alter the first degree of freedom (slider to cart). We cannot touch cart to pole joint.

        # Don't move DOFS in order to make the sphere stay non-rotating.

        # First: bounce if needed
        states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[:self.num_envs*self.num_bodies].view(self.num_envs, self.num_bodies, 13)
        idxs = np.where(np.array(states[:,0,2].cpu()) < .51)[0] 
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        forces[idxs, 0, 2] = 200
        force_positions = self.rb_positions.clone()

        # Second: Convert the forces from each pole to applying WASD movement of the ball. We don't really care about the actual kinematics of the DOFs. 
        actions_view = actions.view(self.num_envs, self.cfg["env"]["numActions"]) # possible bug here in layout ??
        
        # pole1: (+) --> forward; (-) --> backward
        # pole2: (+) --> right; (-) --> left

        # Apply force from 0 to 100 N scaled by the amount. 
        forces[idxs, 0, 0] = 100*actions[idxs, 0]
        forces[idxs, 0, 1] = 100*actions[idxs, 1]
        
        force_positions = force_positions.view(self.num_envs*3, 3)
        forces = forces.view(self.num_envs*3, 3)
        base = torch.zeros((1,3), device=self.device, dtype=torch.float32)

        forces = torch.cat([forces, base])
        force_positions = torch.cat([force_positions, base])
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1

        # Check for any envs needing reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_bouncyball_reward(x, y, z, vx, vy, vz,
                            x_bound, y_bound, reset_buf, progress_buf, max_episode_length, goal):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[int, int], Tuple[int, int], Tensor, Tensor, float, Tuple[int, int]) -> Tuple[Tensor, Tensor]

    # reward is just negative MSE from the GOAL
    # One per environment
    GOAL_X = goal[0]
    GOAL_Y = goal[1]
    reward = -((x - GOAL_X)**2 + (y - GOAL_Y)**2) + 10

    # # adjust reward for reset agents
    # # For agents needing reset, just give them reward -5. If all good, give the computed reward. 
    reward = torch.where(x < x_bound[0], torch.ones_like(reward) * -200.0, reward)
    reward = torch.where(x > x_bound[1], torch.ones_like(reward) * -200.0, reward)

    reward = torch.where(y < y_bound[0], torch.ones_like(reward) * -200.0, reward)
    reward = torch.where(y > y_bound[1], torch.ones_like(reward) * -200.0, reward)
    reward = torch.where(progress_buf > max_episode_length - 1, torch.ones_like(reward) * -200.0, reward)

    # # Mark the ones needing reset as 1. Otherwise keep the same. 
    # # Either past the limits or exceed maximum episode length.

    reset = torch.where(x < x_bound[0], torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(x > x_bound[1], torch.ones_like(reset_buf), reset)
    reset = torch.where(y < y_bound[0], torch.ones_like(reset_buf), reset)
    reset = torch.where(y > y_bound[1], torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf > max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset

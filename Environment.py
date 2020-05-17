from __future__ import division
import numpy as np
import time
import random
import math

# This file is revised for more precise and concise expression.
class D2Dchannels:
    # Simulator of the D2D Channels
    def __init__(self, n_Dev, n_RB):
        self.t = 0
        self.fc = 2     # carrier frequency
        self.n_Dev = n_Dev     # number of devices
        self.n_RB = n_RB    # number of resource blocks

    def update_positions(self, positions):
        self.positions = positions      # (x,y) coordinates, an array of x y coordinates

    # path loss, or path attenuation, is the reduction in power density of an electromagnetic wave as it propagates through space
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))

        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(self.n_Dev, self.n_Dev, self.n_RB) ) + 1j * np.random.normal(size=(self.n_Dev, self.n_Dev, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * self.fc * (10**9)/(3*10**8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 + 2.7 * np.log10(self.fc/5)

        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)

        if min(d1,d2) < 7:
            PL = PL_Los(d)
            self.ifLOS = True
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False

        return PL


class cellular_channels:
    # Simulator of the cellular channels
    def __init__(self, n_Dev, n_RB):
        self.h_bs = 25
        self.Decorrelation_distance = 50
        self.BS_position = [750/2, 1299/2]    # Suppose the BS is in the center
        self.shadow_std = 8     # shadow standard
        self.n_Dev = n_Dev      # number of devices
        self.n_RB = n_RB    # number of resource blocks
        self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1,d2)    # change from meters to kilometers
            self.PathLoss[i] = 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs)**2)/1000)

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:     # initialization
            self.Shadow = np.random.normal(0, self.shadow_std, self.n_Dev)
        else:
            delta_distance = np.asarray(delta_distance_list)
            self.Shadow = np.exp(-1*(delta_distance/self.Decorrelation_distance))* self.Shadow +\
                          np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*np.random.normal(0,self.shadow_std, self.n_Dev)

    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size = (self.n_Dev, self.n_RB)) + 1j* np.random.normal(size = (self.n_Dev, self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))


class Device:
    # Device simulator: includes all the information of a device
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:
    # Enviroment Simulator: Provides states and rewards to agents
    # Evolves to new state based on the actions taken by the d2d agent
    def __init__ (self, down_lanes, up_lanes, left_lanes, right_lanes, width, height):
        self.timestep = 0.01
        self.down_lanes = down_lanes
        self.up_lanes = up_lanes
        self.left_lanes = left_lanes
        self.right_lanes = right_lanes
        self.width = width
        self.height = height
        self.devices = []
        self.demands = []
        self.D2D_power_dB = 23           # dBm
        self.cellular_power_dB = 23      # dBm
        self.D2D_power_dB_List = [23, 10, 5]   # the power levels
        self.sig2_dB = -114
        self.bsAntGain = 8      # base station antenna gain
        self.bsNoiseFigure = 5  # base station noise figure
        self.devAntGain = 3     # device antenna gain
        self.devNoiseFigure = 9     # device noise figure
        self.sig2 = 10**(self.sig2_dB/10)
        self.D2D_Shadowing = []
        self.cellular_Shadowing = []
        self.delta_distance = []
        self.n_RB = 20      # number of resource blocks
        self.n_Dev = 40     # number of devices
        self.D2Dchannels = D2Dchannels(self.n_Dev, self.n_RB)
        self.cellular_channels = cellular_channels(self.n_Dev, self.n_RB)

        self.D2D_Interference_all = np.zeros((self.n_Dev, 3, self.n_RB)) + self.sig2
        self.n_step = 0

    def add_new_devices(self, start_position, start_direction, start_velocity):
        self.devices.append(Device(start_position, start_direction, start_velocity))

    def add_new_devices_by_number(self, n):
        for i in range(n):
            ind = np.random.randint(0,len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0,self.height)]
            start_direction = 'd'
            self.add_new_devices(start_position,start_direction,random.randint(10,15))
            start_position = [self.up_lanes[ind], random.randint(0,self.height)]
            start_direction = 'u'
            self.add_new_devices(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_devices(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_devices(start_position,start_direction,random.randint(10,15))

        self.D2D_Shadowing = np.random.normal(0, 3, [len(self.devices), len(self.devices)])
        self.cellular_Shadowing = np.random.normal(0, 8, len(self.devices))
        self.delta_distance = np.asarray([c.velocity for c in self.devices])
        #self.renew_channel()

    def renew_positions(self):
        # ===========================================================
        # This function update the position of each device
        # ===========================================================
        i = 0

        #for i in range(len(self.position)):
        while(i < len(self.devices)):
            #print ('start iteration ', i)
            #print(self.position, len(self.position), self.direction)
            delta_distance = self.devices[i].velocity * self.timestep
            change_direction = False

            if self.devices[i].direction == 'u':
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.devices[i].position[1] <= self.left_lanes[j]) and ((self.devices[i].position[1] + delta_distance) >= self.left_lanes[j]):   # came to a cross
                        if (random.uniform(0,1) < 0.4):
                            self.devices[i].position = [self.devices[i].position[0] - (delta_distance - (self.left_lanes[j] - self.devices[i].position[1])),self.left_lanes[j]]
                            self.devices[i].direction = 'l'
                            change_direction = True
                            break

                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.devices[i].position[1] <=self.right_lanes[j]) and ((self.devices[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.devices[i].position = [self.devices[i].position[0] + (delta_distance + (self.right_lanes[j] - self.devices[i].position[1])), self.right_lanes[j]]
                                self.devices[i].direction = 'r'
                                change_direction = True
                                break

                if change_direction == False:
                    self.devices[i].position[1] += delta_distance

            if (self.devices[i].direction == 'd') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.devices[i].position[1] >=self.left_lanes[j]) and ((self.devices[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.devices[i].position = [self.devices[i].position[0] - (delta_distance - (self.devices[i].position[1]- self.left_lanes[j])), self.left_lanes[j]]
                            #print ('down with left', self.devices[i].position)
                            self.devices[i].direction = 'l'
                            change_direction = True
                            break

                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.devices[i].position[1] >=self.right_lanes[j]) and (self.devices[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.devices[i].position = [self.devices[i].position[0] + (delta_distance + (self.devices[i].position[1]- self.right_lanes[j])),self.right_lanes[j]]
                                #print ('down with right', self.devices[i].position)
                                self.devices[i].direction = 'r'
                                change_direction = True
                                break

                if change_direction == False:
                    self.devices[i].position[1] -= delta_distance

            if (self.devices[i].direction == 'r') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.devices[i].position[0] <= self.up_lanes[j]) and ((self.devices[i].position[0] + delta_distance) >= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.devices[i].position = [self.up_lanes[j], self.devices[i].position[1] + (delta_distance - (self.up_lanes[j] - self.devices[i].position[0]))]
                            change_direction = True
                            self.devices[i].direction = 'u'
                            break

                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.devices[i].position[0] <= self.down_lanes[j]) and ((self.devices[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.devices[i].position = [self.down_lanes[j], self.devices[i].position[1] - (delta_distance - (self.down_lanes[j] - self.devices[i].position[0]))]
                                change_direction = True
                                self.devices[i].direction = 'd'
                                break

                if change_direction == False:
                    self.devices[i].position[0] += delta_distance

            if (self.devices[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    if (self.devices[i].position[0] >= self.up_lanes[j]) and ((self.devices[i].position[0] - delta_distance) <= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.devices[i].position = [self.up_lanes[j], self.devices[i].position[1] + (delta_distance - (self.devices[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.devices[i].direction = 'u'
                            break

                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.devices[i].position[0] >= self.down_lanes[j]) and ((self.devices[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.devices[i].position = [self.down_lanes[j], self.devices[i].position[1] - (delta_distance - (self.devices[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.devices[i].direction = 'd'
                                break

                    if change_direction == False:
                        self.devices[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.devices[i].position[0] < 0) or (self.devices[i].position[1] < 0) or (self.devices[i].position[0] > self.width) or (self.devices[i].position[1] > self.height):
            # delete
                # print ('delete ', self.position[i])
                if (self.devices[i].direction == 'u'):
                    self.devices[i].direction = 'r'
                    self.devices[i].position = [self.devices[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.devices[i].direction == 'd'):
                        self.devices[i].direction = 'l'
                        self.devices[i].position = [self.devices[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.devices[i].direction == 'l'):
                            self.devices[i].direction = 'u'
                            self.devices[i].position = [self.up_lanes[0],self.devices[i].position[1]]
                        else:
                            if (self.devices[i].direction == 'r'):
                                self.devices[i].direction = 'd'
                                self.devices[i].position = [self.down_lanes[-1],self.devices[i].position[1]]

            i += 1

    def test_channel(self):
        # ===================================
        #   test the D2D channel
        # ===================================
        self.n_step = 0
        self.devices = []
        n_Dev = 20
        self.n_Dev = n_Dev
        self.add_new_devices_by_number(int(self.n_Dev/4))
        step = 1000
        time_step = 0.1  # every 0.1s update
        for i in range(step):
            self.renew_positions()
            positions = [c.position for c in self.devices]
            self.update_large_fading(positions, time_step)
            self.update_small_fading()
            print("Time step: ", i)
            print(" ============== Cellular ===========")
            print("Path Loss: ", self.cellular_channels.PathLoss)
            print("Shadow:",  self.cellular_channels.Shadow)
            print("Fast Fading: ",  self.cellular_channels.FastFading)
            print(" ============== D2D ===========")
            print("Path Loss: ", self.D2Dchannels.PathLoss[0:3])
            print("Fast Fading: ", self.D2Dchannels.FastFading[0:3])

    def update_large_fading(self, positions, time_step):
        self.cellular_channels.update_positions(positions)
        self.D2Dchannels.update_positions(positions)
        self.cellular_channels.update_pathloss()
        self.D2Dchannels.update_pathloss()
        delta_distance = time_step * np.asarray([c.velocity for c in self.devices])
        self.cellular_channels.update_shadow(delta_distance)

    def update_small_fading(self):
        self.cellular_channels.update_fast_fading()
        self.D2Dchannels.update_fast_fading()

    def renew_neighbor(self):
        # ===========================================
        # update the neighbors of each device
        # ===========================================
        for i in range(len(self.devices)):
            self.devices[i].neighbors = []
            self.devices[i].actions = []
            #print('action and neighbors delete', self.vehicles[i].actions, self.vehicles[i].neighbors)
        Distance = np.zeros((len(self.devices),len(self.devices)))
        z = np.array([[complex(c.position[0],c.position[1]) for c in self.devices]])
        Distance = abs(z.T-z)
        for i in range(len(self.devices)):
            sort_idx = np.argsort(Distance[:,i])
            for j in range(3):
                self.devices[i].neighbors.append(sort_idx[j+1])
            destination = np.random.choice(sort_idx[1:int(len(sort_idx)/5)],3, replace = False)
            self.devices[i].destinations = destination

    def renew_channel(self):
        # =============================================================================
        # This function updates all the D2D channels
        # =============================================================================
        positions = [c.position for c in self.devices]
        self.cellular_channels.update_positions(positions)
        self.D2Dchannels.update_positions(positions)
        self.cellular_channels.update_pathloss()
        self.D2Dchannels.update_pathloss()
        delta_distance = 0.002 * np.asarray([c.velocity for c in self.devices])    # time slot is 2 ms
        self.cellular_channels.update_shadow(delta_distance)
        self.D2D_channels_abs = self.D2Dchannels.PathLoss + 50 * np.identity(len(self.devices))
        self.cellular_channels_abs = self.cellular_channels.PathLoss + self.cellular_channels.Shadow

    def renew_channels_fastfading(self):
        # =========================================================================
        # This function updates all the D2D channels
        # =========================================================================
        self.renew_channel()
        self.cellular_channels.update_fast_fading()
        self.D2Dchannels.update_fast_fading()
        D2D_channels_with_fastfading = np.repeat(self.D2D_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.D2D_channels_with_fastfading = D2D_channels_with_fastfading - self.D2Dchannels.FastFading
        cellular_channels_with_fastfading = np.repeat(self.cellular_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.cellular_channels_with_fastfading = cellular_channels_with_fastfading - self.cellular_channels.FastFading
        # print("V2I channels", self.V2I_channels_with_fastfading)

    # not considering interference from one D2D transmitter to another D2D reciever
    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):   # revising based on the fast fading part
        actions = actions_power.copy()[:,:,0]  # the channel_selection_part
        power_selection = actions_power.copy()[:,:,1]
        Rate = np.zeros(len(self.devices))
        Interference = np.zeros(self.n_RB)  # D2D signal interference to cellular links
        for i in range(len(self.devices)):
            for j in range(len(actions[i,:])):
                if not self.activate_links[i,j]:
                    continue
                # print('power selection,', power_selection[i,j])
                Interference[actions[i][j]] += 10**((self.D2D_power_dB_List[power_selection[i,j]]  - self.cellular_channels_with_fastfading[i, actions[i,j]] + self.devAntGain + self.bsAntGain - self.bsNoiseFigure)/10)  # fast fading

        self.cellular_Interference = Interference + self.sig2
        D2D_Interference = np.zeros((len(self.devices), 3))
        D2D_Signal = np.zeros((len(self.devices), 3))

        # remove the effects of none active links
        #print('shapes', actions.shape, self.activate_links.shape)
        #print(not self.activate_links)
        actions[(np.logical_not(self.activate_links))] = -1
        #print('action are', actions)
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.devices[indexes[j,0]].destinations[indexes[j,1]]
                # compute the D2D signal links
                D2D_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.D2D_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] - self.D2D_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.devAntGain - self.devNoiseFigure)/10)

                if i < self.n_Dev:
                    D2D_Interference[indexes[j,0],indexes[j,1]] += 10**((self.cellular_power_dB - self.D2D_channels_with_fastfading[i][receiver_j][i]+ 2*self.devAntGain - self.devNoiseFigure )/10)  # cellular links interference to D2D links
                for k in range(j+1, len(indexes)):                  # computer the peer D2D links
                    #receiver_k = self.vehicles[indexes[k][0]].neighbors[indexes[k][1]]
                    receiver_k = self.devices[indexes[k][0]].destinations[indexes[k][1]]
                    D2D_Interference[indexes[j,0],indexes[j,1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - self.D2D_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.devAntGain - self.devNoiseFigure)/10)
                    D2D_Interference[indexes[k,0],indexes[k,1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - self.D2D_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.devAntGain - self.devNoiseFigure)/10)

        self.D2D_Interference = D2D_Interference + self.sig2
        D2D_Rate = np.zeros(self.activate_links.shape)
        D2D_Rate[self.activate_links] = np.log2(1 + np.divide(D2D_Signal[self.activate_links], self.D2D_Interference[self.activate_links]))

        #print("V2V Rate", V2V_Rate * self.update_time_test * 1500)
        #print ('V2V_Signal is ', np.log(np.mean(V2V_Signal[self.activate_links])))
        cellular_Signals = self.cellular_power_dB-self.cellular_channels_abs[0:min(self.n_RB,self.n_Dev)] + self.devAntGain + self.bsAntGain - self.bsNoiseFigure
        cellular_Rate = np.log2(1 + np.divide(10**(cellular_Signals/10), self.cellular_Interference[0:min(self.n_RB,self.n_Dev)]))


         # --- compute the latency constraits ---
        self.demand -= D2D_Rate * self.update_time_test * 1500    # decrease the demand
        self.test_time_count -= self.update_time_test               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_test         # compute the time left for individual D2D transmission
        self.individual_time_interval -= self.update_time_test      # compute the time interval left for next transmission

        # --- update the demand ---
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape ) + self.D2D_limit
        self.individual_time_limit[new_active] = self.D2D_limit
        self.demand[new_active] = self.demand_amount
        #print("demand is", self.demand)
        #print('mean rate of average D2D link is', np.mean(D2D_Rate[self.activate_links]))

        # --- update the statistics ---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False
        #print('number of activate links is', np.sum(self.activate_links))
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        #if self.n_step % 1000 == 0 :
        #    self.success_transmission = 0
        #    self.failed_transmission = 0
        failed_percentage = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)
        # print('Percentage of failed', np.sum(new_active), self.failed_transmission, self.failed_transmission + self.success_transmission , failed_percentage)
        return cellular_Rate, failed_percentage  # failed percentage

    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):   # revising based on the fast fading part
        # ===================================================
        #  --------- Used for Testing --------
        # ===================================================
        actions = actions_power[:,:,0]  # the channel_selection_part
        power_selection = actions_power[:,:,1]
        Interference = np.zeros(self.n_RB)   # Calculate the interference from D2D to channels
        for i in range(len(self.devices)):
            for j in range(len(actions[i,:])):
                if not self.activate_links[i,j]:
                    continue
                Interference[actions[i][j]] += 10**((self.D2D_power_dB_List[power_selection[i,j]] - \
                                                     self.cellular_channels_with_fastfading[i, actions[i,j]] + \
                                                     self.devAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
        self.cellular_Interference = Interference + self.sig2
        D2D_Interference = np.zeros((len(self.devices), 3))
        D2D_Signal = np.zeros((len(self.devices), 3))
        Interfence_times = np.zeros((len(self.devices), 3))
        actions[(np.logical_not(self.activate_links))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.devices[indexes[j,0]].destinations[indexes[j,1]]
                D2D_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.D2D_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.D2D_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.devAntGain - self.devNoiseFigure)/10)
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10)
                if i<self.n_Dev:
                    D2D_Interference[indexes[j,0],indexes[j,1]] += 10**((self.cellular_power_dB - \
                    self.D2D_channels_with_fastfading[i][receiver_j][i] + 2*self.devAntGain - self.devNoiseFigure )/10)  # cellular links interference to D2D links
                for k in range(j+1, len(indexes)):
                    receiver_k = self.devices[indexes[k][0]].destinations[indexes[k][1]]
                    D2D_Interference[indexes[j,0],indexes[j,1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] -\
                    self.D2D_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.devAntGain - self.devNoiseFigure)/10)
                    D2D_Interference[indexes[k,0],indexes[k,1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.D2D_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.devAntGain - self.devNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1

        self.D2D_Interference = D2D_Interference + self.sig2
        D2D_Rate = np.log2(1 + np.divide(D2D_Signal, self.D2D_Interference))
        cellular_Signals = self.cellular_power_dB-self.cellular_channels_abs[0:min(self.n_RB,self.n_Dev)] + self.devAntGain + self.bsAntGain - self.bsNoiseFigure
        cellular_Rate = np.log2(1 + np.divide(10**(cellular_Signals/10), self.cellular_Interference[0:min(self.n_RB,self.n_Dev)]))
        #print("Cellular information", cellular_Signals, self.cellular_Interference, cellular_Rate)

        # -- compute the latency constraits --
        self.demand -= D2D_Rate * self.update_time_asyn * 1500    # decrease the demand
        self.test_time_count -= self.update_time_asyn               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_asyn         # compute the time left for individual D2D transmission
        self.individual_time_interval -= self.update_time_asyn     # compute the time interval left for next transmission

        # --- update the demand ---
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape) + self.D2D_limit
        self.individual_time_limit[new_active] = self.D2D_limit
        self.demand[new_active] = self.demand_amount

        # --- update the statistics ---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        fail_percent = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)
        return cellular_Rate, fail_percent

    def Compute_Performance_Reward_Batch(self, actions_power, idx):    # add the power dimension to the action selection
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        actions = actions_power.copy()[:,:,0]           #
        power_selection = actions_power.copy()[:,:,1]   #
        D2D_Interference = np.zeros((len(self.devices), 3))
        D2D_Signal = np.zeros((len(self.devices), 3))
        Interfence_times = np.zeros((len(self.devices), 3))    #  3 neighbors
        #print(actions)
        origin_channel_selection = actions[idx[0], idx[1]]
        actions[idx[0], idx[1]] = 100  # something not relavant
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            #print('index',indexes)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.devices[indexes[j,0]].destinations[indexes[j,1]]
                D2D_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.D2D_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.D2D_channels_with_fastfading[indexes[j,0], receiver_j, i]+ 2*self.devAntGain - self.devNoiseFigure)/10)
                D2D_Interference[indexes[j,0],indexes[j,1]] +=  10**((self.cellular_power_dB- self.D2D_channels_with_fastfading[i,receiver_j,i] + \
                2*self.devAntGain - self.devNoiseFigure)/10)  # interference from the cellular links

                for k in range(j+1, len(indexes)):
                    receiver_k = self.devices[indexes[k,0]].destinations[indexes[k,1]]
                    D2D_Interference[indexes[j,0],indexes[j,1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - \
                    self.D2D_channels_with_fastfading[indexes[k,0],receiver_j,i] + 2*self.devAntGain - self.devNoiseFigure)/10)
                    D2D_Interference[indexes[k,0],indexes[k,1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.D2D_channels_with_fastfading[indexes[j,0], receiver_k, i] + 2*self.devAntGain - self.devNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1

        self.D2D_Interference = D2D_Interference + self.sig2
        D2D_Rate_list = np.zeros((self.n_RB, len(self.D2D_power_dB_List)))  # the number of RB times the power level
        Deficit_list = np.zeros((self.n_RB, len(self.D2D_power_dB_List)))
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            D2D_Signal_temp = D2D_Signal.copy()
            #receiver_k = self.vehicles[idx[0]].neighbors[idx[1]]
            receiver_k = self.devices[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.D2D_power_dB_List)):
                D2D_Interference_temp = D2D_Interference.copy()
                D2D_Signal_temp[idx[0],idx[1]] = 10**((self.D2D_power_dB_List[power_idx] - \
                self.D2D_channels_with_fastfading[idx[0], self.devices[idx[0]].destinations[idx[1]],i] + 2*self.devAntGain - self.devNoiseFigure )/10)
                D2D_Interference_temp[idx[0],idx[1]] +=  10**((self.cellular_power_dB - \
                self.D2D_channels_with_fastfading[i,self.devices[idx[0]].destinations[idx[1]],i] + 2*self.devAntGain - self.devNoiseFigure)/10)
                for j in range(len(indexes)):
                    receiver_j = self.devices[indexes[j,0]].destinations[indexes[j,1]]
                    D2D_Interference_temp[idx[0],idx[1]] += 10**((self.D2D_power_dB_List[power_selection[indexes[j,0], indexes[j,1]]] -\
                    self.D2D_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2*self.devAntGain - self.devNoiseFigure)/10)
                    D2D_Interference_temp[indexes[j,0],indexes[j,1]] += 10**((self.D2D_power_dB_List[power_idx]-\
                    self.D2D_channels_with_fastfading[idx[0],receiver_j, i] + 2*self.devAntGain - self.devNoiseFigure)/10)
                D2D_Rate_cur = np.log2(1 + np.divide(D2D_Signal_temp, D2D_Interference_temp))
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    D2D_Rate = D2D_Rate_cur.copy()
                D2D_Rate_list[i, power_idx] = np.sum(D2D_Rate_cur)
                Deficit_list[i,power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(D2D_Signal_temp.shape), (self.demand - self.individual_time_limit * D2D_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)
        cellular_Rate_list = np.zeros((self.n_RB,len(self.D2D_power_dB_List)))    # 3 of power level
        for i in range(len(self.devices)):
            for j in range(len(actions[i,:])):
                if (i ==idx[0] and j == idx[1]):
                    continue
                Interference[actions[i][j]] += 10**((self.D2D_power_dB_List[power_selection[i,j]] - \
                self.cellular_channels_with_fastfading[i, actions[i][j]] + self.devAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
        cellular_Interference = Interference + self.sig2
        for i in range(self.n_RB):
            for j in range(len(self.D2D_power_dB_List)):
                cellular_Interference_temp = cellular_Interference.copy()
                cellular_Interference_temp[i] += 10**((self.D2D_power_dB_List[j] - self.cellular_channels_with_fastfading[idx[0], i] + self.devAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
                cellular_Rate_list[i, j] = np.sum(np.log2(1 + np.divide(10**((self.cellular_power_dB + self.devAntGain + self.bsAntGain \
                - self.bsNoiseFigure-self.cellular_channels_abs[0:min(self.n_RB,self.n_Dev)])/10), cellular_Interference_temp[0:min(self.n_RB,self.n_Dev)])))

        self.demand -= D2D_Rate * self.update_time_train * 1500
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train
        self.individual_time_limit [np.add(self.individual_time_limit <= 0,  self.demand < 0)] = self.D2D_limit
        self.demand[self.demand < 0] = self.demand_amount
        if self.test_time_count == 0:
            self.test_time_count = 10
        return cellular_Rate_list, Deficit_list, self.individual_time_limit[idx[0], idx[1]]

    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        D2D_Interference = np.zeros((len(self.devices), 3, self.n_RB)) + self.sig2
        if len(actions.shape) == 3:
            channel_selection = actions.copy()[:,:,0]
            power_selection = actions[:,:,1]
            channel_selection[np.logical_not(self.activate_links)] = -1
            for i in range(self.n_RB):
                for k in range(len(self.devices)):
                    for m in range(len(channel_selection[k,:])):
                        D2D_Interference[k, m, i] += 10 ** ((self.cellular_power_dB - self.D2D_channels_with_fastfading[i][self.devices[k].destinations[m]][i] + \
                        2 * self.devAntGain - self.devNoiseFigure)/10)
            for i in range(len(self.devices)):
                for j in range(len(channel_selection[i,:])):
                    for k in range(len(self.devices)):
                        for m in range(len(channel_selection[k,:])):
                            if (i==k) or (channel_selection[i,j] >= 0):
                                continue
                            D2D_Interference[k, m, channel_selection[i,j]] += 10**((self.D2D_power_dB_List[power_selection[i,j]] -\
                            self.D2D_channels_with_fastfading[i][self.devices[k].destinations[m]][channel_selection[i,j]] + 2*self.devAntGain - self.devNoiseFigure)/10)

        self.D2D_Interference_all = 10 * np.log10(D2D_Interference)

    def renew_demand(self):
        # generate a new demand of a D2D
        self.demand = self.demand_amount*np.ones((self.n_RB,3))
        self.time_limit = 10

    def act_for_training(self, actions, idx):
        # =============================================
        # This function gives rewards for training
        # ===========================================
        rewards_list = np.zeros(self.n_RB)
        action_temp = actions.copy()
        self.activate_links = np.ones((self.n_Dev,3), dtype = 'bool')
        cellular_rewardlist, D2D_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp,idx)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        rewards_list = rewards_list.T.reshape([-1])
        cellular_rewardlist = cellular_rewardlist.T.reshape([-1])
        D2D_rewardlist = D2D_rewardlist.T.reshape([-1])
        cellular_reward = (cellular_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                      np.min(cellular_rewardlist))/(np.max(cellular_rewardlist) -np.min(cellular_rewardlist) + 0.000001)
        D2D_reward = (D2D_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                     np.min(D2D_rewardlist))/(np.max(D2D_rewardlist) -np.min(D2D_rewardlist) + 0.000001)
        lambdda = 0.1
        #print ("Reward", V2I_reward, V2V_reward, time_left)
        t = lambdda * cellular_reward + (1-lambdda) * D2D_reward
        #print("time left", time_left)
        #return t
        return t - (self.D2D_limit - time_left)/self.D2D_limit

    def act_asyn(self, actions):
        self.n_step += 1
        if self.n_step % 10 == 0:
            self.renew_positions()
            self.renew_channels_fastfading()
        reward = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.Compute_Interference(actions)
        return reward

    def act(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1
        reward = self.Compute_Performance_Reward_fast_fading_with_power(actions)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward

    def new_random_game(self, n_Dev = 0):
        # make a new game
        self.n_step = 0
        self.devices = []
        if n_Dev > 0:
            self.n_Dev = n_Dev
        self.add_new_devices_by_number(int(self.n_Dev/4))
        self.D2Dchannels = D2Dchannels(self.n_Dev, self.n_RB)  # number of devices
        self.cellular_channels = cellular_channels(self.n_Dev, self.n_RB)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Dev,3))
        self.test_time_count = 10
        self.D2D_limit = 0.1  # 100 ms D2D toleratable latency
        self.individual_time_limit = self.D2D_limit * np.ones((self.n_Dev,3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Dev,3))
        self.UnsuccessfulLink = np.zeros((self.n_Dev,3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.002 # 2ms update time for testing
        self.update_time_asyn = 0.0002 # 0.2 ms update one subset of the devices; for each device, the update time is 2 ms
        self.activate_links = np.zeros((self.n_Dev,3), dtype='bool')


if __name__ == "__main__":
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height)
    Env.test_channel()

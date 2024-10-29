clear all;
clc;

load("cross_road_map.mat");

load("test_left_turns_fixed_froms_big.mat");
load("test_right_turns_fixed_froms_big.mat");
load("test_straight_lines_fixed_froms_big.mat");

n_init_states = 200;
n_trajs_per_mode = 100;


trajs = {};
c = 1;


for i=1:n_init_states
    for m=1:3
        for j=1:n_trajs_per_mode
            if m==1
                trajs{c} = list_left{j};
                c=c+1;
            elseif m==2
                trajs{c} = list_straight{j};
                c=c+1;
            else
                trajs{c} = list_right{j};
                c=c+1;
            end
        end
    end
end


save("test_crossroad_trajs_fixed_froms_big","trajs");

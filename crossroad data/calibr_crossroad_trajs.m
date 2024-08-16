clear all;
clc;

load("cross_road_map.mat");

load("calibr_left_turns_big.mat");
load("calibr_right_turns_big.mat");
load("calibr_straight_lines_big.mat");

n_init_states = 40;
n_trajs = 200;

trajs = {};
c = 1;

figure
fig = show(crossroad);

for i=1:n_init_states*n_trajs

    trajs{c} = list_left{i};
    c=c+1;
    hold on
    plot(list_left{i}(:,1),list_left{i}(:,2), 'g')

end

for i=1:n_init_states*n_trajs

    trajs{c} = list_straight{i};
    c=c+1;
    hold on
    plot(list_straight{i}(:,1),list_straight{i}(:,2), 'y')

end

for i=1:n_init_states*n_trajs
    
    trajs{c} = list_right{i};
    c=c+1;
    hold on
    plot(list_right{i}(:,1),list_right{i}(:,2), 'r')

end

save("calibr_crossroad_trajs_big","trajs");
saveas(fig,'plots/trajs_big');
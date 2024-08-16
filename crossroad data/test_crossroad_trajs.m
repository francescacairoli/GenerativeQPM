clear all;
clc;

load("cross_road_map.mat");

load("test_left_turns.mat");
load("test_right_turns.mat");
load("test_straight_lines.mat");

n_init_states = 50;
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

save("test_crossroad_trajs","trajs");
saveas(fig,'plots/test_trajs');
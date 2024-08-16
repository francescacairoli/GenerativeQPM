clear all;
clc;

load("cross_road_map.mat");

load("left_turns.mat");
load("right_turns.mat");
load("straight_lines.mat");

nbTrajs = 300;

trajs = {};
c = 1;

figure
%fig = show(crossroad);
fig = figure;
for i=1:nbTrajs

    trajs{c} = list_left{i};
    c=c+1;
    hold on
    plot(list_left{i}(:,1),list_left{i}(:,2), 'g')

end

for i=1:nbTrajs

    trajs{c} = list_straight{i};
    c=c+1;
    hold on
    plot(list_straight{i}(:,1),list_straight{i}(:,2), 'y')

end

for i=1:nbTrajs
    
    trajs{c} = list_right{i};
    c=c+1;
    hold on
    plot(list_right{i}(:,1),list_right{i}(:,2), 'r')

end

save("dataset_trajs","trajs");
saveas(fig,'plots/trajs_no_bg');
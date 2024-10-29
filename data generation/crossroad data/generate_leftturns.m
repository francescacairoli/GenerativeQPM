clear all;
clc;

nbTrajs = 1000;

n = 51;
w = 15;

load("cross_road_map.mat");
% generate the intersection
M = zeros(n);
M(1:w,:) = 1;
M(n-w+1:n,1:w) = 1;
M(fix(n/2)+1:n,1:fix(n/2)+1) = 1;
M(:,n-w+1:n) = 1;

ML = logical(M);

map = binaryOccupancyMap(ML);
%show(map)

prm = mobileRobotPRM(map,50);
prm.ConnectionDistance = 6;

flb = [27, 1];
fub = [35, 5];

tlb = [1 27];
tub = [5 35];

col = ['b','g','r','y','k'];
figure
fig = show(crossroad);
list_left = {};
i=1;
while i<=nbTrajs

    update(prm)
    
    from = unifrnd(flb, fub, 1, 2);
    to = unifrnd(tlb, tub, 1, 2);
    
    path = findpath(prm,from,to);
    if length(path)>0
        list_left{i} = path;
        i=i+1;
        hold on
        %plot(path(:,1),path(:,2), col(mod(i,5)+1))
        plot(path(:,1),path(:,2), 'g')
    end
end
save("left_turns_big", "list_left");
saveas(fig,'plots/left_trajs');
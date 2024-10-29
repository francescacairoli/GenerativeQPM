clear all;
clc;

n_init_states = 200;
n_trajs = 300;

n = 51;
w = 15;

load("cross_road_map.mat");
% generate the intersection
M = zeros(n);

M(:,1:fix(n/2)+1) = 1;
M(:,n-w+1:n) = 1;

ML = logical(M);

map = binaryOccupancyMap(ML);
%show(map)

prm = mobileRobotPRM(map,50);
prm.ConnectionDistance = 6;
%show(prm)

flb = [27, 1];
fub = [35, 5];

tlb = [27 45];
tub = [35 50];

col = ['b','g','r','y','k'];
figure
fig = show(crossroad);
list_straight = {};
c=1;
for i=1:n_init_states

   i
    
    from = unifrnd(flb, fub, 1, 2);
    j=1;
    while j<=n_trajs
        to = unifrnd(tlb, tub, 1, 2);
        update(prm)
        path = findpath(prm,from,to);
        if length(path) > 0
            list_straight{c} = path;
            c=c+1;
            j=j+1;
            hold on
            %plot(path(:,1),ath(:,2), col(mod(i,5)+1))
            plot(path(:,1),path(:,2), 'y')
        end
    end
end
save("calibr_straight_lines_bigger", "list_straight");
saveas(fig,'plots/calibr_straight_trajs_big');
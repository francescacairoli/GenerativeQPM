clear all;
clc;

n_init_states = 40;
n_trajs = 200;

n = 51;
w = 15;

load("cross_road_map.mat");
% generate the intersection
M = zeros(n);
M(1:fix(n/2)+1,:) = 1;
M(n-w+1:n,n-w+1:n) = 1;
M(w:n,1:fix(n/2)+1) = 1;

ML = logical(M);

map = binaryOccupancyMap(ML);


prm = mobileRobotPRM(map,500);
prm.ConnectionDistance = 2;
show(prm)

flb = [27, 1];
fub = [35, 5];

tlb = [45 15];
tub = [50 25];

col = ['b','g','r','y','k'];
figure
fig = show(crossroad);
list_right = {};
c=1;
for i=1:n_init_states

    
    
    from = unifrnd(flb, fub, 1, 2);
    j=1;
    while j<=n_trajs
        to = unifrnd(tlb, tub, 1, 2);
        update(prm)
        path = findpath(prm,from,to);
        if length(path) > 0
            list_right{c} = path;
            c=c+1;
            j=j+1;
            hold on
            %plot(path(:,1),path(:,2), col(mod(i,5)+1))
            plot(path(:,1),path(:,2), 'r')
        end
    end
end
save("calibr_right_turns_big", "list_right");
saveas(fig,'plots/calibr_right_trajs_big');
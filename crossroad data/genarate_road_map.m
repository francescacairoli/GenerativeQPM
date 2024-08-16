n = 51;
w = 15;

% generate the intersection
M = zeros(n);
M(1:w,1:w) = 1;
M(n-w+1:n,1:w) = 1;
M(1:w,n-w+1:n) = 1;
M(n-w+1:n,n-w+1:n) = 1;

% generate the lanes
M(1:w,fix(n/2)+1) = 1;
M(n-w+1:n,fix(n/2)+1) = 1;
M(fix(n/2)+1,1:w) = 1;
M(fix(n/2)+1,n-w+1:n) = 1;
ML = logical(M);

crossroad = binaryOccupancyMap(ML);
show(crossroad)
save('cross_road_map', "crossroad")

prm = mobileRobotPRM(crossroad,200);
%show(prm)

from = [32 3];
to = [3 32];

for i=1:5
    update(prm)
    path = findpath(prm,from,to)
    fig=show(prm);
    saveas(fig,['plots/traj_',num2str(i)])
end
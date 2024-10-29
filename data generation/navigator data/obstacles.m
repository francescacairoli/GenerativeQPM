n = 30;


% generate the intersection
M = zeros(n);

M(5:10,5:10) = 1;

M(14:20,10:16) = 1;
M(8:12,18:22) = 1;
M(20:25,20:25) = 1;
ML = logical(M);

map = binaryOccupancyMap(ML);

show(map)
%save('obstacle_map', "map")
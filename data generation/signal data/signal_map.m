n = 25;


% generate the intersection
M = zeros(n);

M(5:10, 15:25) = 1;
M(7:9, 7:15) = 1;
M(15:20, 15:25) = 1;
M(17:19, 7:15) = 1;
ML = logical(M);

map = binaryOccupancyMap(ML);

show(map)
save('signal_map', "map")

clear all;
clc;

load obstacle_map.mat;

show(map)


n_init_states = 600;
n_trajs = 300;
prm = mobileRobotPRM(map,120);
%show(prm)
prm.ConnectionDistance = 6;

flb = [1, 1];
fub = [7, 7];

tlb = [29, 29];
tub = [29, 29];


col = ['b','g','r','y','k'];
figure
fig = show(map);
list_trajs = {};


c=1;
from_list = zeros(n_init_states,2);
for i=1:n_init_states

    i

    prm = mobileRobotPRM(map,120);
    prm.ConnectionDistance = 6;
   
    
    from =unifrnd(flb, fub, 1, 2);
    
    
    
    from_list(i,:) = from;
    
    j=0;
    %for j=1:n_trajs
    while(j<n_trajs)
        to = unifrnd(tlb, tub, 1, 2);

        path = findpath(prm,from,to);
        
        update(prm);
        if length(path) > 0
            list_trajs{c} = path;
            c=c+1;
            j=j+1;
            hold on
            %plot(path(:,1),path(:,2), col(mod(i,5)+1))
            plot(path(:,1),path(:,2),  'b-', 'LineWidth', 2)
            plot(from(1), from(2), 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g');
            plot(to(1), to(2), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    
        end
    end
end

%save("train_froms", "from_list");
save("navigator_calibr_trajs_big", "list_trajs");
%saveas(fig,'plots/calibr_trajs_navigator');




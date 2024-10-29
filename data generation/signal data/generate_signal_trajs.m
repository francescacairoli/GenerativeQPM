
clear all;
clc;

load signal_map.mat;

show(map)


n_init_states = 600;
n_trajs = 300;
prm = mobileRobotPRM(map,100);
%show(prm)
prm.ConnectionDistance = 4;

to_xlim = [25, 25];
to_ylim = [1, 25];

from_xlim = [1, 1];
from_ylim = [5, 20];


col = ['b','g','r','y','k'];
figure
fig = show(map);
list_trajs = {};


c=1;
from_list = zeros(n_init_states,2);
for i=1:n_init_states

    i
   
    
    from = sampleRandomFreePoint(map, from_xlim, from_ylim);
    
    
    
    from_list(i,:) = from;
    j=0;
    while(j<n_trajs)   
        to = sampleRandomFreePoint(map, to_xlim, to_ylim);

        path = findpath(prm,from,to);
        
        update(prm);

        
        if length(path) > 0
            list_trajs{c} = path;
            c=c+1;
            j=j+1;
            hold on;
            %plot(path(:,1),path(:,2), col(mod(i,5)+1))
            plot(path(:,1),path(:,2),  'b-', 'LineWidth', 2)
            plot(from(1), from(2), 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g');
            plot(to(1), to(2), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    
        end
    end
end

%save("train_froms", "from_list");
save("signal_calibr_trajs", "list_trajs");
%saveas(fig,'plots/calibr_trajs_signal');




function point = sampleRandomFreePoint(map, xLimits, yLimits)
    isValid = false;
    while ~isValid
        point = [rand*(xLimits(2)-xLimits(1)) + xLimits(1), ...
                 rand*(yLimits(2)-yLimits(1)) + yLimits(1)]; % Generate random point within the map dimensions
        if getOccupancy(map, point) == 0
            isValid = true;
        end
    end
end

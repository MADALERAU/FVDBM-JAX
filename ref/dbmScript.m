%% Cavity with Moving Lid DBM Simulation

% Lattice D2Q9
%% Define Vars
rho_ref = 8; %reference density
Mu = .1; %kinematic Viscosity
Re = 100; %reynolds number
N_x = 100; % number of cells in the x dir
N_y = N_x; % number of cells in the y dir
dx = 1; % spacing in x
dy = 1; % spacing in y
U_lid = Re*Mu/(N_x); %velocity of lid

%% DBM Stuff
Tau = 3*Mu+.5;
c_s = 1/sqrt(3);
Ksi = [0 1 0 -1 0 1 -1 -1 1;...
       0 0 1 0 -1 1 1 -1 -1]; % velocity vectors in each dir
w = [4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36]; % weights in each dir
n = [0,-1,0,1;...
     1,0,-1,0]; %NWSE

f_cc = ones(N_y,N_x,9); % PDF @ center cell
f_surfvert = ones(N_y, N_x+1, 9);   % PDF @ Vertical Surfaces(Flux Calculation)
f_surfhorz = ones(N_y+1, N_x, 9);   % PDF @ Horizontal Surfaces(Flux Calculation)
u = zeros(N_y, N_x);
v = zeros(N_y, N_x);
rho = ones(N_y, N_x)*rho_ref;

% Memory Alloc. For Ghost Cell Boundary Method
f_ghost = ones(1,9);
f_nodes = ones(2,9);

%% Sim Start
iter_count = 10000;

for timer = 1:iter_count
    fprintf('iteration: %d/%d\n',timer,iter_count)
    %% Finding Surface PDFs
    % vertical surfaces
    for j = 1:N_y
        for i = 1:N_x+1
            if i == 1 % left boundary
                
                % Defining Local Env. cuz I'm lazy :)
                f_middle = squeeze(f_cc(j,i,:))-w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
                rho_middle = rho(j,i);
                if j == 1   % top corner
                    rho_top = rho(j,i);
                    rho_bot = rho(j+1,i);
                    f_top = squeeze(f_cc(j,i,:))-w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
                    f_bot = squeeze(f_cc(j+1,i,:))-w'.*rho(j+1,i).*(1+(Ksi'*[u(j+1,i);v(j+1,i)])/(c_s^2)+(Ksi'*[u(j+1,i);v(j+1,i)]).^2/(2*c_s^4)-([u(j+1,i),v(j+1,i)]*[u(j+1,i);v(j+1,i)])/(2*c_s^2));
                elseif j==N_y   %bottom corner
                    rho_top = rho(j-1,i);
                    rho_bot = rho(j,i);
                    f_top = squeeze(f_cc(j-1,i,:))-w'.*rho(j-1,i).*(1+(Ksi'*[u(j-1,i);v(j-1,i)])/(c_s^2)+(Ksi'*[u(j-1,i);v(j-1,i)]).^2/(2*c_s^4)-([u(j-1,i),v(j-1,i)]*[u(j-1,i);v(j-1,i)])/(2*c_s^2));
                    f_bot = squeeze(f_cc(j,i,:))-w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
                else
                    rho_top = rho(j-1,i);
                    rho_bot = rho(j+1,i);
                    f_top = squeeze(f_cc(j-1,i,:))-w'.*rho(j-1,i).*(1+(Ksi'*[u(j-1,i);v(j-1,i)])/(c_s^2)+(Ksi'*[u(j-1,i);v(j-1,i)]).^2/(2*c_s^4)-([u(j-1,i),v(j-1,i)]*[u(j-1,i);v(j-1,i)])/(2*c_s^2));
                    f_bot = squeeze(f_cc(j+1,i,:))-w'.*rho(j+1,i).*(1+(Ksi'*[u(j+1,i);v(j+1,i)])/(c_s^2)+(Ksi'*[u(j+1,i);v(j+1,i)]).^2/(2*c_s^4)-([u(j+1,i),v(j+1,i)]*[u(j+1,i);v(j+1,i)])/(2*c_s^2));
                end
                
                % Node 1
                rho_node = .5*(rho_middle+rho_top);
                u_node = [0,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_top);
                f_nodes(1,:) = f_eq_node+squeeze(f_neq_node);
                % Node 2
                rho_node = .5*(rho_middle+rho_bot);
                u_node = [0,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_bot);
                f_nodes(2,:) = f_eq_node+squeeze(f_neq_node);

                % temp_surface value
                f_surfvert(j,i,:) = mean(f_nodes,1);

                
                % defining ghost cell
                for k = 1:9
                    f_ghost(k) = 2*f_surfvert(j,i,k)-f_cc(j,i,k);
                end

                % avg cells
                f_surfvert(j,i,1) = (f_ghost(1)+f_cc(j,i,1))/2;
                f_surfvert(j,i,3) = (f_ghost(3)+f_cc(j,i,3))/2;
                f_surfvert(j,i,5) = (f_ghost(5)+f_cc(j,i,5))/2;
                
                % left cell
                f_surfvert(j,i,2) = f_ghost(2);
                f_surfvert(j,i,6) = f_ghost(6);
                f_surfvert(j,i,9) = f_ghost(9);
    
                % right cell
                f_surfvert(j,i,4) = f_cc(j,i,4);
                f_surfvert(j,i,7) = f_cc(j,i,7);
                f_surfvert(j,i,8) = f_cc(j,i,8);
                
                
                %{
                Old Method
                f_surfvert(j,i,1) = f_cc(j,i,1);
                f_surfvert(j,i,3) = f_cc(j,i,3);
                f_surfvert(j,i,5) = f_cc(j,i,5);
                f_surfvert(j,i,7) = f_cc(j,i,7);
                f_surfvert(j,i,8) = f_cc(j,i,8);
    
                % bounce-back
                f_surfvert(j,i,4) = f_cc(j,i,4);
                f_surfvert(j,i,2) = f_cc(j,i,4);
    
                % no-slip
                f_surfvert(j,i,6) = f_surfvert(j,i,8)+(f_surfvert(j,i,5)-f_surfvert(j,i,3))/2;
                f_surfvert(j,i,9) = f_surfvert(j,i,8)+f_surfvert(j,i,7)-f_surfvert(j,i,6);
                %}
            elseif i == N_x+1 % right boundary
                
                % Defining Local Env. cuz I'm lazy :)
                f_middle = squeeze(f_cc(j,i-1,:))-w'.*rho(j,i-1).*(1+(Ksi'*[u(j,i-1);v(j,i-1)])/(c_s^2)+(Ksi'*[u(j,i-1);v(j,i-1)]).^2/(2*c_s^4)-([u(j,i-1),v(j,i-1)]*[u(j,i-1);v(j,i-1)])/(2*c_s^2));
                rho_middle = rho(j,i-1);
                if j == 1   % top corner
                    rho_top = rho(j,i-1);
                    rho_bot = rho(j+1,i-1);
                    f_top = squeeze(f_cc(j,i-1,:))-w'.*rho(j,i-1).*(1+(Ksi'*[u(j,i-1);v(j,i-1)])/(c_s^2)+(Ksi'*[u(j,i-1);v(j,i-1)]).^2/(2*c_s^4)-([u(j,i-1),v(j,i-1)]*[u(j,i-1);v(j,i-1)])/(2*c_s^2));
                    f_bot = squeeze(f_cc(j+1,i-1,:))-w'.*rho(j+1,i-1).*(1+(Ksi'*[u(j+1,i-1);v(j+1,i-1)])/(c_s^2)+(Ksi'*[u(j+1,i-1);v(j+1,i-1)]).^2/(2*c_s^4)-([u(j+1,i-1),v(j+1,i-1)]*[u(j+1,i-1);v(j+1,i-1)])/(2*c_s^2));
                elseif j==N_y   %bottom corner
                    rho_top = rho(j-1,i-1);
                    rho_bot = rho(j,i-1);
                    f_top = squeeze(f_cc(j-1,i-1,:))-w'.*rho(j-1,i-1).*(1+(Ksi'*[u(j-1,i-1);v(j-1,i-1)])/(c_s^2)+(Ksi'*[u(j-1,i-1);v(j-1,i-1)]).^2/(2*c_s^4)-([u(j-1,i-1),v(j-1,i-1)]*[u(j-1,i-1);v(j-1,i-1)])/(2*c_s^2));
                    f_bot = squeeze(f_cc(j,i-1,:))-w'.*rho(j,i-1).*(1+(Ksi'*[u(j,i-1);v(j,i-1)])/(c_s^2)+(Ksi'*[u(j,i-1);v(j,i-1)]).^2/(2*c_s^4)-([u(j,i-1),v(j,i-1)]*[u(j,i-1);v(j,i-1)])/(2*c_s^2));
                else
                    rho_top = rho(j-1,i-1);
                    rho_bot = rho(j+1,i-1);
                    f_top = squeeze(f_cc(j-1,i-1,:))-w'.*rho(j-1,i-1).*(1+(Ksi'*[u(j-1,i-1);v(j-1,i-1)])/(c_s^2)+(Ksi'*[u(j-1,i-1);v(j-1,i-1)]).^2/(2*c_s^4)-([u(j-1,i-1),v(j-1,i-1)]*[u(j-1,i-1);v(j-1,i-1)])/(2*c_s^2));
                    f_bot = squeeze(f_cc(j+1,i-1,:))-w'.*rho(j+1,i-1).*(1+(Ksi'*[u(j+1,i-1);v(j+1,i-1)])/(c_s^2)+(Ksi'*[u(j+1,i-1);v(j+1,i-1)]).^2/(2*c_s^4)-([u(j+1,i-1),v(j+1,i-1)]*[u(j+1,i-1);v(j+1,i-1)])/(2*c_s^2));
                end

                % Node 1
                rho_node = .5*(rho_middle+rho_top);
                u_node = [0,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_top);
                f_nodes(1,:) = f_eq_node+squeeze(f_neq_node);
                % Node 2
                rho_node = .5*(rho_middle+rho_bot);
                u_node = [0,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_bot);
                f_nodes(2,:) = f_eq_node+squeeze(f_neq_node);

                % temp_surface value
                f_surfvert(j,i,:) = mean(f_nodes,1);
                
                
                % defining ghost cell
                for k = 1:9
                    f_ghost(k) = 2*f_surfvert(j,i,k)-f_cc(j,i-1,k);
                end
                % final surface value
                f_surfvert(j,i,1) = (f_cc(j,i-1,1)+f_ghost(1))/2;
                f_surfvert(j,i,3) = (f_cc(j,i-1,3)+f_ghost(2))/2;
                f_surfvert(j,i,5) = (f_cc(j,i-1,5)+f_ghost(3))/2;
                
                % left cell
                f_surfvert(j,i,2) = f_cc(j,i-1,2);
                f_surfvert(j,i,6) = f_cc(j,i-1,6);
                f_surfvert(j,i,9) = f_cc(j,i-1,9);
    
                % right cell
                f_surfvert(j,i,4) = f_ghost(4);
                f_surfvert(j,i,7) = f_ghost(7);
                f_surfvert(j,i,8) = f_ghost(8);
                


                %{
                Old Method
                f_surfvert(j,i,1) = f_cc(j,i-1,1);
                f_surfvert(j,i,3) = f_cc(j,i-1,3);
                f_surfvert(j,i,5) = f_cc(j,i-1,5);
                f_surfvert(j,i,6) = f_cc(j,i-1,6);
                f_surfvert(j,i,9) = f_cc(j,i-1,9);
    
                % bounce-back
                f_surfvert(j,i,2) = f_cc(j,i-1,2);
                f_surfvert(j,i,4) = f_cc(j,i-1,2);
    
                % no-slip
                f_surfvert(j,i,8) = f_surfvert(j,i,6)+(f_surfvert(j,i,3)-f_surfvert(j,i,5))/2;
                f_surfvert(j,i,7) = f_surfvert(j,i,6)+f_surfvert(j,i,9)-f_surfvert(j,i,8);
                %}
            else % all vertical cell surfaces
                % avg cells
                f_surfvert(j,i,1) = (f_cc(j,i-1,1)+f_cc(j,i,1))/2;
                f_surfvert(j,i,3) = (f_cc(j,i-1,3)+f_cc(j,i,3))/2;
                f_surfvert(j,i,5) = (f_cc(j,i-1,5)+f_cc(j,i,5))/2;
                
                % left cell
                f_surfvert(j,i,2) = f_cc(j,i-1,2);
                f_surfvert(j,i,6) = f_cc(j,i-1,6);
                f_surfvert(j,i,9) = f_cc(j,i-1,9);
    
                % right cell
                f_surfvert(j,i,4) = f_cc(j,i,4);
                f_surfvert(j,i,7) = f_cc(j,i,7);
                f_surfvert(j,i,8) = f_cc(j,i,8);
            end
        end
    end
    
    % horizontal surfaces
    for j = 1:N_y+1
        for i = 1:N_x
            if j == 1 % top boundary (moving lid)
                % Defining Local Env. cuz I'm lazy :)
                f_middle = squeeze(f_cc(j,i,:))-w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
                rho_middle = rho(j,i);
                if i == 1   % left corner
                    rho_left = rho(j,i);
                    rho_right = rho(j,i+1);
                    f_left = squeeze(f_cc(j,i,:))-w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
                    f_right = squeeze(f_cc(j,i+1,:))-w'.*rho(j,i+1).*(1+(Ksi'*[u(j,i+1);v(j,i+1)])/(c_s^2)+(Ksi'*[u(j,i+1);v(j,i+1)]).^2/(2*c_s^4)-([u(j,i+1),v(j,i+1)]*[u(j,i+1);v(j,i+1)])/(2*c_s^2));
                elseif i==N_x   %right corner
                    rho_left = rho(j,i-1);
                    rho_right = rho(j,i);
                    f_left = squeeze(f_cc(j,i-1,:))-w'.*rho(j,i-1).*(1+(Ksi'*[u(j,i-1);v(j,i-1)])/(c_s^2)+(Ksi'*[u(j,i-1);v(j,i-1)]).^2/(2*c_s^4)-([u(j,i-1),v(j,i-1)]*[u(j,i-1);v(j,i-1)])/(2*c_s^2));
                    f_right = squeeze(f_cc(j,i,:))-w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
                else
                    rho_left = rho(j,i-1);
                    rho_right = rho(j,i+1);
                    f_left = squeeze(f_cc(j,i-1,:))-w'.*rho(j,i-1).*(1+(Ksi'*[u(j,i-1);v(j,i-1)])/(c_s^2)+(Ksi'*[u(j,i-1);v(j,i-1)]).^2/(2*c_s^4)-([u(j,i-1),v(j,i-1)]*[u(j,i-1);v(j,i-1)])/(2*c_s^2));
                    f_right = squeeze(f_cc(j,i+1,:))-w'.*rho(j,i+1).*(1+(Ksi'*[u(j,i+1);v(j,i+1)])/(c_s^2)+(Ksi'*[u(j,i+1);v(j,i+1)]).^2/(2*c_s^4)-([u(j,i+1),v(j,i+1)]*[u(j,i+1);v(j,i+1)])/(2*c_s^2));
                end
                
                % Node 1
                rho_node = .5*(rho_middle+rho_left);
                u_node = [U_lid,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_left);
                f_nodes(1,:) = f_eq_node+squeeze(f_neq_node);
                % Node 2
                rho_node = .5*(rho_middle+rho_right);
                u_node = [U_lid,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_right);
                f_nodes(2,:) = f_eq_node+squeeze(f_neq_node);

                % temp_surface value
                f_surfhorz(j,i,:) = mean(f_nodes,1);

                % defining ghost cell
                for k = 1:9
                    f_ghost(k) = 2*f_surfhorz(j,i,k)-f_cc(j,i,k);
                end
                % final surface value
                %avg cells
                f_surfhorz(j,i,1) = (f_ghost(1)+f_cc(j,i,1))/2;
                f_surfhorz(j,i,2) = (f_ghost(2)+f_cc(j,i,2))/2;
                f_surfhorz(j,i,4) = (f_ghost(4)+f_cc(j,i,4))/2;
    
                % top cell
                f_surfhorz(j,i,5) = f_ghost(5);
                f_surfhorz(j,i,8) = f_ghost(8);
                f_surfhorz(j,i,9) = f_ghost(9);
    
                % bot cell
                f_surfhorz(j,i,3) = f_cc(j,i,3);
                f_surfhorz(j,i,6) = f_cc(j,i,6);
                f_surfhorz(j,i,7) = f_cc(j,i,7);
                %{
                f_surfhorz(j,i,1) = f_cc(j,i,1);
                f_surfhorz(j,i,2) = f_cc(j,i,2);
                f_surfhorz(j,i,4) = f_cc(j,i,4);
                f_surfhorz(j,i,6) = f_cc(j,i,6);
                f_surfhorz(j,i,7) = f_cc(j,i,7);
        
                % bounce-back
                f_surfhorz(j,i,3) = f_cc(j,i,3);
                f_surfhorz(j,i,5) = f_cc(j,i,3);
                
                % no-slip moving lid
                f_surfhorz(j,i,8) = f_surfhorz(j,i,6)+(f_surfhorz(j,i,2)-f_surfhorz(j,i,4)-rho(j,i)*U_lid)/2;
                f_surfhorz(j,i,9) = f_surfhorz(j,i,6)+f_surfhorz(j,i,7)-f_surfhorz(j,i,8);
                %}
            elseif j == N_y+1 % bottom boundary
                % Defining Local Env. cuz I'm lazy :)
                f_middle = squeeze(f_cc(j-1,i,:))-w'.*rho(j-1,i).*(1+(Ksi'*[u(j-1,i);v(j-1,i)])/(c_s^2)+(Ksi'*[u(j-1,i);v(j-1,i)]).^2/(2*c_s^4)-([u(j-1,i),v(j-1,i)]*[u(j-1,i);v(j-1,i)])/(2*c_s^2));
                rho_middle = rho(j-1,i);
                if i == 1   % left corner
                    rho_left = rho(j-1,i);
                    rho_right = rho(j-1,i+1);
                    f_left = squeeze(f_cc(j-1,i,:))-w'.*rho(j-1,i).*(1+(Ksi'*[u(j-1,i);v(j-1,i)])/(c_s^2)+(Ksi'*[u(j-1,i);v(j-1,i)]).^2/(2*c_s^4)-([u(j-1,i),v(j-1,i)]*[u(j-1,i);v(j-1,i)])/(2*c_s^2));
                    f_right = squeeze(f_cc(j-1,i+1,:))-w'.*rho(j-1,i+1).*(1+(Ksi'*[u(j-1,i+1);v(j-1,i+1)])/(c_s^2)+(Ksi'*[u(j-1,i+1);v(j-1,i+1)]).^2/(2*c_s^4)-([u(j-1,i+1),v(j-1,i+1)]*[u(j-1,i+1);v(j-1,i+1)])/(2*c_s^2));
                elseif i==N_x   %right corner
                    rho_left = rho(j-1,i-1);
                    rho_right = rho(j-1,i);
                    f_left = squeeze(f_cc(j-1,i-1,:))-w'.*rho(j-1,i-1).*(1+(Ksi'*[u(j-1,i-1);v(j-1,i-1)])/(c_s^2)+(Ksi'*[u(j-1,i-1);v(j-1,i-1)]).^2/(2*c_s^4)-([u(j-1,i-1),v(j-1,i-1)]*[u(j-1,i-1);v(j-1,i-1)])/(2*c_s^2));
                    f_right = squeeze(f_cc(j-1,i,:))-w'.*rho(j-1,i).*(1+(Ksi'*[u(j-1,i);v(j-1,i)])/(c_s^2)+(Ksi'*[u(j-1,i);v(j-1,i)]).^2/(2*c_s^4)-([u(j-1,i),v(j-1,i)]*[u(j-1,i);v(j-1,i)])/(2*c_s^2));
                else
                    rho_left = rho(j-1,i-1);
                    rho_right = rho(j-1,i+1);
                    f_left = squeeze(f_cc(j-1,i-1,:))-w'.*rho(j-1,i-1).*(1+(Ksi'*[u(j-1,i-1);v(j-1,i-1)])/(c_s^2)+(Ksi'*[u(j-1,i-1);v(j-1,i-1)]).^2/(2*c_s^4)-([u(j-1,i-1),v(j-1,i-1)]*[u(j-1,i-1);v(j-1,i-1)])/(2*c_s^2));
                    f_right = squeeze(f_cc(j-1,i+1,:))-w'.*rho(j-1,i+1).*(1+(Ksi'*[u(j-1,i+1);v(j-1,i+1)])/(c_s^2)+(Ksi'*[u(j-1,i+1);v(j-1,i+1)]).^2/(2*c_s^4)-([u(j-1,i+1),v(j-1,i+1)]*[u(j-1,i+1);v(j-1,i+1)])/(2*c_s^2));
                end
                
                % Node 1
                rho_node = .5*(rho_middle+rho_left);
                u_node = [0,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_left);
                f_nodes(1,:) = f_eq_node+squeeze(f_neq_node);
                % Node 2
                rho_node = .5*(rho_middle+rho_right);
                u_node = [0,0];
                f_eq_node = w'.*rho_node.*(1+(Ksi'*u_node')/(c_s^2)+(Ksi'*u_node').^2/(2*c_s^4)-(u_node*u_node')/(2*c_s^2));
                f_neq_node = .5.*(f_middle+f_right);
                f_nodes(2,:) = f_eq_node+squeeze(f_neq_node);

                % temp_surface value
                f_surfhorz(j,i,:) = mean(f_nodes,1);

                % defining ghost cell
                for k = 1:9
                    f_ghost(k) = 2*f_surfhorz(j,i,k)-f_cc(j-1,i,k);
                end
                % final surface value
                %avg cells
                f_surfhorz(j,i,1) = (f_cc(j-1,i,1)+f_ghost(1))/2;
                f_surfhorz(j,i,2) = (f_cc(j-1,i,2)+f_ghost(2))/2;
                f_surfhorz(j,i,4) = (f_cc(j-1,i,4)+f_ghost(4))/2;
    
                % top cell
                f_surfhorz(j,i,5) = f_cc(j-1,i,5);
                f_surfhorz(j,i,8) = f_cc(j-1,i,8);
                f_surfhorz(j,i,9) = f_cc(j-1,i,9);
    
                % bot cell
                f_surfhorz(j,i,3) = f_ghost(3);
                f_surfhorz(j,i,6) = f_ghost(6);
                f_surfhorz(j,i,7) = f_ghost(7);
                %{
                f_surfhorz(j,i,1) = f_cc(j-1,i,1);
                f_surfhorz(j,i,2) = f_cc(j-1,i,2);
                f_surfhorz(j,i,4) = f_cc(j-1,i,4);
                f_surfhorz(j,i,8) = f_cc(j-1,i,8);
                f_surfhorz(j,i,9) = f_cc(j-1,i,9);
        
                % bounce-back
                f_surfhorz(j,i,5) = f_cc(j-1,i,5);
                f_surfhorz(j,i,3) = f_cc(j-1,i,5);
    
                % no-slip
                f_surfhorz(j,i,6) = f_surfhorz(j,i,8)+(f_surfhorz(j,i,4)-f_surfhorz(j,i,2))/2;
                f_surfhorz(j,i,7) = f_surfhorz(j,i,8)+f_surfhorz(j,i,9)-f_surfhorz(j,i,6);
                %}
            else % all horizontal cell surfaces
                %avg cells
                f_surfhorz(j,i,1) = (f_cc(j-1,i,1)+f_cc(j,i,1))/2;
                f_surfhorz(j,i,2) = (f_cc(j-1,i,2)+f_cc(j,i,2))/2;
                f_surfhorz(j,i,4) = (f_cc(j-1,i,4)+f_cc(j,i,4))/2;
    
                % top cell
                f_surfhorz(j,i,5) = f_cc(j-1,i,5);
                f_surfhorz(j,i,8) = f_cc(j-1,i,8);
                f_surfhorz(j,i,9) = f_cc(j-1,i,9);
    
                % bot cell
                f_surfhorz(j,i,3) = f_cc(j,i,3);
                f_surfhorz(j,i,6) = f_cc(j,i,6);
                f_surfhorz(j,i,7) = f_cc(j,i,7);
            end
        end
    end

    %% Calculating Temporal Term
    for j = 1:N_y
        for i = 1:N_x
            %% Flux Term (surface in order of NWSE)
            f_surfs = squeeze([f_surfhorz(j,i,:), f_surfvert(j,i,:), f_surfhorz(j+1,i,:), f_surfvert(j,i+1,:)]);
            sum_F = zeros(1,9);
            for k = 1:4
                temp_F = zeros(1,9);
                for l = 1:9
                    temp_F(l) = f_surfs(k,l)*dot(Ksi(:,l),n(:,k));
                end
                sum_F = sum_F+temp_F;
            end
            %fprintf('flux term(%d,%d): %s\n',i,j,string(mat2str(sum_F)));
            %% Collision Term
            f_eq = w'.*rho(j,i).*(1+(Ksi'*[u(j,i);v(j,i)])/(c_s^2)+(Ksi'*[u(j,i);v(j,i)]).^2/(2*c_s^4)-([u(j,i),v(j,i)]*[u(j,i);v(j,i)])/(2*c_s^2));
            %fprintf('coll term(%d,%d): %s\n',i,j,string(mat2str(f_eq)));
            %% New Term
            f_cc(j,i,:) = squeeze(f_cc(j,i,:))+.1*(Tau^-1*(f_eq-squeeze(f_cc(j,i,:)))-sum_F');
            %fprintf('new term(%d,%d): %s\n',i,j,string(mat2str(squeeze(f_cc(j,i,:)))));
            %% Moment Calculation
            rho(j,i) = sum(f_cc(j,i,:));
            temp_u=sum(Ksi.*squeeze(f_cc(j,i,:))',2)/rho(j,i);
            u(j,i) = temp_u(1);
            v(j,i) = temp_u(2);
        end
    end
end

%% Post Processing
save('DBM.mat'); %saves iteration
figure(1);
quiver(flipud(u),flipud(v),10);
hold on;
axis equal tight;
hold off;

%% Benchmark Comparison
dbm.y_norm = linspace(1,0,N_y);
dbm.x_norm = linspace(0,1,N_x);
dbm.u_line = mean(u(:,50:51),2)/U_lid;
dbm.v_line = mean(v(50:51,:),1)/U_lid;
save('DBMBenchmark.mat','dbm')
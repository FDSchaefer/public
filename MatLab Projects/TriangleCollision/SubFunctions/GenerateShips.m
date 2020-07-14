function [Z,Tx,Ty,Vx,Vy] = GenerateShips(no)

%Scale PlayZone With number
Z = round(sqrt(no)/1.5);


%Base Triangle at Deg = 0
T0 = [0;0]; T1 = [0;0.5]; T2 = [1;0.25];
T = [T0,T1,T2]; Tc = mean(T,2); %Centre Of Triangle

Tx = zeros(3,no); Ty = zeros(3,no); V = zeros(2,no);

for i = 1:no
    
    l = (rand(2)-0.5)*2*0.95*Z;
    d = rand(1)*2*pi;
    R = [cos(d),-sin(d);sin(d),cos(d)];
    R0 = R*(Tc-T0); R1 = R*(Tc-T1); R2 = R*(Tc-T2);
    Tx(:,i) = [R0(1);R1(1);R2(1)]+l(1); Ty(:,i) = [R0(2);R1(2);R2(2)]+l(2);
    V(:,i) = [d,rand(1)/5];
    
end

%Convert Velocity to cartesian
Vx = V(2,:).*cos(V(1,:)) .* ones(3,no);
Vy = V(2,:).*sin(V(1,:)) .* ones(3,no);

end

function RunSimulation(no,manual)



[Z,Tx,Ty,Vx,Vy] = GenerateShips(no);
Trig = zeros(no,1);
C = zeros(no,1,3); C(:,1,2) = 1;

figure
p = patch(Tx,Ty,C);
xlim([-Z Z]); ylim([-Z Z]);
%set(gca,'xtick',[],'ytick',[]); 
set(gca,'xticklabel',[],'yticklabel',[])
grid on


pause(1)

for i = 1:200
    
    Tx = Tx - Vx; Ty = Ty-Vy;

    %Run Collision Check
    [Z,Tx,Ty,Vx,Vy,Trig,C] = CollideCheck(Z,Tx,Ty,Vx,Vy,Trig,C);
    
    delete(p);
    p = patch(Tx,Ty,C);
    pause(1/30)
    drawnow
    
    if manual
        pause
    end
end




end
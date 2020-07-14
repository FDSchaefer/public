function [Z,Tx,Ty,Vx,Vy,Trig,C] = CollideCheck(Z,Tx,Ty,Vx,Vy,Trig,C)


%Check each ship according to parameters against all others
for i = 1:length(Tx)
    
    if Trig(i) == 1 || Trig(i) == 2
        Trig(i) = 2;
        continue
    end
    
    %Start Proper Colide Check
    %First Check if any points outside the bounds of play area
    if BoundCheck(Z,Tx(:,i),Ty(:,i))
        Trig(i) =  1;
        continue
    end
    
    %Point Check (BaryoCentric)
    %If any ships have points within this one
    if Baryo2D(Tx,Ty,i)
        Trig(i) =  1;
        continue
    end
    
    
    %Line Check
    %If any ships have lines crossing with this one
        if Line2D(Tx,Ty,i)
            Trig(i) =  1;
            continue
        end
  
    
end

for i = 1:length(Tx)
    
	%If nothing happend add to output
    if Trig(i) == 0
        continue
    end    
    
	%If recently collided set speed to 0
    if Trig(i) == 1
        Vx(:,i) = 0; Vy(:,i) = 0;
        C(i,1,:) = [1 0 0];
        continue
    end
    
    %Check if past collide, aka destroy this update
    if Trig(i) == 2
        Tx(:,i) = Z*1.01; Ty(:,i) = Z*1.01;
        Vx(:,i) = 0; Vy(:,i) = 0;
        continue
    end
    
end



end



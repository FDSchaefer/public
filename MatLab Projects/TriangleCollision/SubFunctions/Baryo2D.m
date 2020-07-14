function Return = Baryo2D(Tx,Ty,sel)

Return = 0;

A = [Tx(1,sel),Ty(1,sel)];
B = [Tx(2,sel),Ty(2,sel)];
C = [Tx(3,sel),Ty(3,sel)];

X = Tx; X(:,sel) = []; X = X(:);
Y = Ty; Y(:,sel) = []; Y = Y(:);

Points = [X,Y];

NO = size(Points);
Area = @(a,b,c) ((a(1)*(b(2)-c(2)) + b(1)*(c(2)-a(2)) +c(1)*(a(2)-b(2)))/2);

for i = 1:NO(1)
    
    P = Points(i,:);
    
    ABC = Area(A,B,C); CAP = Area(C,A,P); ABP = Area(A,B,P);
    
    u = CAP/ABC; v = ABP/ABC;
    
    if u>=0 && v>= 0 && u+v<= 1
        Return = 1;
        break
        
    end
    
end


end


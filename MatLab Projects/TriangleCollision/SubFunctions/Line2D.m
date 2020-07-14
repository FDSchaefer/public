function Collision = Line2D(Tx,Ty,sel)
Collision = false;
%Select Tested Triangle
Test = [Tx(:,sel),Ty(:,sel)];

X = Tx; X(:,sel) = [];
Y = Ty; Y(:,sel) = [];

for i = 1:length(X)
    
    Check = [X(:,i),Y(:,i)];
    Distance = mean(Check,1) - mean(Test,1);
    Distance = sqrt(Distance(1)^2 +Distance(2)^2);
    if Distance < 1 %If the triangles are close, then check for line intersect
        
        for j = 1:3 %For each of the Tested triangles line
            
            if j > 2
                p1 = Test(3,:);
                p2 = Test(1,:);
            else
                p1 = Test(j,:);
                p2 = Test(j+1,:);
            end
            
            for k = 1:3 %For each of the checked triangles lines
                
                if k > 2
                    v1 = Check(3,:);
                    v2 = Check(1,:);
                else
                    v1 = Check(k,:);
                    v2 = Check(k+1,:);
                end
                  
                
                %Form Difference Vector
                P = p2 - p1;
                V = v2 - v1;
                D = p1 - v1;
                
                PP = dot(P,P);
                PV = dot(P,V);
                VV = dot(V,V);
                PD = dot(P,D);
                VD = dot(V,D);
                
                disc = dot(PP,VV) - dot(PV,PV);
                
                
                if (disc == 0)
                    Collision = false;
                else
                    %If not Run Solve
                    Ua = (dot(PV,VD) - dot(VV,PD)) / disc;
                    Ub = (dot(PP,VD) - dot(PV,PD)) / disc;
                    
                    %IF solve fits requirements
                    if (Ua <= 1 && Ua >= 0 && Ub <= 1 && Ub >= 0)
                        
                        %Check Distance Between "Intersect Points"
                        x = p1 + P * Ua;
                        y = v1 + V  *Ub;
                        Dist = (x - y);
                        
                        %If True, Return True
                        if (Dist == 0)
                            Collision = true;
                            return
                        else
                            Collision = false;
                        end
                        
                    else
                        Collision = false;
                    end
                end
                
            end
        end
    end
    
end
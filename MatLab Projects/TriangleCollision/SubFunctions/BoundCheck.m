function Return = BoundCheck(Z,Tx,Ty)

P = [Tx;Ty];

if sum(abs(P) >= Z ) > 0
    Return = 1;
else
    Return = 0;
end

end
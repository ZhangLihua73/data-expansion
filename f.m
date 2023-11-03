function dydt = f(t, y, c1, c2, c3)
   dydt = [y(2); c1*(y(2)+c2)./(y(1)+c3)];
end
